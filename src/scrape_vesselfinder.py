import json
import math
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

TRACKED_IMOS_PATH = Path("data/tracked_imos.json")
VESSELS_STATE_PATH = Path("data/vessels_data.json")
PORTS_PATH = Path("data/ports.json")

# Default ports if ports.json is missing
DEFAULT_PORTS = {
    "LAAYOUNE": {"lat": 27.1536, "lon": -13.2033},
    "TAN TAN": {"lat": 28.4927, "lon": -11.3437},
    "TARFAYA": {"lat": 27.9373, "lon": -12.9221},
    "DAKHLA": {"lat": 23.7048, "lon": -15.9336},
}

# WhatsApp (prefer env, fallback to hard-coded for local tests)
CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE") or "212663401022"
CALLMEBOT_API_KEY = os.getenv("CALLMEBOT_API_KEY") or "9206809"

# AIS freshness threshold (min) – only report if last AIS < this
FRESH_SIGNAL_MINUTES = 30

# How much movement / distance change to consider "changed" (NM)
MIN_MOVE_NM = 2.0

# Distance to destination considered "arrived / moored" (NM)
ARRIVAL_RADIUS_NM = 1.0

# Max SOG to consider "moored"
ARRIVAL_MAX_SOG = 1.0


# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def load_tracked_imos() -> list[str]:
    data = load_json(TRACKED_IMOS_PATH, [])

    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]

    if isinstance(data, dict) and "tracked_imos" in data:
        lst = data.get("tracked_imos", [])
        if isinstance(lst, list):
            return [str(x).strip() for x in lst if str(x).strip()]

    return []


def distance_nm(lat1, lon1, lat2, lon2) -> float:
    R = 3440.065  # nautical miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def send_whatsapp_message(msg: str):
    if not CALLMEBOT_PHONE or not CALLMEBOT_API_KEY:
        print("[WARN] CallMeBot not configured, message below:")
        print(msg)
        return

    url = (
        "https://api.callmebot.com/whatsapp.php"
        f"?phone={CALLMEBOT_PHONE}&text={requests.utils.quote(msg)}&apikey={CALLMEBOT_API_KEY}"
    )
    try:
        requests.get(url, timeout=10)
        print(f"[ALERT SENT] {msg}")
    except Exception:
        print("[ERROR] WhatsApp alert failed.")


# ============================================================
# PORT / DESTINATION HELPERS
# ============================================================

def nearest_port(lat: float, lon: float, ports: dict):
    best_name = None
    best_dist = None
    for name, coords in ports.items():
        d = distance_nm(lat, lon, coords["lat"], coords["lon"])
        if best_dist is None or d < best_dist:
            best_dist = d
            best_name = name
    return best_name, best_dist


def match_destination_port(dest_text: str, ports: dict):
    if not dest_text:
        return None
    up = dest_text.upper()
    for name in ports.keys():
        if name in up:
            return name
    return None


# ============================================================
# SCRAPER FOR VESSELFINDER
# ============================================================

def scrape_vesselfinder(imo: str) -> dict:
    url = f"https://www.vesselfinder.com/vessels/details/{imo}"
    print("Fetching:", url)

    # Pretend to be a real browser so we don't get 403
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.vesselfinder.com/",
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)
    except Exception as e:
        print(f"[ERROR] Request failed for {imo}: {e}")
        return {}

    if r.status_code == 403:
        print(f"[WARN] HTTP 403 for {imo} (blocked by Vesselfinder)")
        return {}

    if not r.ok:
        print(f"[WARN] HTTP {r.status_code} for {imo}")
        return {}

    soup = BeautifulSoup(r.text, "html.parser")

    # ---- Vessel name ----
    title = soup.find("h1")
    name = title.text.strip() if title else "UNKNOWN"

    # ---- JSON-LD block ----
    json_blob = soup.find("script", {"type": "application/ld+json"})
    if not json_blob:
        print(f"[WARN] No JSON-LD for {imo}")
        return {}

    try:
        data = json.loads(json_blob.text)
    except Exception as e:
        print(f"[WARN] JSON-LD parse error for {imo}: {e}")
        return {}

    # Depending on page, data can be a list or dict
    if isinstance(data, list) and data:
        data = data[0]

    lat = float(data.get("latitude")) if data.get("latitude") is not None else None
    lon = float(data.get("longitude")) if data.get("longitude") is not None else None
    sog = float(data.get("speed")) if data.get("speed") is not None else 0.0
    cog = float(data.get("course")) if data.get("course") is not None else 0.0
    last_pos_utc = data.get("dateModified")          # e.g. "2025-01-01T12:34:00Z"
    arrival_destination = data.get("arrivalDestination", "")

    return {
        "imo": imo,
        "name": name,
        "lat": lat,
        "lon": lon,
        "sog": sog,
        "cog": cog,
        "last_pos_utc": last_pos_utc,
        "destination": arrival_destination,
    }



# ============================================================
# ALERT + STATE LOGIC
# ============================================================

def compute_age_minutes(last_utc: str | None) -> float | None:
    if not last_utc:
        return None
    try:
        dt = datetime.fromisoformat(last_utc.replace("Z", "+00:00"))
        age_mins = (datetime.now(timezone.utc) - dt).total_seconds() / 60
        return age_mins
    except Exception:
        return None


def detect_changes_and_message(v: dict, ports: dict, previous: dict):
    """
    Returns (new_state_dict, optional_message_str_or_None)
    """
    name = v.get("name", "UNKNOWN")
    imo = v.get("imo")
    lat, lon = v.get("lat"), v.get("lon")
    sog = v.get("sog") or 0.0
    cog = v.get("cog") or 0.0
    dest_text = v.get("destination") or ""
    last_utc = v.get("last_pos_utc")

    if lat is None or lon is None:
        return previous, None

    age_mins = compute_age_minutes(last_utc)
    if age_mins is None or age_mins > FRESH_SIGNAL_MINUTES:
        # AIS too old -> update state but do NOT alert
        new_state = {
            **previous,
            "name": name,
            "imo": imo,
            "lat": lat,
            "lon": lon,
            "sog": sog,
            "cog": cog,
            "destination_text": dest_text,
            "last_pos_utc": last_utc,
            "age_mins": age_mins,
        }
        return new_state, None

    # nearest port / distance
    near_port, near_dist = nearest_port(lat, lon, ports)

    # destination port / distance
    dest_port = match_destination_port(dest_text, ports)
    if dest_port:
        dest_dist = distance_nm(lat, lon, ports[dest_port]["lat"], ports[dest_port]["lon"])
    else:
        dest_dist = None

    # arrival detection (approximate "moored at destination")
    arrived = False
    if dest_port and dest_dist is not None:
        if dest_dist <= ARRIVAL_RADIUS_NM and sog <= ARRIVAL_MAX_SOG:
            arrived = True

    # if already done before, keep done=True and don't alert again
    if previous.get("done") and arrived:
        new_state = {
            **previous,
            "name": name,
            "imo": imo,
            "lat": lat,
            "lon": lon,
            "sog": sog,
            "cog": cog,
            "destination_text": dest_text,
            "last_pos_utc": last_utc,
            "age_mins": age_mins,
            "nearest_port": near_port,
            "nearest_port_nm": near_dist,
            "dest_port": dest_port,
            "dest_dist_nm": dest_dist,
            "done": True,
        }
        return new_state, None

    # movement / change tests
    moved = False
    if "lat" in previous and "lon" in previous:
        prev_d = distance_nm(previous["lat"], previous["lon"], lat, lon)
        if prev_d >= MIN_MOVE_NM:
            moved = True
    else:
        moved = True  # first time tracking

    dest_dist_changed = False
    if dest_dist is not None:
        prev_dest_dist = previous.get("dest_dist_nm")
        if prev_dest_dist is None or abs(dest_dist - prev_dest_dist) >= MIN_MOVE_NM:
            dest_dist_changed = True

    near_port_changed = near_port != previous.get("nearest_port")

    arrival_changed = arrived and not previous.get("done")

    should_alert = moved or dest_dist_changed or near_port_changed or arrival_changed

    # Build message if needed
    msg = None
    if should_alert:
        lines = [
            f"{name} (IMO {imo})",
            f"AIS: {age_mins:.0f} min ago | SOG: {sog:.1f} kn | COG: {cog:.0f}°",
            f"POS: {lat:.4f}, {lon:.4f}",
        ]
        if near_port and near_dist is not None:
            lines.append(f"Nearest port: {near_port} (~{near_dist:.1f} NM)")
        if dest_port and dest_dist is not None:
            lines.append(f"Destination: {dest_port} (~{dest_dist:.1f} NM to go)")
        elif dest_text:
            lines.append(f"Destination (raw): {dest_text}")

        if arrival_changed:
            lines.append(">>> ARRIVED / MOORED at destination – alerts stopped for this IMO.")

        msg = "\n".join(lines)

    new_state = {
        "name": name,
        "imo": imo,
        "lat": lat,
        "lon": lon,
        "sog": sog,
        "cog": cog,
        "destination_text": dest_text,
        "last_pos_utc": last_utc,
        "age_mins": age_mins,
        "nearest_port": near_port,
        "nearest_port_nm": near_dist,
        "dest_port": dest_port,
        "dest_dist_nm": dest_dist,
        "done": previous.get("done") or arrived,
    }

    return new_state, msg


# ============================================================
# MAIN
# ============================================================

def main():
    # Load tracked IMOs
    imos = load_tracked_imos()
    print("Tracked IMOs:", imos)

    # Load ports list (merged with defaults)
    ports_from_file = load_json(PORTS_PATH, {})
    ports = {**DEFAULT_PORTS, **ports_from_file}

    # Previous states (position / dist / done flag per IMO)
    old_data = load_json(VESSELS_STATE_PATH, {})

    new_data = {}

    for imo in imos:
        prev_state = old_data.get(imo, {})

        v = scrape_vesselfinder(imo)
        if not v:
            print(f"[WARN] Could not scrape IMO {imo}")
            continue

        vessel_state, message = detect_changes_and_message(v, ports, prev_state)
        new_data[imo] = vessel_state

        if message:
            print("[ALERT]", message.replace("\n", " | "))
            send_whatsapp_message(message)

    # Save updated vessels + state
    save_json(VESSELS_STATE_PATH, new_data)
    print("Saved vessels_data.json ✔")


if __name__ == "__main__":
    main()
