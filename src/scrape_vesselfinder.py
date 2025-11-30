import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

# ============================================================
# PATHS / CONFIG
# ============================================================

TRACKED_IMOS_PATH = Path("data/tracked_imos.json")
VESSELS_STATE_PATH = Path("data/vessels_data.json")
PORTS_PATH = Path("data/ports.json")

DEFAULT_PORTS = {
    "LAAYOUNE": {"lat": 27.1536, "lon": -13.2033},
    "TAN TAN": {"lat": 28.4927, "lon": -11.3437},
    "TARFAYA": {"lat": 27.9373, "lon": -12.9221},
    "DAKHLA": {"lat": 23.7048, "lon": -15.9336},
}

CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE") or "212663401022"
CALLMEBOT_API_KEY = os.getenv("CALLMEBOT_API_KEY") or "9206809"

FRESH_SIGNAL_MINUTES = 30
MIN_MOVE_NM = 2.0
ARRIVAL_RADIUS_NM = 1.0
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
        print("[WARN] CallMeBot not configured:")
        print(msg)
        return

    url = (
        "https://api.callmebot.com/whatsapp.php"
        f"?phone={CALLMEBOT_PHONE}&text={requests.utils.quote(msg)}&apikey={CALLMEBOT_API_KEY}"
    )

    try:
        r = requests.get(url, timeout=10)
        print(f"[INFO] WhatsApp status: {r.status_code}")
    except Exception as e:
        print("[ERROR] WhatsApp alert failed:", e)


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
# TEMP SCRAPER STUB (for testing)
# ============================================================

def scrape_vesselfinder(imo: str) -> dict:
    """
    TEMPORARY STUB for testing.
    Returns fake but realistic AIS data so we can test workflow,
    state logic, distance, nearest port, WhatsApp alerts.
    """
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    print(f"[STUB] Returning fake AIS for IMO {imo} at {now_utc}")

    return {
        "imo": imo,
        "name": f"TEST VESSEL {imo}",
        "lat": 27.20,
        "lon": -13.40,
        "sog": 11.5,
        "cog": 300.0,
        "last_pos_utc": now_utc,
        "destination": "DAKHLA",
    }


# ============================================================
# ALERT + STATE LOGIC
# ============================================================

def compute_age_minutes(last_utc: str | None) -> float | None:
    if not last_utc:
        return None
    try:
        dt = datetime.fromisoformat(last_utc.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60
    except Exception:
        return None


def detect_changes_and_message(v: dict, ports: dict, previous: dict):
    name = v.get("name", "UNKNOWN")
    imo = v.get("imo")
    lat = v.get("lat")
    lon = v.get("lon")
    sog = v.get("sog") or 0.0
    cog = v.get("cog") or 0.0
    dest_text = v.get("destination") or ""
    last_utc = v.get("last_pos_utc")

    if lat is None or lon is None:
        return previous, None

    age_mins = compute_age_minutes(last_utc)
    if age_mins is None or age_mins > FRESH_SIGNAL_MINUTES:
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

    near_port, near_dist = nearest_port(lat, lon, ports)

    dest_port = match_destination_port(dest_text, ports)
    if dest_port:
        dest_dist = distance_nm(lat, lon, ports[dest_port]["lat"], ports[dest_port]["lon"])
    else:
        dest_dist = None

    arrived = (
        dest_port is not None
        and dest_dist is not None
        and dest_dist <= ARRIVAL_RADIUS_NM
        and sog <= ARRIVAL_MAX_SOG
    )

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

    moved = False
    if "lat" in previous and "lon" in previous:
        prev_move = distance_nm(previous["lat"], previous["lon"], lat, lon)
        if prev_move >= MIN_MOVE_NM:
            moved = True
    else:
        moved = True

    dest_dist_changed = False
    if dest_dist is not None:
        prev_dest_dist = previous.get("dest_dist_nm")
        if prev_dest_dist is None or abs(dest_dist - prev_dest_dist) >= MIN_MOVE_NM:
            dest_dist_changed = True

    near_port_changed = near_port != previous.get("nearest_port")
    arrival_changed = arrived and not previous.get("done")

    should_alert = moved or dest_dist_changed or near_port_changed or arrival_changed

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
        if arrival_changed:
            lines.append(">>> ARRIVED / MOORED at destination — alerts stopped.")
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
    imos = load_tracked_imos()
    print("Tracked IMOs:", imos)

    if not imos:
        print("No IMOs tracked.")
        return

    ports_file = load_json(PORTS_PATH, {})
    ports = {**DEFAULT_PORTS, **ports_file}

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

    save_json(VESSELS_STATE_PATH, new_data)
    print("Saved vessels_data.json ✔")


if __name__ == "__main__":
    main()
