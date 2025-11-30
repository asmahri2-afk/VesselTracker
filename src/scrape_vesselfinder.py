import json
import math
import os
import requests
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

TRACKED_IMOS_PATH = Path("data/tracked_imos.json")
VESSELS_STATE_PATH = Path("data/vessels_data.json")
PORTS_PATH = Path("data/ports.json")

# CallMeBot (env secrets in GitHub)
CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE")
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY")
CALLMEBOT_ENABLED = bool(CALLMEBOT_PHONE and CALLMEBOT_APIKEY)

print(f"[DEBUG] CALLMEBOT_PHONE: {'SET' if CALLMEBOT_PHONE else 'MISSING'}")
print(f"[DEBUG] CALLMEBOT_APIKEY: {'SET' if CALLMEBOT_APIKEY else 'MISSING'}")
print(f"[DEBUG] CALLMEBOT_ENABLED: {CALLMEBOT_ENABLED}")

# Your Render API
RENDER_BASE = "https://vessel-api-s85s.onrender.com"

# AIS age threshold in minutes
MAX_AIS_MINUTES = 30

# Distance threshold (NM) to consider “arrived” at destination
ARRIVAL_RADIUS_NM = 10.0

# Min movement (NM) to consider course/pos “changed”
MIN_MOVE_NM = 5.0


# ============================================================
# UTIL
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
        json.dump(data, f, indent=2, ensure_ascii=False)


def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    """
    Great-circle distance in nautical miles.
    """
    R_km = 6371.0
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    dist_km = R_km * c
    return dist_km * 0.539957  # km → NM


# ============================================================
# PORTS
# ============================================================

def load_ports() -> dict:
    """
    ports.json is expected to be like:
    {
      "LAAYOUNE": {"lat": 27.1536, "lon": -13.2033},
      ...
    }
    """
    ports = load_json(PORTS_PATH, {})
    if ports:
        return ports

    # Fallback if ports.json is missing
    return {
        "LAAYOUNE": {"lat": 27.1536, "lon": -13.2033},
        "TAN TAN": {"lat": 28.4927, "lon": -11.3437},
        "TARFAYA": {"lat": 27.9373, "lon": -12.9221},
        "DAKHLA": {"lat": 23.7048, "lon": -15.9336},
    }


def nearest_port(lat: float, lon: float, ports: dict):
    best_name = None
    best_nm = None
    for name, coords in ports.items():
        d_nm = haversine_nm(lat, lon, coords["lat"], coords["lon"])
        if best_nm is None or d_nm < best_nm:
            best_nm = d_nm
            best_name = name
    return best_name, best_nm


# ============================================================
# TIME / AIS AGE
# ============================================================

def parse_ais_time(s: str):
    """
    Example string from VF: "Nov 30, 2025 17:07 UTC"
    Returns aware datetime in UTC, or None.
    """
    if not s:
        return None
    s = s.strip()
    try:
        dt = datetime.strptime(s, "%b %d, %Y %H:%M %Z")
        # Assume UTC if no tzinfo
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def age_minutes(last_ts: str) -> float | None:
    dt = parse_ais_time(last_ts)
    if not dt:
        return None
    now = datetime.now(timezone.utc)
    delta = now - dt
    return delta.total_seconds() / 60.0


# ============================================================
# CALLMEBOT
# ============================================================

def send_whatsapp_message(text: str):
    if not CALLMEBOT_ENABLED:
        print("[INFO] WhatsApp disabled or not configured.")
        return

    # CallMeBot expects URL-encoded text
    url = (
        f"https://api.callmebot.com/whatsapp.php?"
        f"phone={CALLMEBOT_PHONE}"
        f"&apikey={CALLMEBOT_APIKEY}"
        f"&text={requests.utils.quote(text)}"
    )

    try:
        r = requests.get(url, timeout=20)
        print(f"[INFO] WhatsApp status: {r.status_code}")
    except Exception as e:
        print(f"[ERROR] WhatsApp send failed: {e}")


# ============================================================
# SCRAPER – USE YOUR RENDER API
# ============================================================

def fetch_from_render_api(imo: str) -> dict:
    """
    Calls your Render API /vessel-full/{imo} and normalises the result.
    Expected JSON (fields may be None):
    {
      "found": true,
      "imo": "6417499",
      "name": "NAVE SANTA MARIA",
      "lat": 29,
      "lon": -14,
      "sog": 0,
      "cog": 21,
      "last_pos_utc": "Nov 30, 2025 17:07 UTC",
      "destination": "Tan Tan, Morocco"
    }
    """
    url = f"{RENDER_BASE}/vessel-full/{imo}"
    print("Fetching from API:", url)

    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] API request failed for IMO {imo}: {e}")
        return {}

    data = r.json()

    if data.get("found") is False:
        print(f"[WARN] API: vessel not found for IMO {imo}")
        return {}

    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        print(f"[WARN] API: missing lat/lon for IMO {imo}")
        return {}

    return {
        "imo": imo,
        "name": data.get("name", f"IMO {imo}"),
        "lat": float(lat),
        "lon": float(lon),
        "sog": float(data.get("sog") or 0.0),
        "cog": float(data.get("cog") or 0.0),
        "last_pos_utc": data.get("last_pos_utc"),
        "destination": data.get("destination") or "",
    }


# ============================================================
# ALERT LOGIC
# ============================================================

def build_alert_and_state(v: dict, ports: dict, prev_state: dict | None):
    """
    Returns (alert_message: str | None, new_state: dict).
    - Skips if AIS is older than MAX_AIS_MINUTES.
    - Marks vessel as 'done' when arrived at destination and stopped.
    - Sends alert on:
        * first time we see the vessel
        * significant movement (>= MIN_MOVE_NM)
        * destination change
        * arrival event
    """
    imo = v["imo"]
    name = v.get("name", f"IMO {imo}")
    lat = v["lat"]
    lon = v["lon"]
    sog = v["sog"]
    cog = v["cog"]
    last_pos_utc = v.get("last_pos_utc")
    destination = v.get("destination", "")

    # AIS age filter
    ais_age_min = age_minutes(last_pos_utc) if last_pos_utc else None
    if ais_age_min is not None and ais_age_min > MAX_AIS_MINUTES:
        print(f"[INFO] IMO {imo}: AIS {ais_age_min:.1f} min old, skipping.")
        # still update state but no alert
        pass

    # Nearest port
    nearest_name, nearest_nm = nearest_port(lat, lon, ports)

    new_state = {
        "imo": imo,
        "name": name,
        "lat": lat,
        "lon": lon,
        "sog": sog,
        "cog": cog,
        "last_pos_utc": last_pos_utc,
        "destination": destination,
        "nearest_port": nearest_name,
        "nearest_distance_nm": nearest_nm,
        "done": False,
    }

    # Arrival detection
    arrived = False
    if nearest_nm is not None and nearest_nm <= ARRIVAL_RADIUS_NM and sog <= 0.5:
        arrived = True
        new_state["done"] = True

    # No previous state → first alert
    if not prev_state:
        status = "ARRIVED" if arrived else "FIRST TRACK"
        msg = (
            f"{status} {name} (IMO {imo}) | "
            f"AIS: {('?.?' if ais_age_min is None else f'{ais_age_min:.0f} min')} | "
            f"SOG: {sog:.1f} kn | COG: {cog:.0f}° | "
            f"POS: {lat:.4f}, {lon:.4f} | "
            f"Nearest port: {nearest_name} (~{nearest_nm:.1f} NM) | "
            f"Destination: {destination or 'N/A'}"
        )
        return msg, new_state

    # If already done (arrived earlier), do nothing
    if prev_state.get("done"):
        # Allow sending ONE final message when it first becomes done
        if arrived and not prev_state.get("done"):
            # (This case won't happen: done would be False before.)
            pass
        return None, new_state

    # Destination change?
    dest_changed = (destination or "") != (prev_state.get("destination") or "")

    # Movement?
    prev_lat = prev_state.get("lat")
    prev_lon = prev_state.get("lon")
    move_nm = None
    moved = False
    if prev_lat is not None and prev_lon is not None:
        move_nm = haversine_nm(prev_lat, prev_lon, lat, lon)
        moved = move_nm >= MIN_MOVE_NM

    # Arrival event (now arrived, before not done)
    arrival_event = arrived and not prev_state.get("done")

    if not (dest_changed or moved or arrival_event):
        # Nothing interesting
        return None, new_state

    # Build alert text
    parts = []
    if arrival_event:
        parts.append("ARRIVED")
    elif dest_changed:
        parts.append("DEST-CHG")
    elif moved:
        parts.append("MOVED")

    tag = "/".join(parts) if parts else "UPDATE"

    dist_str = f"{nearest_nm:.1f} NM" if nearest_nm is not None else "N/A"
    age_str = "?.?" if ais_age_min is None else f"{ais_age_min:.0f} min"
    move_str = "" if move_nm is None else f" | Δ={move_nm:.1f} NM"

    msg = (
        f"{tag} {name} (IMO {imo}) | "
        f"AIS: {age_str} | SOG: {sog:.1f} kn | COG: {cog:.0f}° | "
        f"POS: {lat:.4f}, {lon:.4f} | "
        f"Nearest port: {nearest_name} (~{dist_str}) | "
        f"Destination: {destination or 'N/A'}"
        f"{move_str}"
    )

    return msg, new_state


# ============================================================
# MAIN
# ============================================================

def main():
    # Load tracked IMOs
    tracked = load_json(TRACKED_IMOS_PATH, [])
    if isinstance(tracked, dict) and "tracked_imos" in tracked:
        imos = tracked.get("tracked_imos", [])
    else:
        imos = tracked

    imos = [str(i).strip() for i in imos if str(i).strip()]
    if not imos:
        print("No IMOs to track.")
        return

    print("Tracked IMOs:", imos)

    ports = load_ports()
    prev_all = load_json(VESSELS_STATE_PATH, {})
    if not isinstance(prev_all, dict):
        prev_all = {}

    new_all = {}

    for imo in imos:
        v = fetch_from_render_api(imo)
        if not v:
            print(f"[WARN] Could not scrape IMO {imo}")
            continue

        prev_state = prev_all.get(imo)
        alert, new_state = build_alert_and_state(v, ports, prev_state)
        new_all[imo] = new_state

        if alert:
            print("[ALERT]", alert)
            send_whatsapp_message(alert)

    # Save updated state (only if we have at least one vessel)
    print(f"[INFO] Built state for {len(new_all)} vessel(s).")
    if new_all:
        save_json(VESSELS_STATE_PATH, new_all)
        print("Saved vessels_data.json ✔")
    else:
        print("[INFO] No valid vessel data, keeping existing vessels_data.json (not overwriting with {}).")



# ============================================================

if __name__ == "__main__":
    main()
