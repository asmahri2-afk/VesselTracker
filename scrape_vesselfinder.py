import json
import math
import os
import re
import requests
import requests.utils
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
CALLMEBOT_API_URL = "https://api.callmebot.com/whatsapp.php" # Added constant

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
    """Loads JSON data from a file, returning default on error or if file is missing."""
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load JSON from {path}: {e}. Returning default.")
        return default


def save_json(path: Path, data):
    """Saves data to a JSON file, creating the directory if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[ERROR] Failed to write to {path}: {e}")


def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    """
    Great-circle distance in nautical miles.
    
    CRITICAL FIX: Corrected lon2_r calculation to use lon2 instead of lat2.
    """
    R_km = 6371.0  # Earth radius in kilometers
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2) # <<< FIX: Must use lon2 here

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
    Loads ports from ports.json or uses a hardcoded fallback if missing.
    """
    ports = load_json(PORTS_PATH, {})
    if ports:
        return {k.upper(): v for k, v in ports.items()}

    # Fallback if ports.json is missing
    print("[WARN] Using hardcoded fallback ports.")
    return {
        "LAAYOUNE": {"lat": 27.1536, "lon": -13.2033},
        "TAN TAN": {"lat": 28.4927, "lon": -11.3437},
        "TARFAYA": {"lat": 27.9373, "lon": -12.9221},
        "DAKHLA": {"lat": 23.7048, "lon": -15.9336},
    }


def nearest_port(lat: float, lon: float, ports: dict):
    """Finds the nearest known port and distance (NM)."""
    best_name = None
    best_nm = None
    for name, coords in ports.items():
        # Ensure we are using the corrected Haversine here
        d_nm = haversine_nm(lat, lon, coords["lat"], coords["lon"])
        if best_nm is None or d_nm < best_nm:
            best_nm = d_nm
            best_name = name
    return best_name, best_nm


# ============================================================
# TIME / AIS AGE
# ============================================================

def parse_ais_time(s: str) -> datetime | None:
    """
    Parses the AIS time string (e.g., "Nov 30, 2025 17:07 UTC") into a UTC-aware datetime object.
    
    Revised to remove ' UTC' and explicitly set timezone for robustness.
    """
    if not s:
        return None
    s = s.strip()
    
    # Remove ' UTC' to parse the format cleanly without relying on %Z
    s_cleaned = s.replace(' UTC', '').strip()

    try:
        # Parse the date and time string
        dt_naive = datetime.strptime(s_cleaned, "%b %d, %Y %H:%M")
        
        # Make it explicitly UTC aware
        dt_aware = dt_naive.replace(tzinfo=timezone.utc)
        return dt_aware
        
    except Exception as e:
        print(f"[WARN] Failed to parse AIS timestamp '{s}': {e}")
        return None


def age_minutes(last_ts: str) -> float | None:
    """Calculates the age of the AIS message in minutes."""
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
    """Sends a message via CallMeBot, using requests' parameter encoding."""
    if not CALLMEBOT_ENABLED:
        print("[INFO] WhatsApp disabled or not configured.")
        return

    params = {
        "phone": CALLMEBOT_PHONE,
        "apikey": CALLMEBOT_APIKEY,
        "text": text
    }

    try:
        # requests handles the URL encoding of the 'text' parameter automatically
        r = requests.get(CALLMEBOT_API_URL, params=params, timeout=20)
        r.raise_for_status() # Raises HTTPError for bad status codes
        print(f"[INFO] WhatsApp status: {r.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] WhatsApp send failed: {e}")


# ============================================================
# SCRAPER – USE YOUR RENDER API
# ============================================================

def fetch_from_render_api(imo: str) -> dict:
    """
    Calls your Render API /vessel-full/{imo} and normalises the result.
    """
    url = f"{RENDER_BASE}/vessel-full/{imo}"
    print(f"[INFO] Fetching from API: {url}")

    try:
        # more generous timeout for cold / slow responses
        r = requests.get(url, timeout=60)
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

    # Normalize name and destination
    name = (data.get("name") or f"IMO {imo}").strip()
    destination = (data.get("destination") or "").strip()

    return {
        "imo": imo,
        "name": name,
        "lat": float(lat),
        "lon": float(lon),
        "sog": float(data.get("sog") or 0.0),
        "cog": float(data.get("cog") or 0.0),
        "last_pos_utc": data.get("last_pos_utc"),
        "destination": destination,
    }


# ============================================================
# ALERT LOGIC
# ============================================================

def match_destination_port(destination: str, ports: dict):
    """
    Try to match destination string (e.g. 'Tan Tan, Morocco')
    to a port in the known ports dictionary. Returns (port_name, coords_dict) or (None, None).
    """
    if not destination:
        return None, None

    dest_up = destination.upper()
    for port_name, info in ports.items():
        # Check if the port name is contained within the destination string
        if port_name in dest_up:
            return port_name, info

    return None, None


def build_alert_and_state(v: dict, ports: dict, prev_state: dict | None):
    """
    Determines if an alert is necessary based on movement, destination change, or arrival.
    Returns (alert_message: str | None, new_state: dict).
    """

    imo = v["imo"]
    name = v.get("name", f"IMO {imo}")
    lat = v["lat"]
    lon = v["lon"]
    sog = v["sog"]
    cog = v["cog"]
    last_pos_utc = v.get("last_pos_utc")
    destination = v["destination"]

    # AIS age calculation
    ais_age_min = age_minutes(last_pos_utc) if last_pos_utc else None
    if ais_age_min is not None and ais_age_min > MAX_AIS_MINUTES:
        print(f"[INFO] IMO {imo}: AIS {ais_age_min:.1f} min old (>{MAX_AIS_MINUTES}), still updating state.")
    ais_age_text = "N/A" if ais_age_min is None else f"{ais_age_min:.0f} min ago"

    # Nearest port (fallback distance)
    nearest_name, nearest_nm = nearest_port(lat, lon, ports)
    nearest_name_disp = nearest_name.upper() if nearest_name else "N/A"
    nearest_dist_text = "N/A"
    if nearest_nm is not None:
        nearest_dist_text = f"{nearest_nm:.1f} NM"

    # Destination distance (match destination string to a known port)
    dest_port_name = None
    dest_distance_nm = None
    if destination:
        dp_name, dp_coords = match_destination_port(destination, ports)
        if dp_name and dp_coords:
            dest_port_name = dp_name
            dest_distance_nm = haversine_nm(lat, lon, dp_coords["lat"], dp_coords["lon"])

    # Build new state snapshot (saved into vessels_data.json)
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
        "destination_port": dest_port_name,
        "destination_distance_nm": dest_distance_nm,
        "done": False,
    }

    # Arrival detection: within radius AND low speed (0.5 knots)
    arrived = False
    if nearest_nm is not None and nearest_nm <= ARRIVAL_RADIUS_NM and sog <= 0.5:
        arrived = True
        new_state["done"] = True

    # -------- FIRST TIME SEEN --------
    if not prev_state:
        status_line = "First tracking detected"
        dest_line = f"🎯 Destination: {destination or 'N/A'}"
        if dest_port_name and dest_distance_nm is not None:
            dest_line += f" (~{dest_distance_nm:.1f} NM to go)"

        lines = [
            f"🚢 {name} (IMO {imo})",
            f"📌 Status: {status_line}",
            f"🕒 AIS: {ais_age_text}",
            f"⚡ Speed: {sog:.1f} kn | 🧭 Course: {cog:.0f}°",
            f"📍 Position: {lat:.4f} , {lon:.4f}",
            f"⚓ Nearest port: {nearest_name_disp} (~{nearest_dist_text})",
            dest_line,
        ]
        msg = "\n".join(lines)
        return msg, new_state

    # Already marked as done -> no more alerts
    if prev_state.get("done"):
        return None, new_state

    # -------- CHANGE DETECTION --------

    # Destination change?
    old_dest_raw = (prev_state.get("destination") or "").strip()
    new_dest_raw = destination
    dest_changed = bool(old_dest_raw or new_dest_raw) and (
        old_dest_raw.upper() != new_dest_raw.upper()
    )

    # Movement?
    prev_lat = prev_state.get("lat")
    prev_lon = prev_state.get("lon")
    move_nm = None
    moved = False
    if prev_lat is not None and prev_lon is not None:
        move_nm = haversine_nm(prev_lat, prev_lon, lat, lon)
        moved = move_nm >= MIN_MOVE_NM

    # Arrival this run?
    arrival_event = arrived and not prev_state.get("done")

    if not (dest_changed or moved or arrival_event):
        return None, new_state

    # Status label
    if arrival_event:
        status_line = "Arrived at destination area"
    elif dest_changed:
        status_line = "Destination changed"
    elif moved:
        status_line = "Position / track updated"
    else:
        status_line = "Update"

    extra_move = ""
    if move_nm is not None and moved:
        extra_move = f" (Δ {move_nm:.1f} NM)"

    # Destination line with OLD ➜ NEW when changed
    if dest_changed and old_dest_raw:
        dest_line = f"🎯 Destination changed: {old_dest_raw} ➜ {destination or 'N/A'}"
    else:
        dest_line = f"🎯 Destination: {destination or 'N/A'}"

    if dest_port_name and dest_distance_nm is not None:
        dest_line += f" (~{dest_distance_nm:.1f} NM to go)"

    lines = [
        f"🚢 {name} (IMO {imo})",
        f"📌 Status: {status_line}",
        f"🕒 AIS: {ais_age_text}",
        f"⚡ Speed: {sog:.1f} kn | 🧭 Course: {cog:.0f}°{extra_move}",
        f"📍 Position: {lat:.4f} , {lon:.4f}",
        f"⚓ Nearest port: {nearest_name_disp} (~{nearest_dist_text})",
        dest_line,
    ]
    msg = "\n".join(lines)

    return msg, new_state


# ============================================================
# MAIN
# ============================================================

def main():

    # WARM-UP RENDER API (avoid cold-start timeouts)
    try:
        print("[INFO] Warming up Render API...")
        # Assuming RENDER_BASE is correct and '/ping' endpoint exists
        requests.get(f"{RENDER_BASE}/ping", timeout=10)
        print("[INFO] Render API awake ✔")
    except Exception as e:
        print(f"[WARN] Warm-up failed: {e}")

    # Load tracked IMOs
    imos = load_json(TRACKED_IMOS_PATH, [])
    
    # Simple check for list/dict structure (simplified from original logic)
    if isinstance(imos, dict) and "tracked_imos" in imos:
        imos = imos.get("tracked_imos", [])

    if not isinstance(imos, list):
        imos = []

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
            print(f"[WARN] Could not scrape IMO {imo}. Skipping update for this vessel.")
            # If API fails, keep old state for this vessel if it existed
            if imo in prev_all:
                new_all[imo] = prev_all[imo]
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
        # NOTE: This should only happen if all IMOs fail to fetch. If it happens,
        # we retain the last known good state to prevent loss of tracking history.
        print("[INFO] No valid vessel data, keeping existing vessels_data.json.")


# ============================================================

if __name__ == "__main__":
    main()
