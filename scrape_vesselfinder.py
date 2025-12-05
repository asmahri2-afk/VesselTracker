import json
import math
import os
import re
import requests
import requests.utils
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

TRACKED_IMOS_PATH = Path("data/tracked_imos.json")
VESSELS_STATE_PATH = Path("data/vessels_data.json")
PORTS_PATH = Path("data/ports.json")
SHIPID_MAP_PATH = Path("data/shipid_map.json")

# Optional: small test of redirect + shipid extraction using httpbin
HTTPBIN_TEST_ENABLED = False  # <- set to True only for testing
HTTPBIN_TEST_URL = (
    "https://httpbin.org/redirect-to"
    "?url=https://www.marinetraffic.com/en/ais/details/ships/shipid:8338267"
)

# CallMeBot (env secrets in GitHub)
CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE")
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY")
CALLMEBOT_ENABLED = bool(CALLMEBOT_PHONE and CALLMEBOT_APIKEY)
CALLMEBOT_API_URL = "https://api.callmebot.com/whatsapp.php"  # Added constant

print(f"[DEBUG] CALLMEBOT_PHONE: {'SET' if CALLMEBOT_PHONE else 'MISSING'}")
print(f"[DEBUG] CALLMEBOT_APIKEY: {'SET' if CALLMEBOT_APIKEY else 'MISSING'}")
print(f"[DEBUG] CALLMEBOT_ENABLED: {CALLMEBOT_ENABLED}")
print("[DEBUG] ETA VERSION ACTIVE")

# Your Render API (VesselFinder-based)
RENDER_BASE = "https://vessel-api-s85s.onrender.com"

# AIS age threshold in minutes (for general info)
MAX_AIS_MINUTES = 30

# Distance threshold (NM) to consider “arrived” at destination
ARRIVAL_RADIUS_NM = 35.5

# Min movement (NM) to consider course/pos “changed”
MIN_MOVE_NM = 5.0

# ETA config
MIN_SOG_FOR_ETA = 0.5        # kn, below this we don't compute ETA
MAX_ETA_HOURS = 240          # ignore ETA if > 10 days
MAX_ETA_SOG_CAP = 18.0       # cap extreme AIS spikes
MAX_AIS_FOR_ETA_MIN = 360    # if AIS older than 6h, skip ETA
MIN_DISTANCE_FOR_ETA = 5.0   # NM, if closer than this, skip ETA


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
    """
    R_km = 6371.0  # Earth radius in kilometers
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
# PORTS + DESTINATION NORMALIZATION
# ============================================================

def load_ports() -> dict:
    """
    Loads ports from ports.json and returns a dict with upper-case keys.
    """
    ports = load_json(PORTS_PATH, {})

    if not ports:
        raise RuntimeError("ports.json is missing or empty!")

    return {k.upper(): v for k, v in ports.items()}


def _normalize_text(s: str) -> str:
    """
    Uppercase, remove accents, remove non-letters (spaces, dashes, dots…).
    Use this for BOTH ports.json keys and raw destinations.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.upper()
    s = re.sub(r"[^A-Z]", "", s)
    return s


ALIASES_RAW = {
    # ---------------------------
    # MOROCCO
    # ---------------------------

    # LAAYOUNE
    "laayoune": "LAAYOUNE",
    "layoune": "LAAYOUNE",
    "eh eun": "LAAYOUNE",   # common AIS form
    "leyoune": "LAAYOUNE",

    # TAN TAN
    "tantan": "TAN TAN",
    "tan tan": "TAN TAN",
    "tan-tan": "TAN TAN",
    "tan tan anch": "TAN TAN",   # no separate anch known

    # DAKHLA (PORT)
    "dakhla": "DAKHLA",
    "dakhla port": "DAKHLA",
    "ad dakhla": "DAKHLA",

    # DAKHLA ANCHORAGE
    "dakhla anch": "DAKHLA ANCH",
    "dakhla anch.": "DAKHLA ANCH",
    "dakhla anchorage": "DAKHLA ANCH",
    "dakhla anch area": "DAKHLA ANCH",

    # AGADIR
    "agadir": "AGADIR",
    "port agadir": "AGADIR",

    # ESSAOUIRA
    "essaouira": "ESSAOUIRA",

    # SAFI
    "safi": "SAFI",

    # CASABLANCA
    "casa": "CASABLANCA",           # AIS often uses CASA
    "casablanca": "CASABLANCA",
    "cassablanca": "CASABLANCA",

    # MOHAMMEDIA
    "mohammedia": "MOHAMMEDIA",

    # JORF
    "jorf": "JORF LASFAR",
    "jorf lasfar": "JORF LASFAR",

    # KENITRA
    "kenitra": "KENITRA",
    "kenitra": "KENITRA",

    # TANGER
    "tanger": "TANGER VILLE",
    "tangier": "TANGER VILLE",
    "tanger ville": "TANGER VILLE",
    "tanger med": "TANGER MED",
    "tm2": "TANGER MED",

    # NADOR
    "nador": "NADOR",

    # AL HOCEIMA
    "al hoceima": "AL HOCEIMA",
    "alhucemas": "AL HOCEIMA",  # Spanish form

    # ---------------------------
    # CANARY ISLANDS
    # ---------------------------

    # LAS PALMAS
    "las palmas": "LAS PALMAS",
    "lpa": "LAS PALMAS",
    "las palmas anch": "LAS PALMAS",

    # ARRECIFE
    "arrecife": "ARRECIFE",

    # PUERTO DEL ROSARIO
    "puerto del rosario": "PUERTO DEL ROSARIO",
    "pdr": "PUERTO DEL ROSARIO",

    # SANTA CRUZ TENERIFE
    "santa cruz": "SANTA CRUZ DE TENERIFE",
    "sctf": "SANTA CRUZ DE TENERIFE",
    "santa cruz tenerife": "SANTA CRUZ DE TENERIFE",

    # LA GOMERA
    "san sebastian": "SAN SEBASTIAN DE LA GOMERA",

    # EL HIERRO
    "la restinga": "LA RESTINGA",

    # LA PALMA
    "la palma": "LA PALMA",

    # GRANADILLA
    "granadilla": "GRANADILLA",
    "puerto de granadilla": "GRANADILLA",

    # ---------------------------
    # SPAIN MAINLAND
    # ---------------------------

    # CEUTA
    "ceuta": "CEUTA",

    # MELILLA
    "melilla": "MELILLA",

    # ALGECIRAS
    "algeciras": "ALGECIRAS",
    "alg": "ALGECIRAS",

    # GIBRALTAR
    "gibraltar": "GIBRALTAR",
    "gib": "GIBRALTAR",

    # HUELVA (PORT)
    "huelva": "HUELVA",

    # HUELVA ANCH
    "huelva anch": "HUELVA ANCH",
    "huelva anchorage": "HUELVA ANCH",

    # CADIZ
    "cadiz": "CADIZ",
    "cadiz anch": "CADZ",  # if you add an anch area later

    # SEVILLA
    "sevilla": "SEVILLA",
    "seville": "SEVILLA",

    # MALAGA
    "malaga": "MALAGA",

    # MOTRIL
    "motril": "MOTRIL",

    # ALMERIA
    "almeria": "ALMERIA",

    # CARTAGENA
    "cartagena": "CARTAGENA",

    # VALENCIA
    "valencia": "VALENCIA",

    # ---------------------------
    # PORTUGAL
    # ---------------------------

    "sines": "SINES",
    "setubal": "SETUBAL",
    "lisbon": "LISBON",
    "lisboa": "LISBON"
}

# Normalized aliases map
DEST_ALIASES = {
    _normalize_text(raw): canonical
    for raw, canonical in ALIASES_RAW.items()
}


def match_destination_port(destination: str, ports: dict):
    """
    Try to match destination string (e.g. 'TanTan', 'EH EUN', 'Las Palmas anch')
    to a port in the known ports dictionary.
    Returns (port_name, coords_dict) or (None, None).
    """
    if not destination:
        return None, None

    norm_dest = _normalize_text(destination)

    # 1) Exact alias match (covers 'tantan', 'EH EUN', etc.)
    if norm_dest in DEST_ALIASES:
        canonical_name = DEST_ALIASES[norm_dest]
        return canonical_name, ports.get(canonical_name)

    # Build canonical map for ports.json keys
    canonical_map = {_normalize_text(p): p for p in ports.keys()}

    # 2) Exact canonical match (e.g. "LAAYOUNE", "TANTAN" normalized)
    if norm_dest in canonical_map:
        name = canonical_map[norm_dest]
        return name, ports.get(name)

    # 3) Substring/fuzzy: handle 'LASPALMASANCH', 'HUELVAANCH', etc.
    for canon, name in canonical_map.items():
        if canon and canon in norm_dest:
            return name, ports.get(name)

    return None, None


def nearest_port(lat: float, lon: float, ports: dict):
    """Finds the nearest known port and distance (NM)."""
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

def parse_ais_time(s: str) -> datetime | None:
    """
    Parses the AIS time string (e.g., "Nov 30, 2025 17:07 UTC") into a UTC-aware datetime object.
    """
    if not s:
        return None
    s = s.strip()
    s_cleaned = s.replace(' UTC', '').strip()

    try:
        dt_naive = datetime.strptime(s_cleaned, "%b %d, %Y %H:%M")
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
        r = requests.get(CALLMEBOT_API_URL, params=params, timeout=20)
        r.raise_for_status()
        print(f"[INFO] WhatsApp status: {r.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] WhatsApp send failed: {e}")


# ============================================================
# SCRAPER – USE YOUR RENDER API (VESSELFINDER DATA)
# ============================================================

def fetch_from_render_api(imo: str) -> dict:
    """
    Calls your Render API /vessel-full/{imo} and normalises the result.
    """
    url = f"{RENDER_BASE}/vessel-full/{imo}"
    print(f"[INFO] Fetching from API: {url}")

    try:
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
# OPTIONAL: HTTPBIN REDIRECT TEST (EDUCATIONAL ONLY)
# ============================================================

def debug_httpbin_redirect_test():
    """
    Educational-only test:
    - Calls httpbin redirect endpoint
    - Follows redirect
    - Extracts shipid:xxxx from the final URL
    Has NO effect on your tracking logic.
    """
    print("[DEBUG] Starting httpbin redirect test...")
    try:
        # allow_redirects=True by default, but keep explicit
        resp = requests.get(HTTPBIN_TEST_URL, timeout=15, allow_redirects=True)
        final_url = resp.url
        print(f"[DEBUG] httpbin final URL: {final_url}")

        m = re.search(r"shipid:(\d+)", final_url)
        if m:
            shipid = m.group(1)
            print(f"[DEBUG] Extracted shipid from redirect: {shipid}")
        else:
            print("[DEBUG] Could not extract shipid from final URL")
    except Exception as e:
        print(f"[DEBUG] httpbin test failed: {e}")


# ============================================================
# ALERT LOGIC
# ============================================================

def humanize_eta(eta_hours: float) -> str:
    """Turn ETA in hours into a short 'Xd Yh' / 'Xh Ym' text."""
    total_minutes = int(round(eta_hours * 60))
    hours = total_minutes // 60
    mins = total_minutes % 60

    if hours < 24:
        if mins:
            return f"{hours}h {mins}m"
        return f"{hours}h"

    days = hours // 24
    rem_h = hours % 24
    if rem_h:
        return f"{days}d {rem_h}h"
    return f"{days}d"


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

    # For ETA: too old AIS -> no ETA
    too_old_for_eta = ais_age_min is not None and ais_age_min > MAX_AIS_FOR_ETA_MIN

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

    # ETA calculation with safeguards
    eta_hours = None
    eta_utc_str = None
    eta_text = None

    if (
        dest_distance_nm is not None
        and dest_distance_nm > MIN_DISTANCE_FOR_ETA
        and sog is not None
        and sog >= MIN_SOG_FOR_ETA
        and not too_old_for_eta
    ):
        effective_sog = min(max(sog, MIN_SOG_FOR_ETA), MAX_ETA_SOG_CAP)

        try:
            eta_hours_raw = dest_distance_nm / effective_sog
        except ZeroDivisionError:
            eta_hours_raw = None

        if eta_hours_raw is not None and eta_hours_raw <= MAX_ETA_HOURS:
            eta_hours = eta_hours_raw
            eta_dt = datetime.now(timezone.utc) + timedelta(hours=eta_hours)
            eta_utc_str = eta_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            eta_text = humanize_eta(eta_hours)

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
        "eta_hours": eta_hours,
        "eta_utc": eta_utc_str,
        "eta_text": eta_text,
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

        if eta_text and eta_utc_str:
            lines.append(f"⏱ ETA: {eta_text} ({eta_utc_str})")

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

    if eta_text and eta_utc_str:
        lines.append(f"⏱ ETA: {eta_text} ({eta_utc_str})")

    msg = "\n".join(lines)

    return msg, new_state


# ============================================================
# MAIN
# ============================================================

def main():

    # WARM-UP RENDER API (avoid cold-start timeouts)
    try:
        print("[INFO] Warming up Render API...")
        requests.get(f"{RENDER_BASE}/ping", timeout=30)
        print("[INFO] Render API awake ✔")
    except Exception as e:
        print(f"[WARN] Warm-up failed: {e}")

    # Load tracked IMOs
    imos = load_json(TRACKED_IMOS_PATH, [])

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

    # Load IMO -> shipid mapping for MarineTraffic overlay
    shipid_map = load_json(SHIPID_MAP_PATH, {})
    if not isinstance(shipid_map, dict):
        shipid_map = {}
    print(f"[INFO] Loaded shipid_map for {len(shipid_map)} entries.")

    new_all = {}

    for imo in imos:
        v = fetch_from_render_api(imo)
        if not v:
            print(f"[WARN] Could not fetch IMO {imo}. Skipping update for this vessel.")
            if imo in prev_all:
                new_all[imo] = prev_all[imo]
            continue

        # Apply MarineTraffic override (currently httpbin test only)
        v = override_lat_lon_from_marinetraffic(imo, v, shipid_map)

        prev_state = prev_all.get(imo)
        alert, new_state = build_alert_and_state(v, ports, prev_state)
        new_all[imo] = new_state

        if alert:
            print("[ALERT]", alert)
            send_whatsapp_message(alert)

    print(f"[INFO] Built state for {len(new_all)} vessel(s).")
    if new_all:
        save_json(VESSELS_STATE_PATH, new_all)
        print("Saved vessels_data.json ✔")
    else:
        print("[INFO] No valid vessel data, keeping existing vessels_data.json.")


# ============================================================

if __name__ == "__main__":
    main()
