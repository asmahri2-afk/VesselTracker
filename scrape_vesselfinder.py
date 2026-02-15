#!/usr/bin/env python3
"""
Vessel Tracking Script - Battle Ready & Feature Complete
Thresholds updated for specific arrival logic (Speed < 1kn, Dist < 35.5nm).
Now includes Draught support for all cached vessels.
"""

import fcntl
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# =============================================================================
# DYNAMIC PATH CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

TRACKED_IMOS_PATH = DATA_DIR / "tracked_imos.json"
VESSELS_STATE_PATH = DATA_DIR / "vessels_data.json"
PORTS_PATH = DATA_DIR / "ports.json"
STATIC_CACHE_PATH = DATA_DIR / "static_vessel_cache.json"
LOCK_FILE_PATH = DATA_DIR / "vessel_tracker.lock"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================================================
# EXTERNAL SERVICES CONFIG
# =============================================================================

CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE")
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY")
CALLMEBOT_ENABLED = bool(CALLMEBOT_PHONE and CALLMEBOT_APIKEY)
CALLMEBOT_API_URL = "https://api.callmebot.com/whatsapp.php"

RENDER_BASE = "https://vessel-api-s85s.onrender.com"

# =============================================================================
# TRACKING THRESHOLDS (UPDATED)
# =============================================================================

ARRIVAL_RADIUS_NM = 35.5        # User requested: 35nm radius for arrival
MIN_MOVE_NM = 5.0
MIN_SOG_FOR_ETA = 0.5
MAX_ETA_HOURS = 240
MAX_ETA_SOG_CAP = 18.0
MAX_AIS_FOR_ETA_MIN = 360
MIN_DISTANCE_FOR_ETA = 5.0
ARRIVAL_SOG_THRESHOLD = 1.0     # User requested: Speed < 1kn means arrived

API_MAX_RETRIES = 3
API_RETRY_BACKOFF_BASE = 2.0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Corrupt JSON detected at {path}, returning default.")
        return default
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return default

def save_json_atomic(path: Path, data: Any) -> bool:
    try:
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_path.replace(path)
        return True
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")
        return False

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    return (R * 2 * math.asin(math.sqrt(a))) * 0.539957

def validate_imo(imo: str) -> bool:
    imo = str(imo).strip()
    if not re.match(r'^\d{7}$', imo):
        return False
    try:
        total = sum(int(imo[i]) * (7 - i) for i in range(6))
        return int(imo[6]) == total % 10
    except Exception:
        return False

def normalize_string(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Z]", "", s.upper())

def parse_ais_time(s: str) -> Optional[datetime]:
    if not s: return None
    s = s.replace(" UTC", "").strip()
    try:
        return datetime.strptime(s, "%b %d, %Y %H:%M").replace(tzinfo=timezone.utc)
    except:
        return None

def age_minutes(t: str) -> Optional[float]:
    dt = parse_ais_time(t)
    if not dt: return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60

def humanize_eta(h: float) -> str:
    total_m = int(round(h * 60))
    hm, mm = divmod(total_m, 60)
    if hm < 24: return f"{hm}h {mm}m" if mm else f"{hm}h"
    d, rh = divmod(hm, 24)
    return f"{d}d {rh}h" if rh else f"{d}d"

# =============================================================================
# PORT AND ETA LOGIC
# =============================================================================

ALIASES_RAW = {
    "laayoune": "LAAYOUNE", "layoune": "LAAYOUNE", "EH EUN": "LAAYOUNE", "leyoune": "LAAYOUNE",
    "tantan": "TAN TAN", "tan tan": "TAN TAN", "tan-tan": "TAN TAN", "tan tan anch": "TAN TAN",
    "dakhla": "DAKHLA", "dakhla port": "DAKHLA", "ad dakhla": "DAKHLA",
    "dakhla anch": "DAKHLA ANCH", "dakhla anch.": "DAKHLA ANCH", "dakhla anchorage": "DAKHLA ANCH",
    "dakhla anch area": "DAKHLA ANCH",
    "agadir": "AGADIR", "port agadir": "AGADIR",
    "essaouira": "ESSAOUIRA", "safi": "SAFI",
    "casa": "CASABLANCA", "casablanca": "CASABLANCA", "cassablanca": "CASABLANCA",
    "mohammedia": "MOHAMMEDIA",
    "jorf": "JORF LASFAR", "jorf lasfar": "JORF LASFAR",
    "kenitra": "KENITRA",
    "tanger": "TANGER VILLE", "tangier": "TANGER VILLE", "tanger ville": "TANGER VILLE",
    "tanger med": "TANGER MED", "tm2": "TANGER MED",
    "nador": "NADOR",
    "al hoceima": "AL HOCEIMA", "alhucemas": "AL HOCEIMA",
    "las palmas": "LAS PALMAS", "lpa": "LAS PALMAS", "las palmas anch": "LAS PALMAS",
    "arrecife": "ARRECIFE",
    "puerto del rosario": "PUERTO DEL ROSARIO", "pdr": "PUERTO DEL ROSARIO",
    "santa cruz": "SANTA CRUZ DE TENERIFE", "sctf": "SANTA CRUZ DE TENERIFE", "santa cruz tenerife": "SANTA CRUZ DE TENERIFE",
    "san sebastian": "SAN SEBASTIAN DE LA GOMERA",
    "la restinga": "LA RESTINGA",
    "la palma": "LA PALMA",
    "granadilla": "GRANADILLA", "puerto de granadilla": "GRANADILLA",
    "ceuta": "CEUTA", "melilla": "MELILLA",
    "algeciras": "ALGECIRAS", "alg": "ALGECIRAS",
    "gibraltar": "GIBRALTAR", "gib": "GIBRALTAR",
    "huelva": "HUELVA",
    "huelva anch": "HUELVA ANCH", "huelva anchorage": "HUELVA ANCH",
    "cadiz": "CADIZ", "cadiz anch": "CADIZ",
    "sevilla": "SEVILLA", "seville": "SEVILLA",
    "malaga": "MALAGA", "motril": "MOTRIL", "almeria": "ALMERIA",
    "cartagena": "CARTAGENA", "valencia": "VALENCIA",
    "sines": "SINES", "setubal": "SETUBAL", "lisbon": "LISBON", "lisboa": "LISBON"
}
DEST_ALIASES = {normalize_string(k): v for k, v in ALIASES_RAW.items()}

def match_destination_port(dest: str, ports: Dict[str, Dict]) -> Tuple[Optional[str], Optional[Dict]]:
    if not dest: return None, None
    norm = normalize_string(dest)
    
    if norm in DEST_ALIASES:
        canonical = DEST_ALIASES[norm]
        return canonical, ports.get(canonical)
    
    port_lookup = {normalize_string(p): p for p in ports}
    if norm in port_lookup:
        name = port_lookup[norm]
        return name, ports.get(name)
        
    for canon_key, port_name in port_lookup.items():
        if canon_key and canon_key in norm:
            return port_name, ports.get(port_name)
            
    return None, None

def nearest_port(lat: float, lon: float, ports: Dict[str, Dict]) -> Tuple[Optional[str], Optional[float]]:
    best_name, best_dist = None, None
    for name, coords in ports.items():
        try:
            d = haversine_nm(lat, lon, coords["lat"], coords["lon"])
            if best_dist is None or d < best_dist:
                best_dist, best_name = d, name
        except Exception: continue
    return best_name, best_dist

# =============================================================================
# API COMMUNICATION
# =============================================================================

def fetch_with_retry(url: str) -> Optional[Dict]:
    for attempt in range(API_MAX_RETRIES):
        try:
            headers = {'User-Agent': 'VesselTracker/1.0'}
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"API Attempt {attempt+1} failed for {url}: {e}")
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_RETRY_BACKOFF_BASE * (2 ** attempt))
    return None

def fetch_vessel_data(imo: str, static_cache: Dict) -> Dict:
    result = static_cache.get(imo, {}).copy()
    api_data = fetch_with_retry(f"{RENDER_BASE}/vessel-full/{imo}")
    
    if api_data and api_data.get("found") is not False:
        result["lat"] = safe_float(api_data.get("lat")) if api_data.get("lat") is not None else result.get("lat")
        result["lon"] = safe_float(api_data.get("lon")) if api_data.get("lon") is not None else result.get("lon")
        
        result["sog"] = safe_float(api_data.get("sog"), 0.0)
        result["cog"] = safe_float(api_data.get("cog"), 0.0)
        result["last_pos_utc"] = api_data.get("last_pos_utc")
        result["destination"] = (api_data.get("destination") or "").strip()
        
        result["name"] = (api_data.get("vessel_name") or api_data.get("name") or result.get("name") or f"IMO {imo}").strip()
        result["ship_type"] = (api_data.get("ship_type") or result.get("ship_type") or "").strip()
        result["flag"] = (api_data.get("flag") or result.get("flag") or "").strip()
        
        # UPDATED KEY LIST: Includes draught_m and predicted_eta
        keys_to_update = [
            "deadweight_t", "gross_tonnage", "year_of_build", 
            "length_overall_m", "beam_m", "draught_m", "predicted_eta"
        ]
        
        for key in keys_to_update:
            if api_data.get(key) is not None:
                result[key] = api_data.get(key)
            elif result.get(key) is None:
                result[key] = None

    result["imo"] = imo
    return result

# =============================================================================
# CORE LOGIC
# =============================================================================

def build_alert_and_state(v: Dict, ports: Dict, prev: Optional[Dict]) -> Tuple[Optional[str], Dict]:
    imo, name = v["imo"], v.get("name", "Unknown")
    lat, lon, sog = v.get("lat"), v.get("lon"), v.get("sog", 0.0)
    cog = v.get("cog")
    last = v.get("last_pos_utc")
    dest = v.get("destination", "")

    new_state = {
        "imo": imo, "name": name, "lat": lat, "lon": lon, "sog": sog, "cog": cog, 
        "last_pos_utc": last, "destination": dest, "done": False,
        "ship_type": v.get("ship_type"), "flag": v.get("flag"),
        "deadweight_t": v.get("deadweight_t"), "gross_tonnage": v.get("gross_tonnage"),
        "year_of_build": v.get("year_of_build"), "length_overall_m": v.get("length_overall_m"),
        "beam_m": v.get("beam_m"), 
        "draught_m": v.get("draught_m") # ADDED TO ENSURE PERSISTENCE
    }

    if lat is None or lon is None:
        if prev: new_state["done"] = prev.get("done", False)
        return None, new_state

    age = age_minutes(last) if last else None
    age_txt = "N/A" if age is None else f"{age:.0f} min ago"
    too_old = age is not None and age > MAX_AIS_FOR_ETA_MIN

    near_name, near_dist = nearest_port(lat, lon, ports)
    dest_name, dest_data = match_destination_port(dest, ports)
    
    dest_dist = None
    if dest_data:
        dest_dist = haversine_nm(lat, lon, dest_data["lat"], dest_data["lon"])

    eta_h, eta_utc_str, eta_text = None, None, None
    if dest_dist is not None and dest_dist > MIN_DISTANCE_FOR_ETA and sog >= MIN_SOG_FOR_ETA and not too_old:
        speed = min(max(sog, MIN_SOG_FOR_ETA), MAX_ETA_SOG_CAP)
        try:
            raw_h = dest_dist / speed
            if raw_h <= MAX_ETA_HOURS:
                eta_h = raw_h
                eta_dt = datetime.now(timezone.utc) + timedelta(hours=eta_h)
                eta_utc_str = eta_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                eta_text = humanize_eta(eta_h)
        except: pass

    new_state.update({
        "nearest_port": near_name, "nearest_distance_nm": near_dist,
        "destination_port": dest_name, "destination_distance_nm": dest_dist,
        "eta_hours": eta_h, "eta_utc": eta_utc_str, "eta_text": eta_text
    })

    arrived = near_dist is not None and near_dist <= ARRIVAL_RADIUS_NM and sog <= ARRIVAL_SOG_THRESHOLD
    if arrived: new_state["done"] = True

    if not prev:
        msg = [
            f"🚢 {name} (IMO {imo})", "📌 Status: First tracking detected",
            f"🕒 AIS: {age_txt}", f"⚡ Speed: {sog:.1f} kn | 🧭 {cog:.0f}°",
            f"📍 Position: {lat:.4f} , {lon:.4f}",
            f"⚓ Nearest port: {near_name or 'N/A'} (~{near_dist:.1f} NM)" if near_dist else "⚓ Nearest port: N/A",
            f"🎯 Destination: {dest or 'N/A'}"
        ]
        if dest_name and dest_dist is not None: msg[-1] += f" (~{dest_dist:.1f} NM)"
        if eta_text: msg.append(f"⏱ ETA: {eta_text} ({eta_utc_str})")
        return "\n".join(msg), new_state

    if prev.get("done") and new_state["done"]: return None, new_state

    old_dest = (prev.get("destination") or "").strip()
    dest_changed = (old_dest.upper() != dest.upper()) if (old_dest or dest) else False
    
    p_lat, p_lon = prev.get("lat"), prev.get("lon")
    moved, diff = False, None
    if p_lat is not None and p_lon is not None:
        diff = haversine_nm(p_lat, p_lon, lat, lon)
        moved = diff >= MIN_MOVE_NM

    arrival_event = arrived and not prev.get("done")
    if not (dest_changed or moved or arrival_event): return None, new_state

    if arrival_event: status = "Arrived at destination area"
    elif dest_changed: status = "Destination changed"
    else: status = "Position / track updated"
    
    extra = f" (Δ {diff:.1f} NM)" if moved and diff is not None else ""
    
    if dest_changed and old_dest: 
        dest_line = f"🎯 Destination changed: {old_dest} ➜ {dest or 'N/A'}"
    else: 
        dest_line = f"🎯 Destination: {dest or 'N/A'}"
    if dest_name and dest_dist is not None: dest_line += f" (~{dest_dist:.1f} NM)"

    msg = [
        f"🚢 {name} (IMO {imo})", f"📌 Status: {status}",
        f"🕒 AIS: {age_txt}", f"⚡ Speed: {sog:.1f} kn | 🧭 {cog:.0f}°{extra}",
        f"📍 Position: {lat:.4f} , {lon:.4f}",
        f"⚓ Nearest port: {near_name or 'N/A'} (~{near_dist:.1f} NM)" if near_dist else "⚓ Nearest port: N/A",
        dest_line
    ]
    if eta_text: msg.append(f"⏱ ETA: {eta_text} ({eta_utc_str})")
    
    return "\n".join(msg), new_state

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    lock_file = None
    try:
        lock_file = open(LOCK_FILE_PATH, 'w')
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.info("Lock acquired.")
    except BlockingIOError:
        logger.warning("Script already running. Exiting.")
        return
    except Exception as e:
        logger.error(f"Lock error: {e}")
        return

    try:
        try: requests.get(f"{RENDER_BASE}/ping", timeout=10)
        except: pass

        static_cache = load_json(STATIC_CACHE_PATH, {})
        imos_raw = load_json(TRACKED_IMOS_PATH, [])
        imos = imos_raw.get("tracked_imos", []) if isinstance(imos_raw, dict) else imos_raw
        
        ports = {k.upper(): v for k, v in load_json(PORTS_PATH, {}).items()}
        if not ports: raise RuntimeError("ports.json missing or empty")
        
        prev_states = load_json(VESSELS_STATE_PATH, {})
        new_states_all = {}
        
        for imo in imos:
            imo = str(imo).strip()
            if not validate_imo(imo):
                logger.warning(f"Invalid IMO skipped: {imo}")
                continue
            
            v_data = fetch_vessel_data(imo, static_cache)
            alert, state = build_alert_and_state(v_data, ports, prev_states.get(imo))
            
            new_states_all[imo] = state
            
            if alert and CALLMEBOT_ENABLED:
                try:
                    requests.get(CALLMEBOT_API_URL, params={
                        "phone": CALLMEBOT_PHONE, "apikey": CALLMEBOT_APIKEY, "text": alert
                    }, timeout=15)
                    logger.info(f"Alert sent for {imo}")
                    time.sleep(1) 
                except Exception as e:
                    logger.error(f"WhatsApp failed: {e}")

        save_json_atomic(VESSELS_STATE_PATH, new_states_all)
        logger.info("Tracking run completed successfully.")

    finally:
        if lock_file:
            lock_file.close()
        if LOCK_FILE_PATH.exists():
            LOCK_FILE_PATH.unlink()

if __name__ == "__main__":
    main()
