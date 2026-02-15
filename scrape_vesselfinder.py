#!/usr/bin/env python3
"""
Vessel Tracking Script - Production Ready
"""

import json
import logging
import math
import os
import re
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION PATHS
# =============================================================================

TRACKED_IMOS_PATH = Path("data/tracked_imos.json")
VESSELS_STATE_PATH = Path("data/vessels_data.json")
PORTS_PATH = Path("data/ports.json")
STATIC_CACHE_PATH = Path("data/static_vessel_cache.json")
LOCK_FILE = Path("vessel_tracker.lock")

# =============================================================================
# EXTERNAL SERVICES CONFIG
# =============================================================================

CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE")
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY")
CALLMEBOT_ENABLED = bool(CALLMEBOT_PHONE and CALLMEBOT_APIKEY)
CALLMEBOT_API_URL = "https://api.callmebot.com/whatsapp.php"

RENDER_BASE = "https://vessel-api-s85s.onrender.com"

# =============================================================================
# TRACKING THRESHOLDS
# =============================================================================

MAX_AIS_MINUTES = 30
ARRIVAL_RADIUS_NM = 5.0  # Precise arrival detection
MIN_MOVE_NM = 5.0
MIN_SOG_FOR_ETA = 0.5
MAX_ETA_HOURS = 240
MAX_ETA_SOG_CAP = 18.0
MAX_AIS_FOR_ETA_MIN = 360
MIN_DISTANCE_FOR_ETA = 5.0
ARRIVAL_SOG_THRESHOLD = 0.5

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
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return default

def save_json_atomic(path: Path, data: Any) -> bool:
    """Writes to a temp file and renames to ensure data integrity."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
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

# =============================================================================
# PORT AND ETA LOGIC
# =============================================================================

ALIASES_RAW = {
    "tanger": "TANGER VILLE", "tangier": "TANGER VILLE", "tanger med": "TANGER MED",
    "las palmas": "LAS PALMAS", "algeciras": "ALGECIRAS", "gibraltar": "GIBRALTAR",
    "ceuta": "CEUTA", "casablanca": "CASABLANCA", "agadir": "AGADIR"
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

def humanize_eta(hours: float) -> str:
    total_m = int(round(hours * 60))
    h, m = divmod(total_m, 60)
    if h < 24: return f"{h}h {m}m" if m else f"{h}h"
    d, rh = divmod(h, 24)
    return f"{d}d {rh}h" if rh else f"{d}d"

# =============================================================================
# API COMMUNICATION
# =============================================================================

def fetch_with_retry(url: str) -> Optional[Dict]:
    for attempt in range(API_MAX_RETRIES):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"API Attempt {attempt+1} failed: {e}")
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_RETRY_BACKOFF_BASE * (2 ** attempt))
    return None

def fetch_vessel_data(imo: str, static_cache: Dict) -> Dict:
    result = static_cache.get(imo, {}).copy()
    api_data = fetch_with_retry(f"{RENDER_BASE}/vessel-full/{imo}")
    
    if api_data and api_data.get("found") is not False:
        result.update({
            "imo": imo,
            "lat": safe_float(api_data.get("lat")),
            "lon": safe_float(api_res := api_data.get("lon")), # Using walrus for brevity
            "sog": safe_float(api_data.get("sog"), 0.0),
            "cog": safe_float(api_data.get("cog"), 0.0),
            "destination": (api_data.get("destination") or "").strip(),
            "last_pos_utc": api_data.get("last_pos_utc"),
            "name": api_data.get("vessel_name") or api_data.get("name") or result.get("name") or f"IMO {imo}"
        })
    result["imo"] = imo
    return result

# =============================================================================
# CORE LOGIC
# =============================================================================

def build_alert_and_state(v: Dict, ports: Dict, prev: Optional[Dict]) -> Tuple[Optional[str], Dict]:
    imo, name = v["imo"], v.get("name", "Unknown")
    lat, lon, sog = v.get("lat"), v.get("lon"), v.get("sog", 0.0)
    
    new_state = {**v, "done": False}
    if lat is None or lon is None:
        if prev: new_state["done"] = prev.get("done", False)
        return None, new_state

    # 1. Port Matching
    near_name, near_dist = nearest_port(lat, lon, ports)
    dest_str = v.get("destination", "")
    dest_name, dest_data = match_destination_port(dest_str, ports)
    
    dest_dist = None
    if dest_data:
        dest_dist = haversine_nm(lat, lon, dest_data["lat"], dest_data["lon"])

    # 2. ETA
    eta_text = None
    if dest_dist and dest_dist > MIN_DISTANCE_FOR_ETA and sog >= MIN_SOG_FOR_ETA:
        speed = min(max(sog, MIN_SOG_FOR_ETA), MAX_ETA_SOG_CAP)
        hours = dest_dist / speed
        if hours <= MAX_ETA_HOURS:
            eta_text = humanize_eta(hours)

    new_state.update({
        "nearest_port": near_name, "nearest_dist": near_dist,
        "dest_port": dest_name, "dest_dist": dest_dist, "eta_text": eta_text
    })

    # 3. Arrival Check
    arrived = near_dist is not None and near_dist <= ARRIVAL_RADIUS_NM and sog <= ARRIVAL_SOG_THRESHOLD
    if arrived: new_state["done"] = True

    # 4. Alert Construction
    if not prev:
        return f"🚢 {name} ({imo})\n📌 Status: Tracking Started\n📍 Pos: {lat:.4f}, {lon:.4f}\n🎯 Dest: {dest_str}", new_state

    if prev.get("done") and new_state["done"]: return None, new_state

    alerts = []
    if arrived and not prev.get("done"):
        alerts.append(f"🏁 ARRIVED at {near_name}")
    
    old_dest = (prev.get("destination") or "").strip()
    if old_dest.upper() != dest_str.upper() and dest_str:
        alerts.append(f"📝 DEST CHANGE: {dest_str}")

    move_dist = haversine_nm(prev.get("lat", lat), prev.get("lon", lon), lat, lon)
    if move_dist >= MIN_MOVE_NM:
        alerts.append(f"📍 MOVED {move_dist:.1f} NM")

    if not alerts: return None, new_state

    msg = [f"🚢 {name}"] + alerts
    msg.append(f"⚡ Speed: {sog:.1f} kn")
    if eta_text: msg.append(f"⏱ ETA: {eta_text}")
    
    return "\n".join(msg), new_state

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    if LOCK_FILE.exists():
        # Check if lock is old (over 30 mins), if so, ignore it
        if time.time() - LOCK_FILE.stat().st_mtime > 1800:
            LOCK_FILE.unlink()
        else:
            logger.warning("Script already running. Exiting.")
            return
    LOCK_FILE.touch()

    try:
        # Load Data
        imos_raw = load_json(TRACKED_IMOS_PATH, [])
        imos = imos_raw.get("tracked_imos", []) if isinstance(imos_raw, dict) else imos_raw
        ports = load_json(PORTS_PATH, {})
        prev_states = load_json(VESSELS_STATE_PATH, {})
        static_cache = load_json(STATIC_CACHE_PATH, {})

        new_states_all = {}
        
        for imo in imos:
            imo = str(imo).strip()
            if not validate_imo(imo): continue
            
            v_data = fetch_vessel_data(imo, static_cache)
            alert, state = build_alert_and_state(v_data, ports, prev_states.get(imo))
            
            new_states_all[imo] = state
            
            if alert and CALLMEBOT_ENABLED:
                try:
                    requests.get(CALLMEBOT_API_URL, params={
                        "phone": CALLMEBOT_PHONE, "apikey": CALLMEBOT_APIKEY, "text": alert
                    }, timeout=15)
                    time.sleep(1) # Gentle rate limit
                except Exception as e:
                    logger.error(f"WhatsApp failed: {e}")

        save_json_atomic(VESSELS_STATE_PATH, new_states_all)
        logger.info("Tracking run completed successfully.")

    finally:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

if __name__ == "__main__":
    main()
