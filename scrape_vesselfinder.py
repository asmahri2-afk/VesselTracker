#!/usr/bin/env python3
"""
Vessel Tracking Script - Supabase Edition
All original logic preserved. Only storage changed: JSON files → Supabase tables.

Changes from original:
  - Removed: file locking (not needed in GitHub Actions)
  - Removed: load_json / save_json_atomic (replaced by Supabase)
  - Removed: local file paths
  - Added: Supabase client for all reads and writes
  - Everything else: 100% identical to original
"""

import logging
import math
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from supabase import create_client

# =============================================================================
# SUPABASE
# =============================================================================
SUPABASE_URL = "https://rpzcphszvdgjsqnhwdhm.supabase.co"
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================================================================
# LOGGING
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
CALLMEBOT_PHONE   = os.getenv("CALLMEBOT_PHONE")
CALLMEBOT_APIKEY  = os.getenv("CALLMEBOT_APIKEY")
CALLMEBOT_ENABLED = bool(CALLMEBOT_PHONE and CALLMEBOT_APIKEY)
CALLMEBOT_API_URL = "https://api.callmebot.com/whatsapp.php"

RENDER_BASE = os.getenv("RENDER_BASE", "https://vessel-api-s85s.onrender.com")
API_SECRET  = os.getenv("API_SECRET", "")

# =============================================================================
# TRACKING THRESHOLDS (unchanged)
# =============================================================================
ARRIVAL_RADIUS_NM     = 35.5
MIN_MOVE_NM           = 5.0
MIN_SOG_FOR_ETA       = 0.5
MAX_ETA_HOURS         = 240
MAX_ETA_SOG_CAP       = 18.0
MAX_AIS_FOR_ETA_MIN   = 360
MIN_DISTANCE_FOR_ETA  = 5.0
ARRIVAL_SOG_THRESHOLD = 1.0

API_MAX_RETRIES        = 3
API_RETRY_BACKOFF_BASE = 2.0
MAX_IMO_FAILURES       = 5
ALERT_COOLDOWN_MIN     = 45

# =============================================================================
# DATABASE HELPERS (replaces load_json / save_json_atomic)
# =============================================================================

def db_load_tracked_imos() -> List[str]:
    res = sb.table("tracked_imos").select("imo").execute()
    return [r["imo"] for r in (res.data or [])]

def db_load_ports() -> Dict:
    res = sb.table("ports").select("name,lat,lon").execute()
    return {r["name"].upper(): {"lat": r["lat"], "lon": r["lon"]}
            for r in (res.data or [])}

def db_load_vessels_state() -> Dict:
    res = sb.table("vessels").select("*").execute()
    return {r["imo"]: r for r in (res.data or [])}

def db_load_static_cache() -> Dict:
    res = sb.table("static_vessel_cache").select("*").execute()
    return {r["imo"]: r for r in (res.data or [])}

def db_load_failure_counts() -> Dict:
    res = sb.table("failure_counts").select("imo,count").execute()
    return {r["imo"]: r["count"] for r in (res.data or [])}

def db_save_vessel(state: Dict):
    row = dict(state)
    row["updated_at"] = datetime.now(timezone.utc).isoformat()
    sb.table("vessels").upsert(row).execute()

def db_save_static_cache(imo: str, entry: Dict):
    row = dict(entry)
    row["imo"] = imo
    sb.table("static_vessel_cache").upsert(row).execute()

def db_save_failure_count(imo: str, count: int):
    sb.table("failure_counts").upsert({"imo": imo, "count": count}).execute()

# =============================================================================
# UTILITY FUNCTIONS (unchanged)
# =============================================================================

def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = math.sin(dlat / 2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2)**2
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
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Z]", "", s.upper())

def parse_ais_time(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.replace(" UTC", "").strip()
    try:
        return datetime.strptime(s, "%b %d, %Y %H:%M").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def age_minutes(t: str) -> Optional[float]:
    dt = parse_ais_time(t)
    if not dt:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60

def humanize_age(minutes: Optional[float]) -> str:
    if minutes is None:
        return "N/A"
    minutes = int(round(minutes))
    if minutes < 60:
        return f"{minutes} min ago"
    h, m = divmod(minutes, 60)
    if h < 24:
        return f"{h}h {m}m ago" if m else f"{h}h ago"
    d, rh = divmod(h, 24)
    return f"{d}d {rh}h ago" if rh else f"{d}d ago"

def humanize_eta(h: float) -> str:
    total_m = int(round(h * 60))
    hm, mm = divmod(total_m, 60)
    if hm < 24:
        return f"{hm}h {mm}m" if mm else f"{hm}h"
    d, rh = divmod(hm, 24)
    return f"{d}d {rh}h" if rh else f"{d}d"

# =============================================================================
# WHATSAPP (unchanged)
# =============================================================================

def send_whatsapp(text: str) -> bool:
    if not CALLMEBOT_ENABLED:
        return False
    try:
        r = requests.get(CALLMEBOT_API_URL, params={
            "phone": CALLMEBOT_PHONE,
            "apikey": CALLMEBOT_APIKEY,
            "text": text
        }, timeout=15)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"WhatsApp send failed: {e}")
        return False

# =============================================================================
# PORT AND ETA LOGIC (unchanged)
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
    "santa cruz": "SANTA CRUZ DE TENERIFE", "sctf": "SANTA CRUZ DE TENERIFE",
    "santa cruz tenerife": "SANTA CRUZ DE TENERIFE",
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

def match_destination_port(dest: str, ports: Dict) -> Tuple[Optional[str], Optional[Dict]]:
    if not dest:
        return None, None
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

def nearest_port(lat: float, lon: float, ports: Dict) -> Tuple[Optional[str], Optional[float]]:
    best_name, best_dist = None, None
    for name, coords in ports.items():
        try:
            d = haversine_nm(lat, lon, coords["lat"], coords["lon"])
            if best_dist is None or d < best_dist:
                best_dist, best_name = d, name
        except Exception:
            continue
    return best_name, best_dist

# =============================================================================
# API COMMUNICATION (unchanged)
# =============================================================================

def fetch_with_retry(url: str, method: str = "GET", json_body: Dict = None) -> Optional[Dict]:
    headers: Dict[str, str] = {"User-Agent": "VesselTracker/1.0"}
    if API_SECRET:
        headers["X-API-Secret"] = API_SECRET
    for attempt in range(API_MAX_RETRIES):
        try:
            if method == "POST":
                r = requests.post(url, headers=headers, json=json_body, timeout=120)
            else:
                r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"API attempt {attempt+1}/{API_MAX_RETRIES} failed for {url}: {e}")
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_RETRY_BACKOFF_BASE * (2 ** attempt))
    return None

def fetch_vessel_data(imo: str, static_cache: Dict) -> Dict:
    result = static_cache.get(imo, {}).copy()
    api_data = fetch_with_retry(f"{RENDER_BASE}/vessel-full/{imo}")

    if api_data and api_data.get("found") is not False:
        result["lat"]         = safe_float(api_data.get("lat")) if api_data.get("lat") is not None else result.get("lat")
        result["lon"]         = safe_float(api_data.get("lon")) if api_data.get("lon") is not None else result.get("lon")
        result["sog"]         = safe_float(api_data.get("sog"), 0.0)
        result["cog"]         = safe_float(api_data.get("cog"))
        result["last_pos_utc"]= api_data.get("last_pos_utc")
        result["destination"] = (api_data.get("destination") or "").strip()
        result["name"]        = (api_data.get("vessel_name") or api_data.get("name") or result.get("name") or f"IMO {imo}").strip()
        result["ship_type"]   = (api_data.get("ship_type") or result.get("ship_type") or "").strip()
        result["flag"]        = (api_data.get("flag") or result.get("flag") or "").strip()

        static_keys = ["deadweight_t", "gross_tonnage", "year_of_build", "length_overall_m", "beam_m", "draught_m"]
        for key in static_keys:
            if api_data.get(key) is not None:
                result[key] = api_data[key]
            elif result.get(key) is None:
                result[key] = None

        # Update static cache in Supabase
        cache_entry = {k: result.get(k) for k in static_keys + ["name", "ship_type", "flag"]}
        db_save_static_cache(imo, cache_entry)

    result["imo"] = imo
    return result

def fetch_all_vessels_batch(imos: List[str], static_cache: Dict) -> Dict[str, Dict]:
    """Fetch all vessels in one batch request instead of N sequential calls."""
    logger.info(f"Fetching {len(imos)} vessels via batch endpoint...")
    response = fetch_with_retry(
        f"{RENDER_BASE}/vessel-batch",
        method="POST",
        json_body={"imos": imos}
    )
    if not response:
        logger.warning("Batch fetch failed — falling back to sequential")
        return {imo: fetch_vessel_data(imo, static_cache) for imo in imos}

    results = {}
    for imo, api_data in response.get("results", {}).items():
        result = static_cache.get(imo, {}).copy()
        if api_data and api_data.get("found") is not False:
            result["lat"]          = safe_float(api_data.get("lat")) if api_data.get("lat") is not None else result.get("lat")
            result["lon"]          = safe_float(api_data.get("lon")) if api_data.get("lon") is not None else result.get("lon")
            result["sog"]          = safe_float(api_data.get("sog"), 0.0)
            result["cog"]          = safe_float(api_data.get("cog"))
            result["last_pos_utc"] = api_data.get("last_pos_utc")
            result["destination"]  = (api_data.get("destination") or "").strip()
            result["name"]         = (api_data.get("vessel_name") or api_data.get("name") or result.get("name") or f"IMO {imo}").strip()
            result["ship_type"]    = (api_data.get("ship_type") or result.get("ship_type") or "").strip()
            result["flag"]         = (api_data.get("flag") or result.get("flag") or "").strip()
            static_keys = ["deadweight_t", "gross_tonnage", "year_of_build", "length_overall_m", "beam_m", "draught_m"]
            for key in static_keys:
                result[key] = api_data.get(key) if api_data.get(key) is not None else result.get(key)
            cache_entry = {k: result.get(k) for k in static_keys + ["name", "ship_type", "flag"]}
            db_save_static_cache(imo, cache_entry)
        result["imo"] = imo
        results[imo] = result

    # Log errors
    for imo, err in response.get("errors", {}).items():
        logger.warning(f"Batch error for IMO {imo}: {err}")
        results[imo] = static_cache.get(imo, {"imo": imo})

    logger.info(f"Batch complete: {response.get('success', 0)} ok, {response.get('failed', 0)} failed")
    return results

# =============================================================================
# CORE LOGIC (unchanged)
# =============================================================================

def build_alert_and_state(
    v: Dict,
    ports: Dict,
    prev: Optional[Dict],
    failure_counts: Dict[str, int]
) -> Tuple[Optional[str], Dict]:

    imo, name = v["imo"], v.get("name", "Unknown")
    lat, lon  = v.get("lat"), v.get("lon")
    sog       = v.get("sog", 0.0)
    cog       = v.get("cog")
    last      = v.get("last_pos_utc")
    dest      = v.get("destination", "")

    cog_str = f"{cog:.0f}°" if cog is not None else "N/A"

    new_state = {
        "imo": imo, "name": name, "lat": lat, "lon": lon,
        "sog": sog, "cog": cog, "last_pos_utc": last,
        "destination": dest, "done": False,
        "ship_type": v.get("ship_type"), "flag": v.get("flag"),
        "deadweight_t": v.get("deadweight_t"), "gross_tonnage": v.get("gross_tonnage"),
        "year_of_build": v.get("year_of_build"), "length_overall_m": v.get("length_overall_m"),
        "beam_m": v.get("beam_m"), "draught_m": v.get("draught_m"),
        "last_alert_utc": prev.get("last_alert_utc") if prev else None,
    }

    if lat is None or lon is None:
        if prev:
            new_state["done"] = prev.get("done", False)
        return None, new_state

    age     = age_minutes(last) if last else None
    age_txt = humanize_age(age)
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
        except Exception as e:
            logger.warning(f"ETA calc failed for {imo}: {e}")

    new_state.update({
        "nearest_port": near_name, "nearest_distance_nm": near_dist,
        "destination_port": dest_name, "destination_distance_nm": dest_dist,
        "eta_hours": eta_h, "eta_utc": eta_utc_str, "eta_text": eta_text
    })

    check_dist = dest_dist if dest_dist is not None else near_dist
    arrived = check_dist is not None and check_dist <= ARRIVAL_RADIUS_NM and sog <= ARRIVAL_SOG_THRESHOLD
    if arrived:
        new_state["done"] = True

    if not prev:
        msg = [
            f"🚢 {name} (IMO {imo})",
            "📌 Status: First tracking detected",
            f"🕒 AIS: {age_txt}",
            f"⚡ Speed: {sog:.1f} kn | 🧭 {cog_str}",
            f"📍 Position: {lat:.4f}, {lon:.4f}",
            f"⚓ Nearest port: {near_name or 'N/A'} (~{near_dist:.1f} NM)" if near_dist else "⚓ Nearest port: N/A",
            f"🎯 Destination: {dest or 'N/A'}"
        ]
        if dest_name and dest_dist is not None:
            msg[-1] += f" (~{dest_dist:.1f} NM)"
        if eta_text:
            msg.append(f"⏱ ETA: {eta_text} ({eta_utc_str})")
        new_state["last_alert_utc"] = datetime.now(timezone.utc).isoformat()
        return "\n".join(msg), new_state

    if prev.get("done") and new_state["done"]:
        return None, new_state

    old_dest     = (prev.get("destination") or "").strip()
    old_canon, _ = match_destination_port(old_dest, ports)
    new_canon    = dest_name
    dest_changed = old_canon != new_canon if (old_dest or dest) else False

    p_lat, p_lon = prev.get("lat"), prev.get("lon")
    moved, diff  = False, None
    if p_lat is not None and p_lon is not None:
        diff  = haversine_nm(p_lat, p_lon, lat, lon)
        moved = diff >= MIN_MOVE_NM

    arrival_event = arrived and not prev.get("done")

    if not (dest_changed or moved or arrival_event):
        return None, new_state

    if not arrival_event:
        last_alert = prev.get("last_alert_utc")
        if last_alert:
            try:
                last_alert_dt = datetime.fromisoformat(last_alert)
                mins_since = (datetime.now(timezone.utc) - last_alert_dt).total_seconds() / 60
                if mins_since < ALERT_COOLDOWN_MIN:
                    logger.info(f"Cooldown active for {imo} ({mins_since:.0f} min since last alert)")
                    return None, new_state
            except Exception:
                pass

    if arrival_event:
        status = "Arrived at destination area"
    elif dest_changed:
        status = "Destination changed"
    else:
        status = "Position / track updated"

    extra        = f" (Δ {diff:.1f} NM)" if moved and diff is not None else ""
    dest_display = f"{old_dest} ➜ {dest or 'N/A'}" if dest_changed and old_dest else (dest or "N/A")
    dest_icon    = "🎯 Destination changed:" if dest_changed and old_dest else "🎯 Destination:"
    dest_line    = f"{dest_icon} {dest_display}"
    if dest_name and dest_dist is not None:
        dest_line += f" (~{dest_dist:.1f} NM)"

    msg = [
        f"🚢 {name} (IMO {imo})",
        f"📌 Status: {status}",
        f"🕒 AIS: {age_txt}",
        f"⚡ Speed: {sog:.1f} kn | 🧭 {cog_str}{extra}",
        f"📍 Position: {lat:.4f}, {lon:.4f}",
        f"⚓ Nearest port: {near_name or 'N/A'} (~{near_dist:.1f} NM)" if near_dist else "⚓ Nearest port: N/A",
        dest_line
    ]
    if eta_text:
        msg.append(f"⏱ ETA: {eta_text} ({eta_utc_str})")

    new_state["last_alert_utc"] = datetime.now(timezone.utc).isoformat()
    return "\n".join(msg), new_state

# =============================================================================
# RENDER COLD START WAIT (unchanged)
# =============================================================================

def wait_for_render(max_wait_sec: int = 90) -> bool:
    logger.info("Waiting for Render API to be ready...")
    interval = 10
    for attempt in range(max_wait_sec // interval):
        try:
            r = requests.get(f"{RENDER_BASE}/ping", timeout=10)
            if r.ok:
                logger.info(f"Render ready after ~{attempt * interval}s")
                return True
        except Exception:
            pass
        logger.info(f"Render not ready yet, retrying in {interval}s...")
        time.sleep(interval)
    logger.error("Render did not become ready in time.")
    return False

# =============================================================================
# MAIN — replaces file I/O with Supabase reads/writes
# =============================================================================

def main():
    try:
        if not wait_for_render():
            send_whatsapp("⚠️ Vessel Tracker: Render API unavailable. Run skipped.")
            return

        # Load everything from Supabase
        static_cache   = db_load_static_cache()
        imos           = db_load_tracked_imos()
        ports          = db_load_ports()
        failure_counts = db_load_failure_counts()
        prev_states    = db_load_vessels_state()

        if not ports:
            send_whatsapp("⚠️ Vessel Tracker: ports table empty. Run aborted.")
            raise RuntimeError("ports table is empty")

        logger.info(f"Tracking {len(imos)} vessels | {len(ports)} ports loaded")

        # Filter out invalid and max-failed IMOs before batch fetch
        valid_imos = [
            str(imo).strip() for imo in imos
            if validate_imo(str(imo).strip())
            and failure_counts.get(str(imo).strip(), 0) < MAX_IMO_FAILURES
        ]
        skipped = len(imos) - len(valid_imos)
        if skipped:
            logger.warning(f"{skipped} IMO(s) skipped (invalid or max failures reached)")

        # Fetch all vessels in one batch request instead of N sequential calls
        all_vessel_data = fetch_all_vessels_batch(valid_imos, static_cache)

        for imo in valid_imos:
            v_data = all_vessel_data.get(imo, {"imo": imo})
            fails  = failure_counts.get(imo, 0)

            if v_data.get("lat") is None and v_data.get("lon") is None:
                failure_counts[imo] = fails + 1
                db_save_failure_count(imo, failure_counts[imo])
                if failure_counts[imo] == MAX_IMO_FAILURES:
                    send_whatsapp(f"⚠️ Vessel Tracker: IMO {imo} failed {MAX_IMO_FAILURES} times in a row.")
            else:
                failure_counts[imo] = 0
                db_save_failure_count(imo, 0)

            alert, state = build_alert_and_state(v_data, ports, prev_states.get(imo), failure_counts)

            # Save vessel state to Supabase
            db_save_vessel(state)
            logger.info(f"Saved: {state.get('name')} (IMO {imo})")

            if alert:
                logger.info(f"Alert triggered for {imo}:\n{alert}")
                if CALLMEBOT_ENABLED:
                    ok = send_whatsapp(alert)
                    logger.info(f"Alert {'sent' if ok else 'FAILED'} for {imo}")
                    time.sleep(1)

        logger.info("Tracking run completed successfully.")

        # ─────────────────────────────────────────────────────────────────────
        # PER-USER CALLMEBOT ALERTS (multi-user support)
        # ─────────────────────────────────────────────────────────────────────
        logger.info("Checking per-user CallMeBot alerts...")

        import urllib.parse as _urllib_parse

        def _send_whatsapp_alert(phone: str, apikey: str, message: str):
            """Send WhatsApp message via CallMeBot API for a specific user."""
            try:
                encoded_msg = _urllib_parse.quote(message)
                url = f"https://api.callmebot.com/whatsapp.php?phone={_urllib_parse.quote(phone)}&text={encoded_msg}&apikey={_urllib_parse.quote(apikey)}"
                r = requests.get(url, timeout=10)
                logger.info(f"CallMeBot alert sent to {phone}: {r.status_code}")
            except Exception as e:
                logger.warning(f"CallMeBot alert failed for {phone}: {e}")

        def _check_vessel_alerts(imo: str, current_data: dict, previous_data: dict) -> list:
            """Check if vessel status changed and return alert messages."""
            alerts = []
            if not previous_data:
                return alerts

            prev_sog = float(previous_data.get('sog') or 0)
            curr_sog = float(current_data.get('sog') or 0)
            dest = (current_data.get('destination') or '').upper()
            dest_dist = current_data.get('destination_distance_nm')

            # Was moving, now stopped
            if prev_sog > 0.5 and curr_sog <= 0.5:
                name = current_data.get('name', f'IMO {imo}')
                if dest_dist is not None and float(dest_dist) <= 30:
                    alerts.append(f"⚓ {name} arrived at port near {dest or 'destination'}")
                else:
                    alerts.append(f"🔴 {name} has stopped (stalled)")

            # Signal age check
            last_pos = current_data.get('last_pos_utc')
            if last_pos:
                try:
                    from datetime import datetime, timezone as _tz
                    pos_time = datetime.fromisoformat(last_pos.replace(' UTC', '').replace('Z', ''))
                    age_hours = (datetime.now(_tz.utc) - pos_time.replace(tzinfo=_tz.utc)).total_seconds() / 3600
                    if age_hours > 6:
                        name = current_data.get('name', f'IMO {imo}')
                        alerts.append(f"📡 {name} signal lost — last seen {age_hours:.0f}h ago")
                except Exception:
                    pass

            return alerts

        try:
            alert_users_res = sb.table('user_profiles')\
                .select('id, username, callmebot_phone, callmebot_apikey')\
                .eq('callmebot_enabled', True).execute()

            for user in (alert_users_res.data or []):
                if not user.get('callmebot_phone') or not user.get('callmebot_apikey'):
                    continue

                user_imos_res = sb.table('tracked_imos')\
                    .select('imo')\
                    .eq('user_id', user['id']).execute()

                for row in (user_imos_res.data or []):
                    imo = str(row['imo'])
                    # Use already-fetched data from batch instead of re-querying DB
                    vessel_data = all_vessel_data.get(imo)
                    if not vessel_data:
                        continue

                    user_alerts = _check_vessel_alerts(imo, vessel_data, prev_states.get(imo))

                    for alert_msg in user_alerts:
                        _send_whatsapp_alert(
                            user['callmebot_phone'],
                            user['callmebot_apikey'],
                            f"🚢 VesselTracker Alert\n{alert_msg}"
                        )
                        time.sleep(1)

        except Exception as e:
            logger.warning(f"Per-user alert processing error: {e}")

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        send_whatsapp(f"🚨 Vessel Tracker CRASHED: {str(e)[:200]}")

if __name__ == "__main__":
    main()
