#!/usr/bin/env python3
"""
Vessel Tracking Script - Supabase Edition
All original logic preserved. Only storage changed: JSON files → Supabase tables.

Changes from original:
  - Removed: file locking (not needed in GitHub Actions)
  - Removed: load_json / save_json_atomic (replaced by Supabase)
  - Removed: local file paths
  - Added: Supabase client for all reads and writes
  - Added: port calls refresh pass (post-loop, piggy-backed on MST HTML scrape)
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

RENDER_BASE = os.getenv("RENDER_BASE")
API_SECRET  = os.getenv("API_SECRET", "")
WORKER_URL  = os.getenv("WORKER_URL", "https://vesseltracker.asmahri1.workers.dev")

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
API_TIMEOUT_SEC        = 120
MAX_IMO_FAILURES       = 5
BATCH_SIZE             = 5    # pause after every N vessels
BATCH_COOLDOWN_SEC     = 10   # seconds to wait between batches
ALERT_COOLDOWN_MIN     = 45

PORT_CALLS_CACHE_DAYS  = 7    # re-fetch port calls after this many days

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

def db_get_port_calls_age_days(imo: str) -> Optional[float]:
    """Return age in days of the cached port calls for this IMO, or None if absent."""
    try:
        res = (
            sb.table("port_calls")
            .select("updated_at")
            .eq("imo", imo)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not res.data:
            return None
        updated = datetime.fromisoformat(res.data[0]["updated_at"].replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - updated).total_seconds() / 86400
    except Exception as e:
        logger.debug(f"Port calls age check failed for IMO {imo}: {e}")
        return None

def _apply_port_calls_cap(imo: str, cap: int = 7):
    """Keep only the `cap` most recent rows by arrived. Deletes oldest regardless of source."""
    try:
        res = (
            sb.table("port_calls")
            .select("id,arrived")
            .eq("imo", imo)
            .order("arrived", desc=True)
            .execute()
        )
        rows = res.data or []
        if len(rows) <= cap:
            return
        to_delete = [r["id"] for r in rows[cap:]]
        if to_delete:
            sb.table("port_calls").delete().in_("id", to_delete).execute()
            logger.info(f"IMO {imo}: cap trimmed {len(to_delete)} old port call(s)")
    except Exception as e:
        logger.warning(f"_apply_port_calls_cap failed for IMO {imo}: {e}")


def db_save_port_calls(imo: str, calls: List[Dict]) -> bool:
    """
    Replace MST rows for this IMO with fresh scraper data.
    Manual rows (is_manual=true) are never touched.
    After insert, apply 7-row cap across all rows (MST + manual).
    Insert-before-delete pattern prevents data loss if insert fails.
    """
    if not calls:
        return False
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "imo":        imo,
                "port_name":  pc.get("port_name") or "",
                "country":    pc.get("country")   or "",
                "arrived":    pc.get("arrived"),
                "departed":   pc.get("departed"),
                "duration":   pc.get("duration")  or "",
                "is_manual":  False,
                "updated_at": now,
            }
            for pc in calls
        ]
        # Insert new rows first — if this fails, old data is still intact
        sb.table("port_calls").insert(rows).execute()
        # Delete previous MST rows only (manual rows untouched)
        sb.table("port_calls").delete()            .eq("imo", imo)            .eq("is_manual", False)            .neq("updated_at", now)            .execute()
        # Enforce 7-row cap across all rows (MST + manual)
        _apply_port_calls_cap(imo)
        logger.info(f"IMO {imo}: {len(rows)} MST port call(s) saved (manual entries preserved)")
        return True
    except Exception as e:
        logger.warning(f"db_save_port_calls failed for IMO {imo}: {e}")
        return False

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
# PUSH HELPER (module-level, called from send_whatsapp and main)
# =============================================================================

def send_push_via_worker(user_ids, title: str, body: str, imo: str = None, alert_type: str = "alert"):
    """Send push notification to one or more users via Cloudflare Worker /push/send."""
    if not WORKER_URL or not API_SECRET:
        return
    if isinstance(user_ids, str):
        user_ids = [user_ids]
    if not user_ids:
        return
    try:
        payload = {
            "user_ids": user_ids,
            "title": title,
            "body": body,
            "tag": f"vt-{imo or 'system'}-{int(time.time())}",
            "imo": imo,
            "type": alert_type,
        }
        r = requests.post(
            f"{WORKER_URL}/push/send",
            json=payload,
            headers={"X-API-Secret": API_SECRET, "Content-Type": "application/json"},
            timeout=15,
        )
        if r.ok:
            data = r.json()
            logger.info(f"Push: {data.get('sent', 0)} delivered, {data.get('failed', 0)} failed to {len(user_ids)} user(s)")
        else:
            logger.warning(f"Push send failed: HTTP {r.status_code}")
    except Exception as e:
        logger.warning(f"Push send error: {e}")

# =============================================================================
# WHATSAPP (unchanged, now also sends push to admin)
# =============================================================================

def send_whatsapp(text: str) -> bool:
    """Send admin alert via WhatsApp AND push notification."""
    sent = False
    if CALLMEBOT_ENABLED:
        try:
            r = requests.get(CALLMEBOT_API_URL, params={
                "phone": CALLMEBOT_PHONE,
                "apikey": CALLMEBOT_APIKEY,
                "text": text
            }, timeout=15)
            sent = r.status_code == 200
        except Exception as e:
            logger.error(f"WhatsApp send failed: {e}")

    # Also send push to admin via Worker (worker looks up asmahri's user_id)
    try:
        if WORKER_URL and API_SECRET:
            # Get admin user_id from Supabase
            admin_res = sb.table('user_profiles')\
                .select('id').eq('username', 'asmahri').execute()
            if admin_res.data:
                send_push_via_worker(
                    user_ids=admin_res.data[0]['id'],
                    title='🚢 VesselTracker Admin',
                    body=text[:200],
                    alert_type='admin',
                )
    except Exception as e:
        logger.warning(f"Admin push failed: {e}")

    return sent

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

def fetch_with_retry(url: str) -> Optional[Dict]:
    headers: Dict[str, str] = {"User-Agent": "VesselTracker/1.0"}
    if API_SECRET:
        headers["X-API-Secret"] = API_SECRET
    for attempt in range(API_MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, timeout=API_TIMEOUT_SEC)
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
        result["mmsi"]        = api_data.get("mmsi") or result.get("mmsi")

        static_keys = ["mmsi", "deadweight_t", "gross_tonnage", "year_of_build", "length_overall_m", "beam_m", "draught_m"]
        for key in static_keys:
            if api_data.get(key) is not None:
                result[key] = api_data[key]
            elif result.get(key) is None:
                result[key] = None

        # Update static cache in Supabase (includes mmsi now)
        cache_entry = {k: result.get(k) for k in static_keys + ["name", "ship_type", "flag"]}
        db_save_static_cache(imo, cache_entry)

        # Port calls are populated when Tier 3 HTML scrape fired on Render.
        # If present, store them so main() can save to Supabase in one place.
        result["port_calls"] = api_data.get("port_calls") or []

    result["imo"] = imo
    return result

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
# API HEALTH CHECK (Oracle is always alive — short retry loop)
# =============================================================================
def wait_for_render(max_wait_sec: int = 30) -> bool:
    logger.info("Checking API health...")
    interval = 2
    for attempt in range(max_wait_sec // interval):
        try:
            r = requests.get(f"{RENDER_BASE}/ping", timeout=5)
            if r.ok:
                logger.info(f"API ready after ~{attempt * interval}s")
                return True
        except Exception:
            pass
        logger.info(f"API not ready yet, retrying in {interval}s...")
        time.sleep(interval)
    logger.error("API did not become ready in time.")
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

        logger.info(f"Tracking {len(imos)} vessels | {len(ports)} ports loaded | batch_size={BATCH_SIZE} cooldown={BATCH_COOLDOWN_SEC}s")

        alerts_by_imo: Dict[str, str] = {}  # imo → alert message, populated during main loop
        mmsi_map:      Dict[str, str] = {}  # imo → mmsi, built during main loop for port calls pass

        processed = 0
        for imo in imos:
            imo = str(imo).strip()
            if not validate_imo(imo):
                logger.warning(f"Invalid IMO skipped: {imo}")
                continue

            fails = failure_counts.get(imo, 0)
            if fails >= MAX_IMO_FAILURES:
                logger.warning(f"IMO {imo} skipped after {fails} consecutive failures.")
                continue

            v_data = fetch_vessel_data(imo, static_cache)
            processed += 1

            # Track MMSI in memory for the port calls pass below
            if v_data.get("mmsi"):
                mmsi_map[imo] = str(v_data["mmsi"])

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

            # If vessel-full response included port calls (Tier 3 HTML scrape fired on
            # Render), save them now.  This avoids a second MST HTTP request later.
            if v_data.get("port_calls"):
                db_save_port_calls(imo, v_data["port_calls"])
                logger.info(f"IMO {imo}: port calls saved from vessel-full response ({len(v_data['port_calls'])} entries)")

            if alert:
                logger.info(f"Alert triggered for {imo}:\n{alert}")
                alerts_by_imo[imo] = alert  # stored for per-user dispatch below

            # Cooldown after every BATCH_SIZE vessels (skip after last vessel)
            if processed % BATCH_SIZE == 0 and imo != imos[-1]:
                logger.info(f"Batch cooldown after {processed} vessels — sleeping {BATCH_COOLDOWN_SEC}s...")
                time.sleep(BATCH_COOLDOWN_SEC)

        logger.info("Tracking run completed successfully.")

        # ─────────────────────────────────────────────────────────────────────
        # PORT CALLS — refresh stale entries (post-loop)
        #
        # How this avoids double-fetching MST:
        #   • If Tier 3 HTML scrape fired during vessel-full, port calls were
        #     already saved above.  db_get_port_calls_age_days() will return a
        #     very small value → we skip → no extra HTTP request.
        #   • If Tier 1 or Tier 2 handled position (no HTML fetch), port calls
        #     are stale → we call /port-calls/{imo} on Render, which fetches the
        #     MST HTML page exactly once and returns both pos data (ignored) and
        #     port calls.  We then save them here.
        # ─────────────────────────────────────────────────────────────────────
        logger.info("Starting port calls refresh pass...")
        for imo in imos:
            imo = str(imo).strip()
            if not validate_imo(imo):
                continue

            age = db_get_port_calls_age_days(imo)
            if age is not None and age < PORT_CALLS_CACHE_DAYS:
                logger.info(f"IMO {imo}: port calls fresh ({age:.1f}d old), skipping")
                continue

            # Prefer MMSI collected this run; fall back to static_cache
            mmsi = mmsi_map.get(imo) or str(static_cache.get(imo, {}).get("mmsi") or "")
            if not mmsi:
                logger.warning(f"IMO {imo}: no MMSI available, cannot fetch port calls")
                continue

            result = fetch_with_retry(f"{RENDER_BASE}/port-calls/{imo}?mmsi={mmsi}")
            if result and result.get("count", 0) > 0:
                db_save_port_calls(imo, result["port_calls"])
                logger.info(f"IMO {imo}: {result['count']} port calls saved via /port-calls endpoint")
            else:
                logger.info(f"IMO {imo}: no port calls returned from /port-calls endpoint")

            time.sleep(3)   # polite delay between vessels

        # ─────────────────────────────────────────────────────────────────────
        # PER-USER CALLMEBOT ALERTS (multi-user support)
        # ─────────────────────────────────────────────────────────────────────
        if not alerts_by_imo:
            logger.info("No alerts this run — skipping per-user dispatch.")
        else:
            logger.info(f"Dispatching alerts for {len(alerts_by_imo)} vessel(s) to subscribed users...")

            import urllib.parse as _urllib_parse

            def _send_whatsapp_alert(phone: str, apikey: str, message: str):
                try:
                    encoded_msg = _urllib_parse.quote(message)
                    url = (
                        f"https://api.callmebot.com/whatsapp.php"
                        f"?phone={_urllib_parse.quote(phone)}"
                        f"&text={encoded_msg}"
                        f"&apikey={_urllib_parse.quote(apikey)}"
                    )
                    r = requests.get(url, timeout=10)
                    logger.info(f"CallMeBot sent to {phone}: HTTP {r.status_code}")
                except Exception as e:
                    logger.warning(f"CallMeBot failed for {phone}: {e}")

            # ── WhatsApp alerts (CallMeBot — existing behavior, unchanged) ──
            try:
                alert_users_res = sb.table('user_profiles')\
                    .select('id, username, callmebot_phone, callmebot_apikey')\
                    .eq('callmebot_enabled', True).execute()

                for user in (alert_users_res.data or []):
                    phone  = user.get('callmebot_phone')
                    apikey = user.get('callmebot_apikey')
                    if not phone or not apikey:
                        continue

                    user_imos_res = sb.table('tracked_imos')\
                        .select('imo')\
                        .eq('user_id', user['id']).execute()

                    user_imo_set = {str(r['imo']) for r in (user_imos_res.data or [])}
                    matched = user_imo_set & alerts_by_imo.keys()

                    if not matched:
                        logger.info(f"No alerts for user '{user.get('username')}' this run.")
                        continue

                    for imo in matched:
                        _send_whatsapp_alert(phone, apikey, alerts_by_imo[imo])
                        time.sleep(1)

            except Exception as e:
                logger.warning(f"Per-user WhatsApp alert processing error: {e}")

            # ── Push notifications (all users, independent of CallMeBot) ──────
            try:
                alert_imo_list = list(alerts_by_imo.keys())
                # Get ALL users who track any alerting vessel
                tracked_res = sb.table('tracked_imos')\
                    .select('imo, user_id')\
                    .in_('imo', alert_imo_list).execute()

                # Group by IMO → set of user_ids
                imo_to_users = {}
                for row in (tracked_res.data or []):
                    uid = row.get('user_id')
                    if not uid:
                        continue
                    imo_str = str(row['imo'])
                    if imo_str not in imo_to_users:
                        imo_to_users[imo_str] = set()
                    imo_to_users[imo_str].add(uid)

                # Send push per IMO (each vessel alert → all users tracking it)
              for imo, user_id_set in imo_to_users.items():
                  alert_msg = alerts_by_imo[imo]
                  # Use the first line as the push notification title
                  title = alert_msg.split('\n')[0] if alert_msg else 'Vessel Alert'
                  # Use the whole message as the body (preserving line breaks)
                  body = alert_msg
              
                  send_push_via_worker(
                      user_ids=list(user_id_set),
                      title=title,
                      body=body,
                      imo=imo,
                      alert_type="vessel_alert",
                  )

            except Exception as e:
                logger.warning(f"Per-user push alert processing error: {e}")

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        send_whatsapp(f"🚨 Vessel Tracker CRASHED: {str(e)[:200]}")

if __name__ == "__main__":
    main()
