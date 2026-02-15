#!/usr/bin/env python3
"""
Vessel Tracking Script - Production Ready Version
Tracks vessels via AIS data, calculates ETAs, and sends WhatsApp alerts.

Features:
- Fetches live AIS data from Render API with retry logic
- Maintains persistent state for vessel tracking
- Calculates ETA to destination ports
- Sends WhatsApp notifications on position/destination changes
- Caches static vessel specifications
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
ARRIVAL_RADIUS_NM = 35.5
MIN_MOVE_NM = 5.0
MIN_SOG_FOR_ETA = 0.5
MAX_ETA_HOURS = 240
MAX_ETA_SOG_CAP = 18.0
MAX_AIS_FOR_ETA_MIN = 360
MIN_DISTANCE_FOR_ETA = 5.0

# Arrival detection threshold (speed below this = stopped)
ARRIVAL_SOG_THRESHOLD = 0.5

# API retry configuration
API_MAX_RETRIES = 3
API_RETRY_BACKOFF_BASE = 2.0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Safely convert a value to float.
    Handles None, empty strings, and invalid values gracefully.
    
    Args:
        value: The value to convert
        default: Default value to return if conversion fails
        
    Returns:
        Float value or default
    """
    if value is None:
        return default
    if value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def load_json(path: Path, default: Any) -> Any:
    """
    Load JSON from file with error handling.
    
    Args:
        path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Parsed JSON data or default value
    """
    if not path.exists():
        logger.debug(f"File not found: {path}, using default")
        return default
    
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        return default
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return default


def save_json(path: Path, data: Any) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        path: Path to save JSON file
        data: Data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Successfully saved: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")
        return False


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)
        
    Returns:
        Distance in nautical miles
    """
    R = 6371.0  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    distance_km = R * 2 * math.asin(math.sqrt(a))
    
    # Convert km to nautical miles (1 km ≈ 0.539957 NM)
    return distance_km * 0.539957


def validate_imo(imo: str) -> bool:
    """
    Validate IMO number format.
    IMO numbers are 7 digits with a checksum digit.
    
    Args:
        imo: IMO number string
        
    Returns:
        True if valid, False otherwise
    """
    imo = imo.strip()
    
    # Must be exactly 7 digits
    if not re.match(r'^\d{7}$', imo):
        return False
    
    # Checksum validation: last digit = sum(digit[i] * (7-i)) mod 10
    try:
        total = sum(int(imo[i]) * (7 - i) for i in range(6))
        return int(imo[6]) == total % 10
    except (ValueError, IndexError):
        return False


# =============================================================================
# PORT HANDLING
# =============================================================================

def load_ports() -> Dict[str, Dict]:
    """
    Load ports data from JSON file.
    
    Returns:
        Dictionary mapping port names to port data
        
    Raises:
        RuntimeError: If ports.json is missing or empty
    """
    ports = load_json(PORTS_PATH, {})
    if not ports:
        raise RuntimeError(f"ports.json missing or empty at {PORTS_PATH}")
    return {k.upper(): v for k, v in ports.items()}


def normalize_string(s: str) -> str:
    """
    Normalize a string for port matching.
    Removes accents, non-letter characters, and converts to uppercase.
    
    Args:
        s: Input string
        
    Returns:
        Normalized string (uppercase letters only)
    """
    if not s:
        return ""
    
    # Normalize to NFD and remove combining marks (accents)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    
    # Keep only letters and convert to uppercase
    return re.sub(r"[^A-Z]", "", s.upper())


# Port name aliases for flexible matching
ALIASES_RAW = {
    # Morocco - Southern Provinces
    "laayoune": "LAAYOUNE", "layoune": "LAAYOUNE", "EH EUN": "LAAYOUNE", "leyoune": "LAAYOUNE",
    "tantan": "TAN TAN", "tan tan": "TAN TAN", "tan-tan": "TAN TAN", "tan tan anch": "TAN TAN",
    "dakhla": "DAKHLA", "dakhla port": "DAKHLA", "ad dakhla": "DAKHLA",
    "dakhla anch": "DAKHLA ANCH", "dakhla anch.": "DAKHLA ANCH", "dakhla anchorage": "DAKHLA ANCH",
    "dakhla anch area": "DAKHLA ANCH",
    
    # Morocco - Atlantic Coast
    "agadir": "AGADIR", "port agadir": "AGADIR",
    "essaouira": "ESSAOUIRA", "safi": "SAFI",
    "casa": "CASABLANCA", "casablanca": "CASABLANCA", "cassablanca": "CASABLANCA",
    "mohammedia": "MOHAMMEDIA",
    "jorf": "JORF LASFAR", "jorf lasfar": "JORF LASFAR",
    "kenitra": "KENITRA",
    
    # Morocco - Mediterranean
    "tanger": "TANGER VILLE", "tangier": "TANGER VILLE", "tanger ville": "TANGER VILLE",
    "tanger med": "TANGER MED", "tm2": "TANGER MED",
    "nador": "NADOR",
    "al hoceima": "AL HOCEIMA", "alhucemas": "AL HOCEIMA",
    
    # Canary Islands
    "las palmas": "LAS PALMAS", "lpa": "LAS PALMAS", "las palmas anch": "LAS PALMAS",
    "arrecife": "ARRECIFE",
    "puerto del rosario": "PUERTO DEL ROSARIO", "pdr": "PUERTO DEL ROSARIO",
    "santa cruz": "SANTA CRUZ DE TENERIFE", "sctf": "SANTA CRUZ DE TENERIFE", 
    "santa cruz tenerife": "SANTA CRUZ DE TENERIFE",
    "san sebastian": "SAN SEBASTIAN DE LA GOMERA",
    "la restinga": "LA RESTINGA",
    "la palma": "LA PALMA",
    "granadilla": "GRANADILLA", "puerto de granadilla": "GRANADILLA",
    
    # Spain - Mainland & Enclaves
    "ceuta": "CEUTA", "melilla": "MELILLA",
    "algeciras": "ALGECIRAS", "alg": "ALGECIRAS",
    "gibraltar": "GIBRALTAR", "gib": "GIBRALTAR",
    "huelva": "HUELVA",
    "huelva anch": "HUELVA ANCH", "huelva anchorage": "HUELVA ANCH",
    "cadiz": "CADIZ", "cadiz anch": "CADIZ",
    "sevilla": "SEVILLA", "seville": "SEVILLA",
    "malaga": "MALAGA", "motril": "MOTRIL", "almeria": "ALMERIA",
    "cartagena": "CARTAGENA", "valencia": "VALENCIA",
    
    # Portugal
    "sines": "SINES", "setubal": "SETUBAL", "lisbon": "LISBON", "lisboa": "LISBON"
}

# Build normalized alias lookup
DEST_ALIASES = {normalize_string(k): v for k, v in ALIASES_RAW.items()}


def match_destination_port(dest: str, ports: Dict[str, Dict]) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Match a destination string to a known port.
    
    Args:
        dest: Destination string from AIS
        ports: Dictionary of port data
        
    Returns:
        Tuple of (matched_port_name, port_data) or (None, None) if no match
    """
    if not dest:
        return None, None
    
    normalized = normalize_string(dest)
    
    # Check aliases first (highest priority)
    if normalized in DEST_ALIASES:
        canonical_name = DEST_ALIASES[normalized]
        return canonical_name, ports.get(canonical_name)
    
    # Build normalized port name lookup
    port_lookup = {normalize_string(p): p for p in ports}
    
    # Exact match on port name
    if normalized in port_lookup:
        port_name = port_lookup[normalized]
        return port_name, ports.get(port_name)
    
    # Partial match: destination contains port name
    for canonical_port, port_name in port_lookup.items():
        if canonical_port and canonical_port in normalized:
            return port_name, ports.get(port_name)
    
    return None, None


def nearest_port(lat: float, lon: float, ports: Dict[str, Dict]) -> Tuple[Optional[str], Optional[float]]:
    """
    Find the nearest port to given coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        ports: Dictionary of port data
        
    Returns:
        Tuple of (port_name, distance_nm) or (None, None) if no ports
    """
    nearest_name = None
    nearest_distance = None
    
    for name, coords in ports.items():
        try:
            distance = haversine_nm(lat, lon, coords["lat"], coords["lon"])
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_name = name
        except (KeyError, TypeError) as e:
            logger.warning(f"Invalid coordinates for port {name}: {e}")
            continue
    
    return nearest_name, nearest_distance


# =============================================================================
# TIME HANDLING
# =============================================================================

def parse_ais_time(time_str: str) -> Optional[datetime]:
    """
    Parse AIS timestamp string to datetime object.
    
    Args:
        time_str: Timestamp string in format "Mon DD, YYYY HH:MM UTC"
        
    Returns:
        Datetime object with UTC timezone, or None if parsing fails
    """
    if not time_str:
        return None
    
    # Remove " UTC" suffix if present and strip whitespace
    time_str = time_str.replace(" UTC", "").strip()
    
    try:
        dt = datetime.strptime(time_str, "%b %d, %Y %H:%M")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        logger.debug(f"Failed to parse AIS time '{time_str}': {e}")
        return None


def age_minutes(time_str: str) -> Optional[float]:
    """
    Calculate the age of a timestamp in minutes.
    
    Args:
        time_str: AIS timestamp string
        
    Returns:
        Age in minutes, or None if timestamp is invalid
    """
    dt = parse_ais_time(time_str)
    if not dt:
        return None
    
    now = datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 60


# =============================================================================
# COMMUNICATION
# =============================================================================

def send_whatsapp_message(text: str) -> bool:
    """
    Send a WhatsApp message via CallMeBot API.
    
    Args:
        text: Message text to send
        
    Returns:
        True if successful, False otherwise
    """
    if not CALLMEBOT_ENABLED:
        logger.debug("WhatsApp notifications disabled (missing credentials)")
        return False
    
    try:
        response = requests.get(
            CALLMEBOT_API_URL,
            params={
                "phone": CALLMEBOT_PHONE,
                "apikey": CALLMEBOT_APIKEY,
                "text": text
            },
            timeout=20
        )
        response.raise_for_status()
        logger.info("WhatsApp message sent successfully")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send WhatsApp message: {e}")
        return False


# =============================================================================
# API COMMUNICATION
# =============================================================================

def fetch_with_retry(url: str, timeout: int = 60) -> Optional[Dict]:
    """
    Fetch data from API with exponential backoff retry.
    
    Args:
        url: API URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        JSON response data or None if all retries fail
    """
    last_exception = None
    
    for attempt in range(API_MAX_RETRIES):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout as e:
            last_exception = e
            logger.warning(f"API timeout (attempt {attempt + 1}/{API_MAX_RETRIES}): {url}")
        except requests.exceptions.HTTPError as e:
            last_exception = e
            logger.warning(f"HTTP error (attempt {attempt + 1}/{API_MAX_RETRIES}): {e}")
        except requests.exceptions.RequestException as e:
            last_exception = e
            logger.warning(f"Request error (attempt {attempt + 1}/{API_MAX_RETRIES}): {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {url}: {e}")
            return None
        
        # Wait before retry (exponential backoff)
        if attempt < API_MAX_RETRIES - 1:
            sleep_time = API_RETRY_BACKOFF_BASE * (2 ** attempt)
            logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    logger.error(f"All {API_MAX_RETRIES} API attempts failed for {url}")
    return None


def fetch_from_render_api(imo: str, static_cache: Dict) -> Dict:
    """
    Fetch vessel data from Render API, with static cache fallback.
    
    Args:
        imo: IMO number of the vessel
        static_cache: Static vessel data cache
        
    Returns:
        Dictionary containing vessel data (merged API + cache data)
    """
    # Initialize with cached static data
    result = static_cache.get(imo, {}).copy()
    
    # Fetch live data from API
    api_data = fetch_with_retry(f"{RENDER_BASE}/vessel-full/{imo}")
    
    if api_data is None:
        logger.warning(f"API fetch failed for IMO {imo}, using cached data")
        return result if result else {}
    
    # Check for explicit "not found" response
    if api_data.get("found") is False:
        logger.info(f"IMO {imo} not found in API")
        return result if result else {}
    
    # Merge API data into result
    if api_data:
        # Dynamic fields (position, movement)
        result["lat"] = safe_float(api_data.get("lat"), result.get("lat"))
        result["lon"] = safe_float(api_data.get("lon"), result.get("lon"))
        result["sog"] = safe_float(api_data.get("sog"), 0.0) or 0.0
        result["cog"] = safe_float(api_data.get("cog"), 0.0) or 0.0
        result["last_pos_utc"] = api_data.get("last_pos_utc")
        result["destination"] = (api_data.get("destination") or "").strip()
        
        # Static fields (vessel specs) - prioritize API, fallback to cache
        result["name"] = (
            api_data.get("vessel_name") or 
            api_data.get("name") or 
            result.get("name") or 
            f"IMO {imo}"
        ).strip()
        result["ship_type"] = (api_data.get("ship_type") or result.get("ship_type") or "").strip()
        result["flag"] = (api_data.get("flag") or result.get("flag") or "").strip()
        result["deadweight_t"] = api_data.get("deadweight_t") if api_data.get("deadweight_t") is not None else result.get("deadweight_t")
        result["gross_tonnage"] = api_data.get("gross_tonnage") if api_data.get("gross_tonnage") is not None else result.get("gross_tonnage")
        result["year_of_build"] = api_data.get("year_of_build") if api_data.get("year_of_build") is not None else result.get("year_of_build")
        result["length_overall_m"] = api_data.get("length_overall_m") if api_data.get("length_overall_m") is not None else result.get("length_overall_m")
        result["beam_m"] = api_data.get("beam_m") if api_data.get("beam_m") is not None else result.get("beam_m")
    
    # Ensure IMO is set
    result["imo"] = imo
    
    return result


# =============================================================================
# ETA CALCULATION
# =============================================================================

def humanize_eta(hours: float) -> str:
    """
    Convert hours to human-readable ETA string.
    
    Args:
        hours: Hours (can be fractional)
        
    Returns:
        Human-readable string like "2h 30m" or "3d 4h"
    """
    total_minutes = int(round(hours * 60))
    h = total_minutes // 60
    m = total_minutes % 60
    
    if h < 24:
        if m > 0:
            return f"{h}h {m}m"
        return f"{h}h"
    
    days = h // 24
    remaining_hours = h % 24
    
    if remaining_hours > 0:
        return f"{days}d {remaining_hours}h"
    return f"{days}d"


# =============================================================================
# STATE AND ALERT BUILDING
# =============================================================================

def build_alert_and_state(
    vessel_data: Dict, 
    ports: Dict[str, Dict], 
    prev_state: Optional[Dict]
) -> Tuple[Optional[str], Dict]:
    """
    Build alert message and new state for a vessel.
    
    Args:
        vessel_data: Current vessel data from API/cache
        ports: Port data dictionary
        prev_state: Previous state of this vessel (or None if new)
        
    Returns:
        Tuple of (alert_message, new_state)
        alert_message is None if no alert should be sent
    """
    # Extract vessel identification
    imo = vessel_data["imo"]
    name = vessel_data.get("name") or f"IMO {imo}"
    
    # Extract static specs
    ship_type = vessel_data.get("ship_type")
    flag = vessel_data.get("flag")
    dwt = vessel_data.get("deadweight_t")
    gt = vessel_data.get("gross_tonnage")
    year = vessel_data.get("year_of_build")
    length = vessel_data.get("length_overall_m")
    beam = vessel_data.get("beam_m")
    
    # Build base state with static data (always preserved)
    base_state = {
        "imo": imo,
        "name": name,
        "ship_type": ship_type,
        "flag": flag,
        "deadweight_t": dwt,
        "gross_tonnage": gt,
        "year_of_build": year,
        "length_overall_m": length,
        "beam_m": beam,
        "destination": vessel_data.get("destination", ""),
    }
    
    # Extract position data
    lat = vessel_data.get("lat")
    lon = vessel_data.get("lon")
    sog = vessel_data.get("sog")
    cog = vessel_data.get("cog")
    last_pos = vessel_data.get("last_pos_utc")
    
    # If key position data is missing, return state without alerts
    if lat is None or lon is None:
        logger.debug(f"No position data for {name} (IMO {imo})")
        return None, {
            **base_state,
            "lat": None, "lon": None, "sog": None, "cog": None,
            "last_pos_utc": last_pos,
            "nearest_port": None, "nearest_distance_nm": None,
            "destination_port": None, "destination_distance_nm": None,
            "eta_hours": None, "eta_utc": None, "eta_text": None,
            "done": False,
        }
    
    # Ensure sog and cog have values
    sog = sog or 0.0
    cog = cog or 0.0
    
    # Calculate position age
    pos_age = age_minutes(last_pos) if last_pos else None
    age_text = "N/A" if pos_age is None else f"{pos_age:.0f} min ago"
    too_old = pos_age is not None and pos_age > MAX_AIS_FOR_ETA_MIN
    
    # Find nearest port
    nearest_name, nearest_dist = nearest_port(lat, lon, ports)
    nearest_display = nearest_name or "N/A"
    nearest_dist_text = "N/A" if nearest_dist is None else f"{nearest_dist:.1f} NM"
    
    # Match destination port
    dest = vessel_data.get("destination", "")
    dest_name, dest_data = match_destination_port(dest, ports)
    dest_dist = None
    if dest_name and dest_data:
        try:
            dest_dist = haversine_nm(lat, lon, dest_data["lat"], dest_data["lon"])
        except (KeyError, TypeError):
            pass
    
    # Calculate ETA
    eta_hours = None
    eta_utc_str = None
    eta_text = None
    
    can_calculate_eta = (
        dest_dist is not None and
        dest_dist > MIN_DISTANCE_FOR_ETA and
        sog >= MIN_SOG_FOR_ETA and
        not too_old
    )
    
    if can_calculate_eta:
        # Cap speed for ETA calculation to avoid unrealistic estimates
        effective_sog = min(max(sog, MIN_SOG_FOR_ETA), MAX_ETA_SOG_CAP)
        
        try:
            raw_hours = dest_dist / effective_sog
            if raw_hours <= MAX_ETA_HOURS:
                eta_hours = raw_hours
                eta_dt = datetime.now(timezone.utc) + timedelta(hours=eta_hours)
                eta_utc_str = eta_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                eta_text = humanize_eta(eta_hours)
        except (ZeroDivisionError, ValueError):
            pass
    
    # Build complete new state
    new_state = {
        **base_state,
        "lat": lat,
        "lon": lon,
        "sog": sog,
        "cog": cog,
        "last_pos_utc": last_pos,
        "nearest_port": nearest_name,
        "nearest_distance_nm": nearest_dist,
        "destination_port": dest_name,
        "destination_distance_nm": dest_dist,
        "eta_hours": eta_hours,
        "eta_utc": eta_utc_str,
        "eta_text": eta_text,
        "done": False,
    }
    
    # Check for arrival (within radius and stopped)
    arrived = (
        nearest_dist is not None and
        nearest_dist <= ARRIVAL_RADIUS_NM and
        sog <= ARRIVAL_SOG_THRESHOLD
    )
    if arrived:
        new_state["done"] = True
    
    # ==================================================
    # ALERT GENERATION
    # ==================================================
    
    # First tracking detection (no previous state)
    if not prev_state:
        msg_lines = [
            f"🚢 {name} (IMO {imo})",
            "📌 Status: First tracking detected",
            f"🕒 AIS: {age_text}",
            f"⚡ Speed: {sog:.1f} kn | 🧭 {cog:.0f}°",
            f"📍 Position: {lat:.4f}, {lon:.4f}",
            f"⚓ Nearest port: {nearest_display} (~{nearest_dist_text})",
            f"🎯 Destination: {dest or 'N/A'}",
        ]
        
        if dest_name and dest_dist is not None:
            msg_lines[-1] += f" (~{dest_dist:.1f} NM)"
        
        if eta_text:
            msg_lines.append(f"⏱ ETA: {eta_text} ({eta_utc_str})")
        
        return "\n".join(msg_lines), new_state
    
    # Already marked as done (arrived) - no more alerts
    if prev_state.get("done"):
        return None, new_state
    
    # Check for changes
    old_dest = (prev_state.get("destination") or "").strip()
    dest_changed = (old_dest.upper() != dest.upper()) if (old_dest or dest) else False
    
    # Check for movement
    prev_lat = prev_state.get("lat")
    prev_lon = prev_state.get("lon")
    moved = False
    move_distance = None
    
    if prev_lat is not None and prev_lon is not None:
        move_distance = haversine_nm(prev_lat, prev_lon, lat, lon)
        moved = move_distance >= MIN_MOVE_NM
    
    # Check for arrival event
    arrival_event = arrived and not prev_state.get("done")
    
    # No significant changes - no alert
    if not (dest_changed or moved or arrival_event):
        return None, new_state
    
    # Build status text
    if arrival_event:
        status = "Arrived at destination area"
    elif dest_changed:
        status = "Destination changed"
    else:
        status = "Position / track updated"
    
    # Build movement indicator
    movement_indicator = f" (Δ {move_distance:.1f} NM)" if moved and move_distance is not None else ""
    
    # Build destination line
    if dest_changed and old_dest:
        dest_line = f"🎯 Destination changed: {old_dest} ➜ {dest or 'N/A'}"
    else:
        dest_line = f"🎯 Destination: {dest or 'N/A'}"
    
    if dest_name and dest_dist is not None:
        dest_line += f" (~{dest_dist:.1f} NM)"
    
    # Build alert message
    msg_lines = [
        f"🚢 {name} (IMO {imo})",
        f"📌 Status: {status}",
        f"🕒 AIS: {age_text}",
        f"⚡ Speed: {sog:.1f} kn | 🧭 {cog:.0f}°{movement_indicator}",
        f"📍 Position: {lat:.4f}, {lon:.4f}",
        f"⚓ Nearest port: {nearest_display} (~{nearest_dist_text})",
        dest_line,
    ]
    
    if eta_text:
        msg_lines.append(f"⏱ ETA: {eta_text} ({eta_utc_str})")
    
    return "\n".join(msg_lines), new_state


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def ping_api() -> bool:
    """
    Ping the Render API to wake it up.
    
    Returns:
        True if ping successful, False otherwise
    """
    try:
        response = requests.get(f"{RENDER_BASE}/ping", timeout=30)
        response.raise_for_status()
        logger.debug("API ping successful")
        return True
    except requests.exceptions.RequestException as e:
        logger.warning(f"API ping failed: {e}")
        return False


def main() -> None:
    """
    Main execution function.
    Orchestrates vessel tracking, state updates, and notifications.
    """
    logger.info("=" * 60)
    logger.info("Starting vessel tracking run")
    logger.info("=" * 60)
    
    # 1. Ping API to wake it up (best effort)
    ping_api()
    
    # 2. Load static vessel cache
    static_cache = load_json(STATIC_CACHE_PATH, {})
    logger.info(f"Loaded static cache with {len(static_cache)} vessels")
    
    # 3. Load tracked IMO list
    imos_data = load_json(TRACKED_IMOS_PATH, [])
    
    # Handle different JSON formats
    if isinstance(imos_data, dict) and "tracked_imos" in imos_data:
        imos = imos_data["tracked_imos"]
    elif isinstance(imos_data, list):
        imos = imos_data
    else:
        imos = []
    
    # Validate and clean IMO list
    imos = [str(i).strip() for i in imos if str(i).strip()]
    valid_imos = [imo for imo in imos if validate_imo(imo)]
    
    if len(imos) != len(valid_imos):
        invalid_count = len(imos) - len(valid_imos)
        logger.warning(f"Filtered out {invalid_count} invalid IMO numbers")
    
    if not valid_imos:
        logger.warning("No valid IMO numbers to track")
        return
    
    logger.info(f"Tracking {len(valid_imos)} vessels")
    
    # 4. Load ports
    try:
        ports = load_ports()
        logger.info(f"Loaded {len(ports)} ports")
    except RuntimeError as e:
        logger.error(f"Failed to load ports: {e}")
        return
    
    # 5. Load previous state
    prev_state = load_json(VESSELS_STATE_PATH, {})
    if not isinstance(prev_state, dict):
        logger.warning("Previous state was not a dictionary, resetting")
        prev_state = {}
    
    # 6. Process each vessel
    new_state_all = {}
    alerts_sent = 0
    
    for i, imo in enumerate(valid_imos, 1):
        logger.info(f"Processing vessel {i}/{len(valid_imos)}: IMO {imo}")
        
        # Fetch vessel data
        vessel_data = fetch_from_render_api(imo, static_cache)
        
        # If fetch completely failed, preserve previous state
        if not vessel_data:
            if imo in prev_state:
                logger.info(f"Using previous state for IMO {imo}")
                new_state_all[imo] = prev_state[imo]
            continue
        
        # Build alert and new state
        alert, new_state = build_alert_and_state(
            vessel_data, 
            ports, 
            prev_state.get(imo)
        )
        
        new_state_all[imo] = new_state
        
        # Send alert if generated
        if alert:
            logger.info(f"Sending alert for IMO {imo}")
            if send_whatsapp_message(alert):
                alerts_sent += 1
    
    # 7. Save new state
    if new_state_all:
        if save_json(VESSELS_STATE_PATH, new_state_all):
            logger.info(f"Saved state for {len(new_state_all)} vessels")
        else:
            logger.error("Failed to save vessel state")
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Tracking run complete")
    logger.info(f"  Vessels processed: {len(valid_imos)}")
    logger.info(f"  Alerts sent: {alerts_sent}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
