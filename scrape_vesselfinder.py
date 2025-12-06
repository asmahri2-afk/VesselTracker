import json, math, os, re, requests, unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- CONFIGURATION PATHS ---
TRACKED_IMOS_PATH = Path("data/tracked_imos.json")
VESSELS_STATE_PATH = Path("data/vessels_data.json")
PORTS_PATH = Path("data/ports.json")
STATIC_CACHE_PATH = Path("data/static_vessel_cache.json") # ⭐ NEW STATIC CACHE PATH

# --- EXTERNAL SERVICES CONFIG ---
CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE")
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY")
CALLMEBOT_ENABLED = bool(CALLMEBOT_PHONE and CALLMEBOT_APIKEY)
CALLMEBOT_API_URL = "https://api.callmebot.com/whatsapp.php"

RENDER_BASE = "https://vessel-api-s85s.onrender.com"

# --- TRACKING THRESHOLDS ---
MAX_AIS_MINUTES = 30
ARRIVAL_RADIUS_NM = 35.5
MIN_MOVE_NM = 5.0
MIN_SOG_FOR_ETA = 0.5
MAX_ETA_HOURS = 240
MAX_ETA_SOG_CAP = 18.0
MAX_AIS_FOR_ETA_MIN = 360
MIN_DISTANCE_FOR_ETA = 5.0

# --- UTILITY FUNCTIONS ---

def load_json(p, d):
    if not p.exists(): return d
    try:
        with p.open("r", encoding="utf-8") as f: return json.load(f)
    except: return d

def save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)

def haversine_nm(a, b, c, d):
    R = 6371.0 # Earth radius in km
    a1, b1, c1, d1 = map(math.radians, [a, b, c, d])
    da = c1 - a1
    db = d1 - b1
    x = math.sin(da / 2)**2 + math.cos(a1) * math.cos(c1) * math.sin(db / 2)**2
    # Result is km, convert to nautical miles (1 km ≈ 0.539957 NM)
    return (R * 2 * math.asin(math.sqrt(x))) * 0.539957

def load_ports():
    ports = load_json(PORTS_PATH, {})
    if not ports: raise RuntimeError("ports.json missing")
    return {k.upper(): v for k, v in ports.items()}

def _norm(s):
    if not s: return ""
    # Normalize and remove accents (NFD format, category 'Mn' is combining mark)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # Keep only letters and convert to uppercase
    return re.sub(r"[^A-Z]", "", s.upper())

ALIASES_RAW = {
 "laayoune":"LAAYOUNE","layoune":"LAAYOUNE","eh eun":"LAAYOUNE","leyoune":"LAAYOUNE",
 "tantan":"TAN TAN","tan tan":"TAN TAN","tan-tan":"TAN TAN","tan tan anch":"TAN TAN",
 "dakhla":"DAKHLA","dakhla port":"DAKHLA","ad dakhla":"DAKHLA",
 "dakhla anch":"DAKHLA ANCH","dakhla anch.":"DAKHLA ANCH","dakhla anchorage":"DAKHLA ANCH",
 "dakhla anch area":"DAKHLA ANCH",
 "agadir":"AGADIR","port agadir":"AGADIR",
 "essaouira":"ESSAOUIRA","safi":"SAFI",
 "casa":"CASABLANCA","casablanca":"CASABLANCA","cassablanca":"CASABLANCA",
 "mohammedia":"MOHAMMEDIA",
 "jorf":"JORF LASFAR","jorf lasfar":"JORF LASFAR",
 "kenitra":"KENITRA",
 "tanger":"TANGER VILLE","tangier":"TANGER VILLE","tanger ville":"TANGER VILLE",
 "tanger med":"TANGER MED","tm2":"TANGER MED",
 "nador":"NADOR",
 "al hoceima":"AL HOCEIMA","alhucemas":"AL HOCEIMA",
 "las palmas":"LAS PALMAS","lpa":"LAS PALMAS","las palmas anch":"LAS PALMAS",
 "arrecife":"ARRECIFE",
 "puerto del rosario":"PUERTO DEL ROSARIO","pdr":"PUERTO DEL ROSARIO",
 "santa cruz":"SANTA CRUZ DE TENERIFE","sctf":"SANTA CRUZ DE TENERIFE","santa cruz tenerife":"SANTA CRUZ DE TENERIFE",
 "san sebastian":"SAN SEBASTIAN DE LA GOMERA",
 "la restinga":"LA RESTINGA",
 "la palma":"LA PALMA",
 "granadilla":"GRANADILLA","puerto de granadilla":"GRANADILLA",
 "ceuta":"CEUTA","melilla":"MELILLA",
 "algeciras":"ALGECIRAS","alg":"ALGECIRAS",
 "gibraltar":"GIBRALTAR","gib":"GIBRALTAR",
 "huelva":"HUELVA",
 "huelva anch":"HUELVA ANCH","huelva anchorage":"HUELVA ANCH",
 "cadiz":"CADIZ","cadiz anch":"CADIZ",
 "sevilla":"SEVILLA","seville":"SEVILLA",
 "malaga":"MALAGA","motril":"MOTRIL","almeria":"ALMERIA",
 "cartagena":"CARTAGENA","valencia":"VALENCIA",
 "sines":"SINES","setubal":"SETUBAL","lisbon":"LISBON","lisboa":"LISBON"
}
DEST_ALIASES = {_norm(k): v for k, v in ALIASES_RAW.items()}

def match_destination_port(dest, ports):
    if not dest: return None, (None)
    n = _norm(dest)
    if n in DEST_ALIASES:
        c = DEST_ALIASES[n]; return c, ports.get(c)
    cmap = {_norm(p): p for p in ports}
    if n in cmap:
        p = cmap[n]; return p, ports.get(p)
    for canon, name in cmap.items():
        if canon and canon in n: return name, ports.get(name)
    return None, None

def nearest_port(lat, lon, ports):
    bn, bnm = None, None
    for n, c in ports.items():
        d = haversine_nm(lat, lon, c["lat"], c["lon"])
        if bnm is None or d < bnm: bnm = d; bn = n
    return bn, bnm

def parse_ais_time(s):
    if not s: return None
    s = s.replace(" UTC", "").strip()
    try:
        return datetime.strptime(s, "%b %d, %Y %H:%M").replace(tzinfo=timezone.utc)
    except: return None

def age_minutes(t):
    dt = parse_ais_time(t)
    if not dt: return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60

def send_whatsapp_message(txt):
    if not CALLMEBOT_ENABLED: return
    try:
        requests.get(CALLMEBOT_API_URL, params={"phone": CALLMEBOT_PHONE, "apikey": CALLMEBOT_APIKEY, "text": txt}, timeout=20).raise_for_status()
    except: pass

def fetch_from_render_api(imo, static_cache):
    """
    Fetches dynamic data from the Render API. 
    Uses the static_cache for vessel specs if API fetch fails or specs are missing.
    """
    
    # 1. Initialize result with static data from the persistent cache
    result = static_cache.get(imo, {}).copy() # Use .copy() to ensure 'result' is mutable
    
    # 2. Try to fetch live data
    api_data = {}
    try:
        r = requests.get(f"{RENDER_BASE}/vessel-full/{imo}", timeout=60)
        r.raise_for_status()
        api_data = r.json()
    except Exception as e:
        # If API fails, log error and proceed with only static cache data
        print(f"Warning: Failed to fetch live data for IMO {imo}. Error: {e}")

    # 3. Check for 'not found' from the API
    if api_data.get("found") is False:
        # If explicitly not found, return empty dict or static if it exists
        return result if result else {}
    
    # 4. Merge API data (dynamic and static specs) into the result
    if api_data:
        # Dynamic Fields (always overwrite/update)
        result["lat"] = float(api_data.get("lat")) if api_data.get("lat") is not None else result.get("lat")
        result["lon"] = float(api_data.get("lon")) if api_data.get("lon") is not None else result.get("lon")
        result["sog"] = float(api_data.get("sog") or 0.0)
        result["cog"] = float(api_data.get("cog") or 0.0)
        result["last_pos_utc"] = api_data.get("last_pos_utc")
        result["destination"] = (api_data.get("destination") or "").strip()
        
        # Static Fields (Prioritize API data if available, fall back to cache data in 'result')
        result["name"] = (api_data.get("vessel_name") or api_data.get("name") or result.get("name") or f"IMO {imo}").strip()
        result["ship_type"] = (api_data.get("ship_type") or result.get("ship_type") or "").strip()
        result["flag"] = (api_data.get("flag") or result.get("flag") or "").strip()
        result["deadweight_t"] = api_data.get("deadweight_t") if api_data.get("deadweight_t") is not None else result.get("deadweight_t")
        result["gross_tonnage"] = api_data.get("gross_tonnage") if api_data.get("gross_tonnage") is not None else result.get("gross_tonnage")
        result["year_of_build"] = api_data.get("year_of_build") if api_data.get("year_of_build") is not None else result.get("year_of_build")
        result["length_overall_m"] = api_data.get("length_overall_m") if api_data.get("length_overall_m") is not None else result.get("length_overall_m")
        result["beam_m"] = api_data.get("beam_m") if api_data.get("beam_m") is not None else result.get("beam_m")
        
    # 5. Final validation: Must have IMO key
    if not result: return {}
    result["imo"] = imo 
    
    return result

def humanize_eta(h):
    m = int(round(h * 60)); H = m // 60; M = m % 60
    if H < 24: return f"{H}h {M}m" if M else f"{H}h"
    d = H // 24; r = H % 24
    return f"{d}d {r}h" if r else f"{d}d"

def build_alert_and_state(v, ports, prev):
    # Unpack all fields, including the new static fields from the 'v' dictionary
    imo = v["imo"]; name = v.get("name"); lat = v.get("lat"); lon = v.get("lon"); sog = v.get("sog"); cog = v.get("cog"); last = v.get("last_pos_utc"); dest = v["destination"]
    ship_type = v.get("ship_type")
    flag = v.get("flag")
    dwt = v.get("deadweight_t")
    gt = v.get("gross_tonnage")
    year = v.get("year_of_build")
    length = v.get("length_overall_m")
    beam = v.get("beam_m")
    
    # If key position data is missing, we cannot calculate movement/ETA/alerts
    if lat is None or lon is None or sog is None:
        # We still return the state with static data for saving, but skip alerts/complex calculations
        new_state={
            "imo":imo,"name":name,"lat":None,"lon":None,"sog":None,"cog":None,"last_pos_utc":None,
            "destination":dest,"nearest_port":None,"nearest_distance_nm":None,
            "destination_port":None,"destination_distance_nm":None,
            "eta_hours":None,"eta_utc":None,"eta_text":None,"done":False,
            "ship_type": ship_type,"flag": flag,"deadweight_t": dwt,"gross_tonnage": gt,
            "year_of_build": year,"length_overall_m": length,"beam_m": beam,
        }
        return None, new_state


    age = age_minutes(last) if last else None
    age_txt = "N/A" if age is None else f"{age:.0f} min ago"
    too_old = age is not None and age > MAX_AIS_FOR_ETA_MIN
    nearest, nmd = nearest_port(lat, lon, ports)
    nearest_disp = nearest or "N/A"
    nmd_txt = "N/A" if nmd is None else f"{nmd:.1f} NM"
    
    dest_name, dest_nm = None, None
    if dest:
        dn, c = match_destination_port(dest, ports)
        if dn and c: dest_name = dn; dest_nm = haversine_nm(lat, lon, c["lat"], c["lon"])
        
    eta_h = eta_str = eta_text = None
    if dest_nm is not None and dest_nm > MIN_DISTANCE_FOR_ETA and sog >= MIN_SOG_FOR_ETA and not too_old:
        es = min(max(sog, MIN_SOG_FOR_ETA), MAX_ETA_SOG_CAP)
        try: raw = dest_nm / es
        except: raw = None
        if raw is not None and raw <= MAX_ETA_HOURS:
            eta_h = raw
            eta_dt = datetime.now(timezone.utc) + timedelta(hours=eta_h)
            eta_str = eta_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            eta_text = humanize_eta(eta_h)
            
    # 🚢 Include all static fields in the state dictionary
    new_state = {
        "imo": imo, "name": name, "lat": lat, "lon": lon, "sog": sog, "cog": cog, "last_pos_utc": last,
        "destination": dest, "nearest_port": nearest, "nearest_distance_nm": nmd,
        "destination_port": dest_name, "destination_distance_nm": dest_nm,
        "eta_hours": eta_h, "eta_utc": eta_str, "eta_text": eta_text, "done": False,
        
        # STATIC SPECIFICATIONS SAVED TO JSON
        "ship_type": ship_type,
        "flag": flag,
        "deadweight_t": dwt,
        "gross_tonnage": gt,
        "year_of_build": year,
        "length_overall_m": length,
        "beam_m": beam,
    }
    
    arrived = nmd is not None and nmd <= ARRIVAL_RADIUS_NM and sog <= 0.5
    if arrived: new_state["done"] = True
    
    # --- WhatsApp Alert Logic ---
    if not prev:
        msg = [
            f"🚢 {name} (IMO {imo})",
            "📌 Status: First tracking detected",
            f"🕒 AIS: {age_txt}",
            f"⚡ Speed: {sog:.1f} kn | 🧭 {cog:.0f}°",
            f"📍 Position: {lat:.4f} , {lon:.4f}",
            f"⚓ Nearest port: {nearest_disp} (~{nmd_txt})",
            f"🎯 Destination: {dest or 'N/A'}"
        ]
        if dest_name and dest_nm is not None: msg[-1] += f" (~{dest_nm:.1f} NM)"
        if eta_text: msg.append(f"⏱ ETA: {eta_text} ({eta_str})")
        return "\n".join(msg), new_state
    
    if prev.get("done"): return None, new_state
    old_dest = (prev.get("destination") or "").strip()
    changed = (old_dest.upper() != dest.upper()) if (old_dest or dest) else False
    prev_lat, prev_lon = prev.get("lat"), prev.get("lon")
    moved, diff = False, None
    if prev_lat is not None and prev_lon is not None:
        diff = haversine_nm(prev_lat, prev_lon, lat, lon)
        moved = diff >= MIN_MOVE_NM
        
    arrival_event = arrived and not prev.get("done")
    if not (changed or moved or arrival_event): return None, new_state
    
    if arrival_event: status = "Arrived at destination area"
    elif changed: status = "Destination changed"
    else: status = "Position / track updated"
    
    extra = f" (Δ {diff:.1f} NM)" if moved and diff is not None else ""
    if changed and old_dest: dest_line = f"🎯 Destination changed: {old_dest} ➜ {dest or 'N/A'}"
    else: dest_line = f"🎯 Destination: {dest or 'N/A'}"
    if dest_name and dest_nm is not None: dest_line += f" (~{dest_nm:.1f} NM)"
    
    msg = [
        f"🚢 {name} (IMO {imo})",
        f"📌 Status: {status}",
        f"🕒 AIS: {age_txt}",
        f"⚡ Speed: {sog:.1f} kn | 🧭 {cog:.0f}°{extra}",
        f"📍 Position: {lat:.4f} , {lon:.4f}",
        f"⚓ Nearest port: {nearest_disp} (~{nmd_txt})",
        dest_line
    ]
    if eta_text: msg.append(f"⏱ ETA: {eta_text} ({eta_str})")
    return "\n".join(msg), new_state

# --- MAIN EXECUTION ---

def main():
    # 1. Ping the Render API to wake it up (optional, but good practice)
    try: requests.get(f"{RENDER_BASE}/ping", timeout=30)
    except: pass
    
    # 2. Load the central Static Cache (read-only for this script)
    static_cache = load_json(STATIC_CACHE_PATH, {})

    # 3. Load tracked IMOs list
    imos = load_json(TRACKED_IMOS_PATH, [])
    if isinstance(imos, dict) and "tracked_imos" in imos: imos = imos["tracked_imos"]
    if not isinstance(imos, list): imos = []
    imos = [str(i).strip() for i in imos if str(i).strip()]
    if not imos: return
    
    # 4. Load ports and previous state
    ports = load_ports()
    prev = load_json(VESSELS_STATE_PATH, {})
    if not isinstance(prev, dict): prev = {}
    new_all = {}
    
    # 5. Process each tracked vessel
    for imo in imos:
        # Fetch vessel data, using static_cache as fallback
        v = fetch_from_render_api(imo, static_cache)
        
        # If fetch failed and we have no position data, use the previous state if available
        if not v or v.get("lat") is None or v.get("lon") is None:
            if imo in prev: new_all[imo] = prev[imo]
            continue
            
        # Build alert and new state
        alert, ns = build_alert_and_state(v, ports, prev.get(imo))
        new_all[imo] = ns
        
        # Send alert if necessary
        if alert: send_whatsapp_message(alert)
        
    # 6. Save the new state for all vessels
    if new_all: save_json(VESSELS_STATE_PATH, new_all)

if __name__ == "__main__":
    main()

