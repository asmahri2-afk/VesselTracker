import json
import math
import os
import re
import requests
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path

TRACKED_IMOS_PATH = Path("data/tracked_imos.json")
VESSELS_STATE_PATH = Path("data/vessels_data.json")
PORTS_PATH = Path("data/ports.json")

CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE")
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY")
CALLMEBOT_ENABLED = bool(CALLMEBOT_PHONE and CALLMEBOT_APIKEY)
CALLMEBOT_API_URL = "https://api.callmebot.com/whatsapp.php"

print(f"[DEBUG] CALLMEBOT_PHONE: {'SET' if CALLMEBOT_PHONE else 'MISSING'}")
print(f"[DEBUG] CALLMEBOT_APIKEY: {'SET' if CALLMEBOT_APIKEY else 'MISSING'}")
print(f"[DEBUG] CALLMEBOT_ENABLED: {CALLMEBOT_ENABLED}")
print("[DEBUG] ETA VERSION ACTIVE")

RENDER_BASE = "https://vessel-api-s85s.onrender.com"

MAX_AIS_MINUTES = 30
ARRIVAL_RADIUS_NM = 35.5
MIN_MOVE_NM = 5.0

MIN_SOG_FOR_ETA = 0.5
MAX_ETA_HOURS = 240
MAX_ETA_SOG_CAP = 18.0
MAX_AIS_FOR_ETA_MIN = 360
MIN_DISTANCE_FOR_ETA = 5.0

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    R_km = 6371.0
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat/2)**2 + math.cos(lat1_r)*math.cos(lat2_r)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return (R_km * c) * 0.539957

def load_ports() -> dict:
    ports = load_json(PORTS_PATH, {})
    if not ports:
        raise RuntimeError("ports.json missing")
    return {k.upper(): v for k, v in ports.items()}

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.upper()
    return re.sub(r"[^A-Z]", "", s)

ALIASES_RAW = {
    "laayoune": "LAAYOUNE", "layoune": "LAAYOUNE", "eh eun": "LAAYOUNE", "leyoune": "LAAYOUNE",
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

DEST_ALIASES = {_normalize_text(k): v for k, v in ALIASES_RAW.items()}

def match_destination_port(destination: str, ports: dict):
    if not destination:
        return None, None
    norm = _normalize_text(destination)
    if norm in DEST_ALIASES:
        c = DEST_ALIASES[norm]
        return c, ports.get(c)
    canonical_map = {_normalize_text(p): p for p in ports.keys()}
    if norm in canonical_map:
        p = canonical_map[norm]
        return p, ports.get(p)
    for canon, name in canonical_map.items():
        if canon and canon in norm:
            return name, ports.get(name)
    return None, None

def nearest_port(lat: float, lon: float, ports: dict):
    best = None
    best_nm = None
    for name, c in ports.items():
        d = haversine_nm(lat, lon, c["lat"], c["lon"])
        if best_nm is None or d < best_nm:
            best_nm = d
            best = name
    return best, best_nm

def parse_ais_time(s: str) -> datetime | None:
    if not s:
        return None
    s = s.replace(" UTC", "").strip()
    try:
        dt = datetime.strptime(s, "%b %d, %Y %H:%M")
        return dt.replace(tzinfo=timezone.utc)
    except:
        return None

def age_minutes(t: str):
    dt = parse_ais_time(t)
    if not dt:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds()/60

def send_whatsapp_message(text: str):
    if not CALLMEBOT_ENABLED:
        return
    try:
        requests.get(CALLMEBOT_API_URL, params={
            "phone": CALLMEBOT_PHONE,
            "apikey": CALLMEBOT_APIKEY,
            "text": text
        }, timeout=20).raise_for_status()
    except:
        pass

def fetch_from_render_api(imo: str) -> dict:
    try:
        r = requests.get(f"{RENDER_BASE}/vessel-full/{imo}", timeout=60)
        r.raise_for_status()
        d = r.json()
    except:
        return {}
    if d.get("found") is False:
        return {}
    if d.get("lat") is None or d.get("lon") is None:
        return {}
    return {
        "imo": imo,
        "name": (d.get("name") or f"IMO {imo}").strip(),
        "lat": float(d.get("lat")),
        "lon": float(d.get("lon")),
        "sog": float(d.get("sog") or 0.0),
        "cog": float(d.get("cog") or 0.0),
        "last_pos_utc": d.get("last_pos_utc"),
        "destination": (d.get("destination") or "").strip()
    }

def humanize_eta(h):
    m = int(round(h*60))
    H = m//60
    M = m%60
    if H < 24:
        return f"{H}h {M}m" if M else f"{H}h"
    d = H//24
    r = H%24
    return f"{d}d {r}h" if r else f"{d}d"

def build_alert_and_state(v, ports, prev):
    imo = v["imo"]
    name = v.get("name")
    lat = v["lat"]
    lon = v["lon"]
    sog = v["sog"]
    cog = v["cog"]
    last = v.get("last_pos_utc")
    dest = v["destination"]

    age = age_minutes(last) if last else None
    age_txt = "N/A" if age is None else f"{age:.0f} min ago"
    too_old = age is not None and age > MAX_AIS_FOR_ETA_MIN

    nearest, nmd = nearest_port(lat, lon, ports)
    nearest_disp = nearest or "N/A"
    nmd_txt = "N/A" if nmd is None else f"{nmd:.1f} NM"

    dest_name = None
    dest_nm = None
    if dest:
        dn, c = match_destination_port(dest, ports)
        if dn and c:
            dest_name = dn
            dest_nm = haversine_nm(lat, lon, c["lat"], c["lon"])

    eta_h = None
    eta_str = None
    eta_text = None

    if dest_nm is not None and dest_nm > MIN_DISTANCE_FOR_ETA and sog >= MIN_SOG_FOR_ETA and not too_old:
        es = min(max(sog, MIN_SOG_FOR_ETA), MAX_ETA_SOG_CAP)
        try:
            raw = dest_nm / es
        except:
            raw = None
        if raw is not None and raw <= MAX_ETA_HOURS:
            eta_h = raw
            eta_dt = datetime.now(timezone.utc) + timedelta(hours=eta_h)
            eta_str = eta_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            eta_text = humanize_eta(eta_h)

    new_state = {
        "imo": imo, "name": name, "lat": lat, "lon": lon,
        "sog": sog, "cog": cog, "last_pos_utc": last,
        "destination": dest, "nearest_port": nearest,
        "nearest_distance_nm": nmd, "destination_port": dest_name,
        "destination_distance_nm": dest_nm,
        "eta_hours": eta_h, "eta_utc": eta_str,
        "eta_text": eta_text, "done": False
    }

    arrived = nmd is not None and nmd <= ARRIVAL_RADIUS_NM and sog <= 0.5
    if arrived:
        new_state["done"] = True

    if not prev:
        msg = [
            f"🚢 {name} (IMO {imo})",
            f"📌 Status: First tracking detected",
            f"🕒 AIS: {age_txt}",
            f"⚡ Speed: {sog:.1f} kn | 🧭 {cog:.0f}°",
            f"📍 {lat:.4f} , {lon:.4f}",
            f"⚓ {nearest_disp} (~{nmd_txt})",
            f"🎯 {dest or 'N/A'}"
        ]
        if dest_name and dest_nm is not None:
            msg[-1] += f" (~{dest_nm:.1f} NM)"
        if eta_text:
            msg.append(f"⏱ ETA: {eta_text} ({eta_str})")
        return "\n".join(msg), new_state

    if prev.get("done"):
        return None, new_state

    old_dest = (prev.get("destination") or "").strip()
    changed = (old_dest.upper() != dest.upper()) if (old_dest or dest) else False

    prev_lat = prev.get("lat")
    prev_lon = prev.get("lon")
    moved = False
    diff = None
    if prev_lat is not None and prev_lon is not None:
        diff = haversine_nm(prev_lat, prev_lon, lat, lon)
        moved = diff >= MIN_MOVE_NM

    arrival_event = arrived and not prev.get("done")
    if not (changed or moved or arrival_event):
        return None, new_state

    if arrival_event:
        status = "Arrived at destination area"
    elif changed:
        status = "Destination changed"
    else:
        status = "Position / track updated"

    extra = f" (Δ {diff:.1f} NM)" if moved and diff is not None else ""
    if changed and old_dest:
        dest_line = f"🎯 {old_dest} ➜ {dest or 'N/A'}"
    else:
        dest_line = f"🎯 {dest or 'N/A'}"

    if dest_name and dest_nm is not None:
        dest_line += f" (~{dest_nm:.1f} NM)"

    msg = [
        f"🚢 {name} (IMO {imo})",
        f"📌 {status}",
        f"🕒 AIS: {age_txt}",
        f"⚡ {sog:.1f} kn | 🧭 {cog:.0f}°{extra}",
        f"📍 {lat:.4f} , {lon:.4f}",
        f"⚓ {nearest_disp} (~{nmd_txt})",
        dest_line
    ]
    if eta_text:
        msg.append(f"⏱ ETA: {eta_text} ({eta_str})")

    return "\n".join(msg), new_state

def main():
    try:
        requests.get(f"{RENDER_BASE}/ping", timeout=30)
    except:
        pass

    imos = load_json(TRACKED_IMOS_PATH, [])
    if isinstance(imos, dict) and "tracked_imos" in imos:
        imos = imos["tracked_imos"]
    if not isinstance(imos, list):
        imos = []
    imos = [str(i).strip() for i in imos if str(i).strip()]
    if not imos:
        return

    ports = load_ports()
    prev = load_json(VESSELS_STATE_PATH, {})
    if not isinstance(prev, dict):
        prev = {}

    new_all = {}

    for imo in imos:
        v = fetch_from_render_api(imo)
        if not v:
            if imo in prev:
                new_all[imo] = prev[imo]
            continue
        alert, new_state = build_alert_and_state(v, ports, prev.get(imo))
        new_all[imo] = new_state
        if alert:
            send_whatsapp_message(alert)

    if new_all:
        save_json(VESSELS_STATE_PATH, new_all)

if __name__ == "__main__":
    main()
