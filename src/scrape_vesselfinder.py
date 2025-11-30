import json
import math
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

TRACKED_IMOS_PATH = Path("data/tracked_imos.json")
VESSELS_STATE_PATH = Path("data/vessels_data.json")
PORTS_PATH = Path("data/ports.json")

# Your 4 ports (NM radius can be changed)
DEFAULT_PORTS = {
    "LAAYOUNE": {"lat": 27.1536, "lon": -13.2033},
    "TAN TAN": {"lat": 28.4927, "lon": -11.3437},
    "TARFAYA": {"lat": 27.9373, "lon": -12.9221},
    "DAKHLA": {"lat": 23.7048, "lon": -15.9336}
}

# CallMeBot WhatsApp variables (replace with env for security)
CALLMEBOT_PHONE = "212663401022"        # "2126xxxxxxxx"
CALLMEBOT_API_KEY = "9206809"      # Key from CallMeBot website

# AIS freshness threshold
FRESH_SIGNAL_MINUTES = 30
APPROACH_RADIUS_NM = 50


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


# ============================================================
# LOAD TRACKED IMOs (supports list format)
# ============================================================

def load_tracked_imos() -> list[str]:
    data = load_json(TRACKED_IMOS_PATH, [])

    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]

    if isinstance(data, dict) and "tracked_imos" in data:
        lst = data.get("tracked_imos", [])
        if isinstance(lst, list):
            return [str(x).strip() for x in lst if str(x).strip()]

    return []


# ============================================================
# HAVERSINE FOR NM DISTANCE
# ============================================================

def distance_nm(lat1, lon1, lat2, lon2):
    R = 3440.065  # nautical miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ============================================================
# WHATSAPP ALERT
# ============================================================

def send_whatsapp_message(msg: str):
    url = (
        f"https://api.callmebot.com/whatsapp.php"
        f"?phone={CALLMEBOT_PHONE}&text={requests.utils.quote(msg)}&apikey={CALLMEBOT_API_KEY}"
    )
    try:
        requests.get(url, timeout=10)
        print(f"[ALERT SENT] {msg}")
    except:
        print("[ERROR] WhatsApp alert failed.")


# ============================================================
# SCRAPER FOR VESSELFINDER
# ============================================================

def scrape_vesselfinder(imo: str) -> dict:
    url = f"https://www.vesselfinder.com/vessels/details/{imo}"
    print("Fetching:", url)

    r = requests.get(url, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    # Vessel name
    title = soup.find("h1")
    name = title.text.strip() if title else "UNKNOWN"

    # JSON in the HTML
    json_blob = soup.find("script", {"type": "application/ld+json"})
    if not json_blob:
        return {}

    try:
        data = json.loads(json_blob.text)
    except:
        return {}

    # Extract core fields
    lat = float(data.get("latitude")) if data.get("latitude") else None
    lon = float(data.get("longitude")) if data.get("longitude") else None
    sog = float(data.get("speed")) if data.get("speed") else 0
    cog = float(data.get("course")) if data.get("course") else 0

    last_pos_utc = data.get("dateModified")  # e.g. 2025-01-01T12:34:00Z
    arrival_destination = data.get("arrivalDestination", "")

    return {
        "imo": imo,
        "name": name,
        "lat": lat,
        "lon": lon,
        "sog": sog,
        "cog": cog,
        "last_pos_utc": last_pos_utc,
        "destination": arrival_destination
    }


# ============================================================
# ALERT LOGIC
# ============================================================

def detect_alerts(v: dict, ports: dict):
    alerts = []
    name = v.get("name", "UNKNOWN")
    lat, lon = v.get("lat"), v.get("lon")
    last_utc = v.get("last_pos_utc")

    # ---- 1. Signal freshness alert ----
    if last_utc:
        try:
            dt = datetime.fromisoformat(last_utc.replace("Z", "+00:00"))
            age_mins = (datetime.now(timezone.utc) - dt).total_seconds() / 60
        except:
            age_mins = None
    else:
        age_mins = None

    if age_mins is not None and age_mins <= FRESH_SIGNAL_MINUTES:
        alerts.append(f"🛰️ Fresh AIS: {name} updated {age_mins:.0f} min ago")

    # ---- 2. Port distance alerts ----
    if lat and lon:
        for port, coords in ports.items():
            dist = distance_nm(lat, lon, coords["lat"], coords["lon"])
            if dist <= APPROACH_RADIUS_NM:
                alerts.append(f"⚓ {name} is {dist:.1f} NM from {port}")

    return alerts


# ============================================================
# MAIN
# ============================================================

def main():
    # Load tracked IMOs
    imos = load_tracked_imos()
    print("Tracked IMOs:", imos)

    # Load ports list
    ports = load_json(PORTS_PATH, DEFAULT_PORTS)

    # Previous states
    old_data = load_json(VESSELS_STATE_PATH, {})

    new_data = {}

    for imo in imos:
        v = scrape_vesselfinder(imo)
        if not v:
            print(f"[WARN] Could not scrape IMO {imo}")
            continue

        new_data[imo] = v

        # ---- ALERT PROCESSING ----
        alerts = detect_alerts(v, ports)
        for a in alerts:
            print("[ALERT]", a)
            send_whatsapp_message(a)

    # Save updated vessels
    save_json(VESSELS_STATE_PATH, new_data)
    print("Saved vessels_data.json ✔")


# ============================================================

if __name__ == "__main__":
    main()
