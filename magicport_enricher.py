"""
magicport_enricher.py
─────────────────────
Scrape vessel-at-port data from magicport.ai for all ports listed
in ports.txt and upsert into Supabase static_vessel_cache.

Env:
    SUPABASE_URL
    SUPABASE_SERVICE_KEY  (or SUPABASE_KEY)

Usage:
    python magicport_enricher.py
"""

import os
import re
import sys
import time
import requests
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PORTS_FILE   = "/mnt/agents/upload/ports.txt"   # fallback path
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_KEY", "")
REQUEST_DELAY = 1.5   # seconds between port requests
VESSEL_DELAY  = 0.2   # seconds between individual vessel upserts (optional)

# ══════════════════════════════════════════════════════════════════════════════
# ISO 3166-1 alpha-2 → full country name
# ══════════════════════════════════════════════════════════════════════════════

FLAG_MAP = {
    "AF": "Afghanistan", "AL": "Albania", "DZ": "Algeria", "AS": "American Samoa",
    "AD": "Andorra", "AO": "Angola", "AI": "Anguilla", "AQ": "Antarctica",
    "AG": "Antigua and Barbuda", "AR": "Argentina", "AM": "Armenia", "AW": "Aruba",
    "AU": "Australia", "AT": "Austria", "AZ": "Azerbaijan", "BS": "Bahamas",
    "BH": "Bahrain", "BD": "Bangladesh", "BB": "Barbados", "BY": "Belarus",
    "BE": "Belgium", "BZ": "Belize", "BJ": "Benin", "BM": "Bermuda",
    "BT": "Bhutan", "BO": "Bolivia", "BQ": "Bonaire, Sint Eustatius and Saba",
    "BA": "Bosnia and Herzegovina", "BW": "Botswana", "BV": "Bouvet Island",
    "BR": "Brazil", "IO": "British Indian Ocean Territory", "BN": "Brunei Darussalam",
    "BG": "Bulgaria", "BF": "Burkina Faso", "BI": "Burundi", "CV": "Cabo Verde",
    "KH": "Cambodia", "CM": "Cameroon", "CA": "Canada", "KY": "Cayman Islands",
    "CF": "Central African Republic", "TD": "Chad", "CL": "Chile", "CN": "China",
    "CX": "Christmas Island", "CC": "Cocos (Keeling) Islands", "CO": "Colombia",
    "KM": "Comoros", "CG": "Congo", "CD": "Congo, Democratic Republic of the",
    "CK": "Cook Islands", "CR": "Costa Rica", "HR": "Croatia", "CU": "Cuba",
    "CW": "Curacao", "CY": "Cyprus", "CZ": "Czechia", "CI": "Cote d'Ivoire",
    "DK": "Denmark", "DJ": "Djibouti", "DM": "Dominica", "DO": "Dominican Republic",
    "EC": "Ecuador", "EG": "Egypt", "SV": "El Salvador", "GQ": "Equatorial Guinea",
    "ER": "Eritrea", "EE": "Estonia", "SZ": "Eswatini", "ET": "Ethiopia",
    "FK": "Falkland Islands (Malvinas)", "FO": "Faroe Islands", "FJ": "Fiji",
    "FI": "Finland", "FR": "France", "GF": "French Guiana", "PF": "French Polynesia",
    "TF": "French Southern Territories", "GA": "Gabon", "GM": "Gambia", "GE": "Georgia",
    "DE": "Germany", "GH": "Ghana", "GI": "Gibraltar", "GR": "Greece", "GL": "Greenland",
    "GD": "Grenada", "GP": "Guadeloupe", "GU": "Guam", "GT": "Guatemala", "GG": "Guernsey",
    "GN": "Guinea", "GW": "Guinea-Bissau", "GY": "Guyana", "HT": "Haiti",
    "HM": "Heard Island and McDonald Islands", "VA": "Holy See", "HN": "Honduras",
    "HK": "Hong Kong", "HU": "Hungary", "IS": "Iceland", "IN": "India",
    "ID": "Indonesia", "IR": "Iran", "IQ": "Iraq", "IE": "Ireland", "IM": "Isle of Man",
    "IL": "Israel", "IT": "Italy", "JM": "Jamaica", "JP": "Japan", "JE": "Jersey",
    "JO": "Jordan", "KZ": "Kazakhstan", "KE": "Kenya", "KI": "Kiribati",
    "KP": "Korea (Democratic People's Republic of)", "KR": "Korea, Republic of",
    "KW": "Kuwait", "KG": "Kyrgyzstan", "LA": "Lao People's Democratic Republic",
    "LV": "Latvia", "LB": "Lebanon", "LS": "Lesotho", "LR": "Liberia", "LY": "Libya",
    "LI": "Liechtenstein", "LT": "Lithuania", "LU": "Luxembourg", "MO": "Macao",
    "MG": "Madagascar", "MW": "Malawi", "MY": "Malaysia", "MV": "Maldives", "ML": "Mali",
    "MT": "Malta", "MH": "Marshall Islands", "MQ": "Martinique", "MR": "Mauritania",
    "MU": "Mauritius", "YT": "Mayotte", "MX": "Mexico", "FM": "Micronesia (Federated States of)",
    "MD": "Moldova, Republic of", "MC": "Monaco", "MN": "Mongolia", "ME": "Montenegro",
    "MS": "Montserrat", "MA": "Morocco", "MZ": "Mozambique", "MM": "Myanmar",
    "NA": "Namibia", "NR": "Nauru", "NP": "Nepal", "NL": "Netherlands", "NC": "New Caledonia",
    "NZ": "New Zealand", "NI": "Nicaragua", "NE": "Niger", "NG": "Nigeria", "NU": "Niue",
    "NF": "Norfolk Island", "MK": "North Macedonia", "MP": "Northern Mariana Islands",
    "NO": "Norway", "OM": "Oman", "PK": "Pakistan", "PW": "Palau", "PS": "Palestine, State of",
    "PA": "Panama", "PG": "Papua New Guinea", "PY": "Paraguay", "PE": "Peru",
    "PH": "Philippines", "PN": "Pitcairn", "PL": "Poland", "PT": "Portugal",
    "PR": "Puerto Rico", "QA": "Qatar", "RE": "Reunion", "RO": "Romania",
    "RU": "Russian Federation", "RW": "Rwanda", "BL": "Saint Barthelemy",
    "SH": "Saint Helena, Ascension and Tristan da Cunha", "KN": "Saint Kitts and Nevis",
    "LC": "Saint Lucia", "MF": "Saint Martin (French part)", "PM": "Saint Pierre and Miquelon",
    "VC": "Saint Vincent and the Grenadines", "WS": "Samoa", "SM": "San Marino",
    "ST": "Sao Tome and Principe", "SA": "Saudi Arabia", "SN": "Senegal", "RS": "Serbia",
    "SC": "Seychelles", "SL": "Sierra Leone", "SG": "Singapore", "SX": "Sint Maarten (Dutch part)",
    "SK": "Slovakia", "SI": "Slovenia", "SB": "Solomon Islands", "SO": "Somalia",
    "ZA": "South Africa", "GS": "South Georgia and the South Sandwich Islands",
    "SS": "South Sudan", "ES": "Spain", "LK": "Sri Lanka", "SD": "Sudan", "SR": "Suriname",
    "SJ": "Svalbard and Jan Mayen", "SE": "Sweden", "CH": "Switzerland",
    "SY": "Syrian Arab Republic", "TW": "Taiwan, Province of China", "TJ": "Tajikistan",
    "TZ": "Tanzania, United Republic of", "TH": "Thailand", "TL": "Timor-Leste", "TG": "Togo",
    "TK": "Tokelau", "TO": "Tonga", "TT": "Trinidad and Tobago", "TN": "Tunisia",
    "TR": "Turkey", "TM": "Turkmenistan", "TC": "Turks and Caicos Islands", "TV": "Tuvalu",
    "UG": "Uganda", "UA": "Ukraine", "AE": "United Arab Emirates", "GB": "United Kingdom",
    "US": "United States of America", "UM": "United States Minor Outlying Islands",
    "UY": "Uruguay", "UZ": "Uzbekistan", "VU": "Vanuatu", "VE": "Venezuela",
    "VN": "Viet Nam", "VG": "Virgin Islands (British)", "VI": "Virgin Islands (U.S.)",
    "WF": "Wallis and Futuna", "EH": "Western Sahara", "YE": "Yemen", "ZM": "Zambia",
    "ZW": "Zimbabwe",
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log(level: str, msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def resolve_flag(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = str(code).strip().upper()
    return FLAG_MAP.get(code, code)


def extract_mmsi(route: Optional[str]) -> Optional[str]:
    if not route:
        return None
    m = re.search(r"mmsi-(\d+)", str(route))
    return m.group(1) if m else None


def fetch_port_vessels(port_url: str) -> List[Dict[str, Any]]:
    """POST to /vessel-at-port/ and return list of vessel dicts."""
    url = port_url.rstrip("/") + "/vessel-at-port/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Referer": "https://magicport.ai/",
    }
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, timeout=30, json={})
            r.raise_for_status()
            data = r.json()

            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ("data", "vessels", "results", "items", "ships", "vessel"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # Single vessel object
                if "imo" in data:
                    return [data]
            return []
        except requests.exceptions.Timeout:
            log("WARNING", f"{url} timeout (attempt {attempt + 1}/3)")
        except Exception as e:
            log("WARNING", f"{url} error (attempt {attempt + 1}/3): {e}")
        if attempt < 2:
            time.sleep(5 * (attempt + 1))
    return []


def upsert_vessel(row: dict) -> bool:
    """Upsert a single row into static_vessel_cache."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("ERROR", "Supabase credentials missing — skipping upsert")
        return False
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/static_vessel_cache",
            json=row,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            timeout=15,
        )
        if r.status_code in (200, 201, 204):
            return True
        log("WARNING", f"Upsert IMO {row.get('imo')} → HTTP {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        log("WARNING", f"Upsert error IMO {row.get('imo')}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  MAGICPORT ENRICHER — Vessel-at-Port Cache Builder")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    if not SUPABASE_URL or not SUPABASE_KEY:
        log("CRITICAL", "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        sys.exit(1)

    # Read ports file
    if not os.path.exists(PORTS_FILE):
        # Try relative path
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ports.txt")
        if os.path.exists(alt):
            ports_path = alt
        else:
            log("CRITICAL", f"Ports file not found: {PORTS_FILE}")
            sys.exit(1)
    else:
        ports_path = PORTS_FILE

    with open(ports_path, "r", encoding="utf-8") as f:
        ports = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    log("INFO", f"Loaded {len(ports)} ports from {ports_path}")

    total_inserted = 0
    total_failed = 0
    total_skipped = 0

    for idx, port_url in enumerate(ports, 1):
        log("INFO", f"[{idx}/{len(ports)}] Fetching {port_url} ...")
        vessels = fetch_port_vessels(port_url)

        if not vessels:
            log("INFO", "  → 0 vessels")
            time.sleep(REQUEST_DELAY)
            continue

        log("INFO", f"  → {len(vessels)} vessel(s)")

        for v in vessels:
            imo_raw = v.get("imo")
            if not imo_raw:
                total_skipped += 1
                continue

            imo = str(imo_raw).strip()
            # Validate IMO: must be 7 digits (magicport sometimes returns 7-digit IMOs)
            if not re.match(r"^\d{7}$", imo):
                log("DEBUG", f"  Skipping invalid IMO: {imo}")
                total_skipped += 1
                continue

            # Build upsert row — only fields we have from magicport
            row: Dict[str, Any] = {"imo": imo}

            name = v.get("name")
            if name and str(name).strip():
                row["name"] = str(name).strip()

            ship_type = v.get("type")
            if ship_type and str(ship_type).strip() and str(ship_type).strip() != "-":
                row["ship_type"] = str(ship_type).strip()

            flag_code = v.get("flag")
            if flag_code and str(flag_code).strip() and str(flag_code).strip() != "-":
                resolved = resolve_flag(str(flag_code))
                if resolved:
                    row["flag"] = resolved

            dwt = v.get("dwt")
            if dwt is not None and str(dwt).strip() and str(dwt).strip() != "-":
                row["deadweight_t"] = str(dwt).strip()

            length = v.get("length")
            if length is not None and str(length).strip() and str(length).strip() != "-":
                row["length_overall_m"] = str(length).strip()

            mmsi = extract_mmsi(v.get("route"))
            if mmsi:
                row["mmsi"] = mmsi

            # Cache timestamp
            row["cached_at"] = datetime.now(timezone.utc).isoformat()

            if upsert_vessel(row):
                total_inserted += 1
            else:
                total_failed += 1

            if VESSEL_DELAY > 0:
                time.sleep(VESSEL_DELAY)

        time.sleep(REQUEST_DELAY)

    print("=" * 60)
    log("DONE", f"Inserted/Updated: {total_inserted} | Failed: {total_failed} | Skipped: {total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()
