"""
magicport_enricher.py  — OPTIMIZED VERSION
─────────────────────
Scrape vessel-at-port data from magicport.ai and upsert into Supabase.

OPTIMIZATIONS:
  • Batch upserts (100 rows per request) instead of 1-by-1
  • Persistent Supabase session with keep-alive
  • Zero delay between vessel DB writes
  • Pre-compiled regex
  • Connection reuse for both MagicPort and Supabase

Env:
    SUPABASE_URL
    SUPABASE_SERVICE_KEY

Usage:
    python magicport_enricher.py
"""

import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Set

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PORTS_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ports.txt")
SUPABASE_URL  = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY  = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_KEY", "")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "1.0"))   # seconds between ports
SKIP_EXISTING = True
BATCH_SIZE    = 100   # Supabase bulk upsert batch size

# Pre-compiled regex
IMO_RE = re.compile(r"^\d{7}$")
MMSI_RE = re.compile(r"mmsi-(\d+)")

# ══════════════════════════════════════════════════════════════════════════════
# FLAG MAP (truncated for brevity in display — full map preserved in output)
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
    m = MMSI_RE.search(str(route))
    return m.group(1) if m else None


def get_csrf_token(session) -> Optional[str]:
    """Fetch magicport.ai homepage to obtain session cookies + CSRF token."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://magicport.ai/",
    }
    try:
        r = session.get("https://magicport.ai/", headers=headers, timeout=30)
        r.raise_for_status()
        html = r.text

        m = re.search(r'<meta[^>]+name=["\']csrf-token["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if not m:
            m = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']csrf-token["\']', html, re.IGNORECASE)
        if not m:
            m = re.search(r'"csrfToken"\s*:\s*"([^"]+)"', html)
        if m:
            token = m.group(1)
            log("INFO", f"CSRF token obtained: {token[:20]}...")
            return token
        log("WARNING", "CSRF token not found in homepage HTML")
    except Exception as e:
        log("WARNING", f"Failed to fetch homepage for CSRF: {e}")
    return None


def fetch_port_vessels(port_url: str, session, csrf_token: str) -> List[Dict[str, Any]]:
    """POST /vessel-at-port using shared session + CSRF token."""
    url = port_url.rstrip("/") + "/vessel-at-port"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Content-Type": "application/json",
        "Origin": "https://magicport.ai",
        "Referer": port_url.rstrip("/") + "/",
        "X-Requested-With": "XMLHttpRequest",
        "X-CSRF-Token": csrf_token,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    for attempt in range(3):
        try:
            r = session.post(url, headers=headers, json={}, timeout=30)
            r.raise_for_status()
            data = r.json()

            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ("data", "vessels", "results", "items", "ships", "vessel"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
                if "imo" in data:
                    return [data]
            return []
        except Exception as e:
            log("WARNING", f"{url} error (attempt {attempt + 1}/3): {e}")
        if attempt < 2:
            time.sleep(5 * (attempt + 1))
    return []


def get_existing_imos(supa_session) -> Set[str]:
    """Fetch all existing IMOs from static_vessel_cache (paginated)."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("WARNING", "Cannot fetch existing IMOs — credentials missing")
        return set()

    existing: Set[str] = set()
    offset = 0
    limit = 1000

    while True:
        try:
            r = supa_session.get(
                f"{SUPABASE_URL}/rest/v1/static_vessel_cache?select=imo",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Range": f"{offset}-{offset + limit - 1}",
                },
                timeout=30,
            )
            if not r.ok:
                log("WARNING", f"Fetch existing IMOs failed: HTTP {r.status_code}")
                break
            rows = r.json()
            if not rows:
                break
            for row in rows:
                if row.get("imo"):
                    existing.add(str(row["imo"]))
            if len(rows) < limit:
                break
            offset += limit
        except Exception as e:
            log("WARNING", f"Error fetching existing IMOs: {e}")
            break

    log("INFO", f"Found {len(existing)} existing IMOs in cache (will skip)")
    return existing


def flush_batch(supa_session, batch: List[dict]) -> tuple:
    """Bulk upsert a batch of rows. Returns (success_count, fail_count)."""
    if not batch:
        return 0, 0
    try:
        r = supa_session.post(
            f"{SUPABASE_URL}/rest/v1/static_vessel_cache",
            json=batch,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            timeout=30,
        )
        if r.status_code in (200, 201, 204):
            return len(batch), 0
        log("WARNING", f"Batch upsert failed HTTP {r.status_code}: {r.text[:200]}")
        # Fallback: individual inserts
        ok = 0
        fail = 0
        for row in batch:
            try:
                rr = supa_session.post(
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
                if rr.status_code in (200, 201, 204):
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
        return ok, fail
    except Exception as e:
        log("WARNING", f"Batch upsert error: {e}")
        return 0, len(batch)


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

    try:
        from curl_cffi import requests as curl_requests
        import requests
        log("INFO", "Using curl_cffi (Chrome impersonation)")
    except ImportError as e:
        log("CRITICAL", f"Missing dependency: {e}. Install: pip install curl_cffi requests")
        sys.exit(1)

    # Read ports file
    if not os.path.exists(PORTS_FILE):
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

    # Persistent Supabase session (keep-alive, connection reuse)
    supa_session = requests.Session()
    supa_session.headers.update({
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })

    # Load existing IMOs once
    existing_imos: Set[str] = set()
    if SKIP_EXISTING:
        existing_imos = get_existing_imos(supa_session)

    # Create curl_cffi session for MagicPort
    session = curl_requests.Session(impersonate="chrome120")

    # Obtain CSRF token
    log("INFO", "Fetching magicport.ai homepage to obtain session cookies...")
    csrf_token = get_csrf_token(session)
    if not csrf_token:
        log("CRITICAL", "Failed to obtain CSRF token. Cloudflare may be blocking.")
        sys.exit(1)

    total_inserted = 0
    total_failed = 0
    total_skipped = 0
    total_existing = 0
    batch: List[dict] = []

    for idx, port_url in enumerate(ports, 1):
        log("INFO", f"[{idx}/{len(ports)}] {port_url}")
        vessels = fetch_port_vessels(port_url, session, csrf_token)

        if not vessels:
            log("INFO", "  → 0 vessels")
            time.sleep(REQUEST_DELAY)
            continue

        new_in_port = 0
        existing_in_port = 0

        for v in vessels:
            imo_raw = v.get("imo")
            if not imo_raw:
                total_skipped += 1
                continue

            imo = str(imo_raw).strip()
            if not IMO_RE.match(imo):
                total_skipped += 1
                continue

            # Skip if already in DB
            if SKIP_EXISTING and imo in existing_imos:
                existing_in_port += 1
                total_existing += 1
                continue

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

            row["cached_at"] = datetime.now(timezone.utc).isoformat()

            batch.append(row)
            new_in_port += 1
            existing_imos.add(imo)

            # Flush batch when full
            if len(batch) >= BATCH_SIZE:
                ok, fail = flush_batch(supa_session, batch)
                total_inserted += ok
                total_failed += fail
                batch = []

        # Report per port
        if existing_in_port > 0:
            log("INFO", f"  → {new_in_port} new | {existing_in_port} already in DB")
        elif new_in_port > 0:
            log("INFO", f"  → {new_in_port} new vessel(s)")
        else:
            log("INFO", "  → 0 vessels")

        time.sleep(REQUEST_DELAY)

    # Flush remaining batch
    if batch:
        ok, fail = flush_batch(supa_session, batch)
        total_inserted += ok
        total_failed += fail

    print("=" * 60)
    log("DONE", f"Inserted: {total_inserted} | Existing skipped: {total_existing} | Failed: {total_failed} | Invalid skipped: {total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()
