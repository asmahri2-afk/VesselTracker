"""
magicport_owner_enricher.py
───────────────────────────
Scrape vessel ownership/management data from magicport.ai
/vessels/load/management/{IMO} and enrich:
  - static_vessel_cache (equasis_owner, equasis_address)
  - vessel_owners (name, address, equasis_source, equasis_address)

Flow: visit vessel page → extract CSRF → POST management endpoint.

Env:
    SUPABASE_URL
    SUPABASE_SERVICE_KEY

Usage:
    python magicport_owner_enricher.py
"""

import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_KEY", "")
REQUEST_DELAY = 2.0   # seconds between vessels
ONLY_MISSING = True

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log(level: str, msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def extract_csrf_from_html(html: str) -> Optional[str]:
    """Extract Laravel CSRF token from HTML meta tag."""
    m = re.search(r'''<meta[^>]+name=["']csrf-token["'][^>]+content=["']([^"']+)["']''', html, re.IGNORECASE)
    if not m:
        m = re.search(r'''<meta[^>]+content=["']([^"']+)["'][^>]+name=["']csrf-token["']''', html, re.IGNORECASE)
    if not m:
        m = re.search(r'''"csrfToken"\s*:\s*"([^"]+)"''', html)
    return m.group(1) if m else None


def fetch_vessel_page(imo: str, session) -> Optional[str]:
    """GET vessel page to establish session context + extract CSRF."""
    url = f"https://magicport.ai/vessels/{imo}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://magicport.ai/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
    }
    try:
        r = session.get(url, headers=headers, timeout=30, allow_redirects=True)
        if r.status_code == 200:
            return r.text
        log("DEBUG", f"IMO {imo} vessel page → {r.status_code}")
    except Exception as e:
        log("DEBUG", f"IMO {imo} vessel page error: {e}")
    return None


def fetch_management(imo: str, session, referer: str, csrf_token: str) -> Optional[str]:
    """POST to management endpoint with proper context."""
    url = f"https://magicport.ai/vessels/load/management/{imo}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Content-Type": "application/json",
        "Origin": "https://magicport.ai",
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest",
        "X-CSRF-Token": csrf_token,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    try:
        r = session.post(url, headers=headers, json={}, timeout=30)
        if r.status_code == 200:
            return r.text
        log("DEBUG", f"IMO {imo} management POST → {r.status_code}")
    except Exception as e:
        log("DEBUG", f"IMO {imo} management POST error: {e}")
    return None


def parse_management_html(html: str) -> Dict[str, Any]:
    """Parse HTML and extract owner/manager info."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log("ERROR", "beautifulsoup4 is required. Install: pip install beautifulsoup4")
        return {}

    soup = BeautifulSoup(html, "html.parser")
    sections = {}

    for item in soup.find_all("div", class_="accordion__item"):
        toggle = item.find("h3", class_="accordion__item-toggle")
        if not toggle:
            continue
        label_span = toggle.find("span", class_="accordion__item-toggle-label")
        if not label_span:
            continue
        label = label_span.get_text(strip=True)

        content = item.find("div", class_="accordion__item-content")
        if not content:
            continue

        name = None
        p = content.find("p", class_=lambda x: x and "text-style" in x)
        if p:
            a = p.find("a", class_="text--primary")
            if a:
                name = a.get_text(strip=True)
            else:
                name = p.get_text(strip=True)

        address = None
        ul = content.find("ul", class_="list--icon")
        if ul:
            for li in ul.find_all("li", class_="list__item"):
                icon_use = li.find("use")
                if icon_use and "icon-map" in icon_use.get("xlink:href", ""):
                    addr_span = li.find("span", class_="list__item-label")
                    if addr_span:
                        text = addr_span.get_text(strip=True)
                        if text and not text.startswith("•") and len(text) > 5:
                            address = text
                            break

        if name and len(name) > 2:
            sections[label] = {"name": name, "address": address}

    return sections


def extract_owner(parsed: dict) -> tuple:
    """Priority: Registered Owner > Commercial Manager > ISM Manager."""
    for key in ["Registered Owner", "Commercial Manager", "ISM Manager"]:
        if key in parsed and parsed[key].get("name"):
            return parsed[key]["name"], parsed[key].get("address")
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE
# ══════════════════════════════════════════════════════════════════════════════

def get_imos() -> List[str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("CRITICAL", "Supabase credentials missing")
        return []
    imos = []
    url = f"{SUPABASE_URL}/rest/v1/static_vessel_cache?select=imo"
    if ONLY_MISSING:
        url += "&equasis_owner=is.null"
    try:
        import requests
        r = requests.get(url, headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }, timeout=30)
        r.raise_for_status()
        for row in r.json():
            imos.append(row["imo"])
        log("INFO", f"Fetched {len(imos)} IMOs from Supabase" + (" (missing owner only)" if ONLY_MISSING else ""))
    except Exception as e:
        log("WARNING", f"Failed to fetch IMOs: {e}")
    return imos


def supabase_get_owner_row(imo: str) -> Optional[dict]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        import requests
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/vessel_owners?imo=eq.{imo}&select=imo,name,address,phone,email,equasis_source,equasis_address,pi_club",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=10,
        )
        if r.ok:
            rows = r.json()
            return rows[0] if rows else None
    except Exception as e:
        log("WARNING", f"vessel_owners fetch error IMO {imo}: {e}")
    return None


def upsert_static_cache(imo: str, owner_name: str, owner_address: Optional[str]) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    row = {"imo": imo, "equasis_owner": owner_name}
    if owner_address:
        row["equasis_address"] = owner_address
    try:
        import requests
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/static_vessel_cache",
            json=row,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            timeout=10,
        )
        return r.status_code in (200, 201, 204)
    except Exception as e:
        log("WARNING", f"static_cache upsert error IMO {imo}: {e}")
        return False


def upsert_vessel_owner(imo: str, owner_name: str, owner_address: Optional[str]) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    existing = supabase_get_owner_row(imo)
    now = datetime.now(timezone.utc).isoformat()

    if existing is None:
        row = {
            "imo": str(imo), "name": owner_name, "address": owner_address,
            "equasis_address": owner_address, "equasis_source": True,
            "updated_by": "magicport", "updated_at": now,
        }
        row = {k: v for k, v in row.items() if v is not None}
    elif existing.get("equasis_source") is True:
        row = {
            "imo": str(imo), "name": owner_name, "address": owner_address,
            "equasis_address": owner_address, "equasis_source": True,
            "updated_by": "magicport", "updated_at": now,
        }
        row = {k: v for k, v in row.items() if v is not None}
    else:
        row = {"imo": str(imo), "updated_at": now}
        if not existing.get("equasis_address") and owner_address:
            row["equasis_address"] = owner_address
        if not existing.get("address") and owner_address:
            row["address"] = owner_address
        if not existing.get("name") and owner_name:
            row["name"] = owner_name
        if len(row) == 2:
            return True

    try:
        import requests
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/vessel_owners",
            json=row,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            timeout=10,
        )
        return r.status_code in (200, 201, 204)
    except Exception as e:
        log("WARNING", f"vessel_owners upsert error IMO {imo}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  MAGICPORT OWNER ENRICHER — Management Data Scraper")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    if not SUPABASE_URL or not SUPABASE_KEY:
        log("CRITICAL", "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        sys.exit(1)

    try:
        from curl_cffi import requests as curl_requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        log("CRITICAL", f"Missing dependency: {e}. Install: pip install curl_cffi beautifulsoup4")
        sys.exit(1)

    imos = get_imos()
    if not imos:
        log("INFO", "No IMOs to process — exiting")
        return

    session = curl_requests.Session(impersonate="chrome120")

    total_enriched = 0
    total_failed = 0
    total_no_data = 0
    total_410 = 0

    for idx, imo in enumerate(imos, 1):
        log("INFO", f"[{idx}/{len(imos)}] IMO {imo}")

        # Step 1: Visit vessel page to get context + CSRF
        vessel_html = fetch_vessel_page(imo, session)
        if vessel_html is None:
            log("INFO", "  → Vessel page not found")
            total_failed += 1
            time.sleep(REQUEST_DELAY)
            continue

        csrf = extract_csrf_from_html(vessel_html)
        if not csrf:
            log("INFO", "  → CSRF not found on vessel page")
            total_no_data += 1
            time.sleep(REQUEST_DELAY)
            continue

        # Determine referer from any link in the page, or construct it
        referer = f"https://magicport.ai/vessels/{imo}"
        m = re.search(r'''href=["'](https://magicport\.ai/vessels/[^"']+)["']''', vessel_html)
        if m:
            referer = m.group(1)

        # Step 2: POST to management endpoint
        mgmt_html = fetch_management(imo, session, referer, csrf)
        if mgmt_html is None:
            log("INFO", "  → Management endpoint returned 410 (no access)")
            total_410 += 1
            time.sleep(REQUEST_DELAY)
            continue

        parsed = parse_management_html(mgmt_html)
        if not parsed:
            log("INFO", "  → No management sections found")
            total_no_data += 1
            time.sleep(REQUEST_DELAY)
            continue

        owner_name, owner_address = extract_owner(parsed)
        if not owner_name:
            log("INFO", "  → No owner name extracted")
            total_no_data += 1
            time.sleep(REQUEST_DELAY)
            continue

        log("INFO", f"  → Owner: {owner_name[:50]} | Address: {owner_address[:50] if owner_address else '—'}")

        ok1 = upsert_static_cache(imo, owner_name, owner_address)
        ok2 = upsert_vessel_owner(imo, owner_name, owner_address)

        if ok1 and ok2:
            total_enriched += 1
        else:
            total_failed += 1

        time.sleep(REQUEST_DELAY)

    print("=" * 60)
    log("DONE", f"Enriched: {total_enriched} | No data: {total_no_data} | 410: {total_410} | Errors: {total_failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
