"""
magicport_owner_enricher.py
───────────────────────────
Scrape vessel ownership/management data from magicport.ai
/vessels/load/management/{IMO} and enrich:
  - static_vessel_cache (equasis_owner, equasis_address)
  - vessel_owners (name, address, equasis_source, equasis_address)

Uses curl_cffi to bypass Cloudflare + extracts Laravel CSRF token from homepage.

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
REQUEST_DELAY = 1.5   # seconds between management page requests
BATCH_SIZE = 200      # Supabase fetch chunk size

# Only process IMOs missing owner? Set to False to re-scrape all.
ONLY_MISSING = True

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log(level: str, msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


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
        # Pattern 1: <meta name="csrf-token" content="...">
        m = re.search(r'''<meta[^>]+name=["']csrf-token["'][^>]+content=["']([^"']+)["']''', html, re.IGNORECASE)
        if not m:
            m = re.search(r'''<meta[^>]+content=["']([^"']+)["'][^>]+name=["']csrf-token["']''', html, re.IGNORECASE)
        if not m:
            m = re.search(r'''"csrfToken"\s*:\s*"([^"]+)"''', html)
        if m:
            token = m.group(1)
            log("INFO", f"CSRF token obtained: {token[:20]}...")
            return token
        log("WARNING", "CSRF token not found in homepage HTML")
    except Exception as e:
        log("WARNING", f"Failed to fetch homepage for CSRF: {e}")
    return None


def fetch_management_page(imo: str, session, csrf_token: str) -> Optional[str]:
    """GET /vessels/load/management/{imo} and return HTML."""
    url = f"https://magicport.ai/vessels/load/management/{imo}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": f"https://magicport.ai/vessels/{imo}",
        "X-CSRF-Token": csrf_token,
        "X-Requested-With": "XMLHttpRequest",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    for attempt in range(3):
        try:
            r = session.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            log("WARNING", f"IMO {imo} fetch error (attempt {attempt + 1}/3): {e}")
        if attempt < 2:
            time.sleep(5 * (attempt + 1))
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

    # Find all accordion items (works for both full page and partial HTML)
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

        # Extract name from link or plain text
        name = None
        p = content.find("p", class_=lambda x: x and "text-style" in x)
        if p:
            a = p.find("a", class_="text--primary")
            if a:
                name = a.get_text(strip=True)
            else:
                name = p.get_text(strip=True)

        # Extract address (first list item with map icon, not locked)
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
# SUPABASE — read / write
# ══════════════════════════════════════════════════════════════════════════════

def get_imos() -> List[str]:
    """Fetch IMOs from static_vessel_cache. If ONLY_MISSING, only those without equasis_owner."""
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
        log("WARNING", f"Failed to fetch IMOs from Supabase: {e}")
    return imos


def supabase_get_owner_row(imo: str) -> Optional[dict]:
    """Fetch current vessel_owners row for this IMO."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        import requests
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/vessel_owners?imo=eq.{imo}&select=imo,name,address,phone,email,equasis_source,equasis_address,pi_club",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            },
            timeout=10,
        )
        if r.ok:
            rows = r.json()
            return rows[0] if rows else None
    except Exception as e:
        log("WARNING", f"vessel_owners fetch error for IMO {imo}: {e}")
    return None


def upsert_static_cache(imo: str, owner_name: str, owner_address: Optional[str]) -> bool:
    """Update equasis_owner / equasis_address in static_vessel_cache."""
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
    """Upsert into vessel_owners with same merge logic as Equasis enricher."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False

    existing = supabase_get_owner_row(imo)
    now = datetime.now(timezone.utc).isoformat()

    if existing is None:
        # Brand new — insert full MagicPort record
        row = {
            "imo":             str(imo),
            "name":            owner_name,
            "address":         owner_address,
            "equasis_address": owner_address,
            "equasis_source":  True,
            "updated_by":      "magicport",
            "updated_at":      now,
        }
        row = {k: v for k, v in row.items() if v is not None}

    elif existing.get("equasis_source") is True:
        # Auto record — update freely
        row = {
            "imo":             str(imo),
            "name":            owner_name,
            "address":         owner_address,
            "equasis_address": owner_address,
            "equasis_source":  True,
            "updated_by":      "magicport",
            "updated_at":      now,
        }
        row = {k: v for k, v in row.items() if v is not None}

    else:
        # User-owned record — only fill genuine nulls
        row = {"imo": str(imo), "updated_at": now}
        if not existing.get("equasis_address") and owner_address:
            row["equasis_address"] = owner_address
        if not existing.get("address") and owner_address:
            row["address"] = owner_address
        if not existing.get("name") and owner_name:
            row["name"] = owner_name
        if len(row) == 2:
            return True  # Nothing to update

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
        if r.status_code in (200, 201, 204):
            return True
        log("WARNING", f"vessel_owners upsert {imo} → {r.status_code}: {r.text[:200]}")
        return False
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
        log("INFO", "Using curl_cffi + BeautifulSoup")
    except ImportError as e:
        log("CRITICAL", f"Missing dependency: {e}. Install: pip install curl_cffi beautifulsoup4")
        sys.exit(1)

    # 1. Get IMOs to process
    imos = get_imos()
    if not imos:
        log("INFO", "No IMOs to process — exiting")
        return

    # 2. Create session and fetch CSRF token
    session = curl_requests.Session(impersonate="chrome120")
    log("INFO", "Fetching magicport.ai homepage for session cookies...")
    csrf_token = get_csrf_token(session)
    if not csrf_token:
        log("CRITICAL", "Failed to obtain CSRF token")
        sys.exit(1)

    total_enriched = 0
    total_failed = 0
    total_no_data = 0

    for idx, imo in enumerate(imos, 1):
        log("INFO", f"[{idx}/{len(imos)}] IMO {imo}")

        html = fetch_management_page(imo, session, csrf_token)
        if html is None:
            total_failed += 1
            time.sleep(REQUEST_DELAY)
            continue

        parsed = parse_management_html(html)
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
    log("DONE", f"Enriched: {total_enriched} | No data: {total_no_data} | Errors: {total_failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
