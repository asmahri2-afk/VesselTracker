"""
magicport_owner_enricher.py  — HARDENED VERSION
────────────────────────────────────────────────
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

FIXES vs previous version:
  • Paginated DB reads — handles 17k+ rows (was capped at 1000)
  • Preloaded vessel_owners dict — eliminates N+1 GET per vessel
  • Persistent Supabase session — connection reuse across all DB calls
  • Retry on Supabase upserts — up to 3 attempts on transient errors
  • Iterative retries (not recursive) — no RecursionError risk
  • get_already_enriched_imos() only called when ONLY_MISSING=False
  • Fixed referer — removed fragile regex, always uses constructed URL
  • Cloudflare block detection with 120s pause
  • Split counters: not_found / no_data / no_access / cf_block / db_error
"""

import os
import re
import sys
import time
import random
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Set

import requests  # top-level import; session reuse via get_supa_session()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_KEY", "")

# MagicPort is aggressive on rate limiting — 6-10s minimum between requests
REQUEST_DELAY_MIN = float(os.getenv("REQUEST_DELAY_MIN", "6.0"))
REQUEST_DELAY_MAX = float(os.getenv("REQUEST_DELAY_MAX", "10.0"))

ONLY_MISSING = True   # False = reprocess all (slow; only for full rebuilds)

# 429 back-off: doubles each retry — 30s → 60s → 120s
BACKOFF_BASE   = float(os.getenv("BACKOFF_BASE",   "30.0"))
MAX_RETRIES    = int(os.getenv("MAX_RETRIES",      "3"))
DB_MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES",   "3"))    # retries for Supabase writes
DB_PAGE_SIZE   = 1000    # Supabase PostgREST page size (fixed maximum)

# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT SUPABASE SESSION — shared across all DB calls
# ══════════════════════════════════════════════════════════════════════════════

_supa_session: Optional[requests.Session] = None


def get_supa_session() -> requests.Session:
    """Lazy singleton — one TCP connection pool for all Supabase requests."""
    global _supa_session
    if _supa_session is None:
        _supa_session = requests.Session()
        _supa_session.headers.update({
            "apikey":        SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        })
    return _supa_session


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def jitter_delay() -> float:
    return random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)


def is_cloudflare_block(html: str) -> bool:
    """Detect CF JS-challenge / block pages that arrive as HTTP 200."""
    lower = html.lower()
    return (
        "cf-ray"                in lower
        or "checking your browser" in lower
        or "enable javascript"     in lower
        or "just a moment"         in lower
    )


def extract_csrf_from_html(html: str) -> Optional[str]:
    """Extract Laravel CSRF token from HTML meta tag (three patterns)."""
    for pattern in [
        r'<meta[^>]+name=["\']csrf-token["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']csrf-token["\']',
        r'"csrfToken"\s*:\s*"([^"]+)"',
    ]:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAGICPORT HTTP — iterative retries (no recursion risk)
# ══════════════════════════════════════════════════════════════════════════════

_MP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def fetch_vessel_page(imo: str, session) -> Optional[str]:
    """
    GET vessel page to establish session context + extract CSRF.
    Iterative retry with exponential back-off on 429.
    Returns HTML string or None on non-200 / exhausted retries.
    """
    url = f"https://magicport.ai/vessels/{imo}"
    headers = {
        "User-Agent":      _MP_UA,
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer":         "https://magicport.ai/",
        "Sec-Fetch-Dest":  "document",
        "Sec-Fetch-Mode":  "navigate",
        "Sec-Fetch-Site":  "same-origin",
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=headers, timeout=30, allow_redirects=True)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:
                wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 10)
                log("WARN", f"IMO {imo} vessel page → 429, back-off {wait:.0f}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                    continue
            # 404, 410, 5xx, etc.
            log("DEBUG", f"IMO {imo} vessel page → HTTP {r.status_code}")
            return None
        except Exception as e:
            log("DEBUG", f"IMO {imo} vessel page error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * (attempt + 1))
    return None


def fetch_management(imo: str, session, csrf_token: str) -> Optional[str]:
    """
    POST to /vessels/load/management/{imo}.
    Referer is always the vessel page URL (no fragile link-extraction regex).
    Iterative retry with exponential back-off on 429.
    """
    url     = f"https://magicport.ai/vessels/load/management/{imo}"
    referer = f"https://magicport.ai/vessels/{imo}"   # always correct; removed broken regex
    headers = {
        "User-Agent":        _MP_UA,
        "Accept":            "*/*",
        "Accept-Language":   "en-US,en;q=0.9",
        "Accept-Encoding":   "gzip, deflate, br",
        "Content-Type":      "application/json",
        "Origin":            "https://magicport.ai",
        "Referer":           referer,
        "X-Requested-With":  "XMLHttpRequest",
        "X-CSRF-Token":      csrf_token,
        "Sec-Fetch-Dest":    "empty",
        "Sec-Fetch-Mode":    "cors",
        "Sec-Fetch-Site":    "same-origin",
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.post(url, headers=headers, json={}, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:
                wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 10)
                log("WARN", f"IMO {imo} management POST → 429, back-off {wait:.0f}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                    continue
            log("DEBUG", f"IMO {imo} management POST → HTTP {r.status_code}")
            return None
        except Exception as e:
            log("DEBUG", f"IMO {imo} management POST error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * (attempt + 1))
    return None


# ══════════════════════════════════════════════════════════════════════════════
# HTML PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_management_html(html: str) -> Dict[str, Any]:
    """Parse management accordion HTML → dict of {label: {name, address}}."""
    from bs4 import BeautifulSoup   # already verified at startup

    soup     = BeautifulSoup(html, "html.parser")
    sections: Dict[str, Any] = {}

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

        name: Optional[str] = None
        p = content.find("p", class_=lambda x: x and "text-style" in x)
        if p:
            a = p.find("a", class_="text--primary")
            name = (a or p).get_text(strip=True)

        address: Optional[str] = None
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
# SUPABASE — paginated reads
# ══════════════════════════════════════════════════════════════════════════════

def _paginate(url_base: str) -> List[dict]:
    """
    Fetch ALL rows from a Supabase REST endpoint using Range-header pagination.
    Handles tables with 17k+ rows (default limit is 1000 rows per request).
    Breaks early on partial page (last page) or error.
    """
    supa  = get_supa_session()
    rows: List[dict] = []
    offset = 0

    while True:
        try:
            r = supa.get(
                url_base,
                headers={"Range": f"{offset}-{offset + DB_PAGE_SIZE - 1}"},
                timeout=30,
            )
            if not r.ok:
                log("WARNING", f"_paginate HTTP {r.status_code} at offset {offset} — stopping")
                break
            page = r.json()
            if not page:
                break
            rows.extend(page)
            if len(page) < DB_PAGE_SIZE:
                break          # last page
            offset += DB_PAGE_SIZE
        except Exception as e:
            log("WARNING", f"_paginate error at offset {offset}: {e}")
            break

    return rows


def get_imos() -> List[str]:
    """
    Fetch all target IMOs from static_vessel_cache.
    Fully paginated — safe for 17k+ rows.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("CRITICAL", "Supabase credentials missing")
        return []

    url = f"{SUPABASE_URL}/rest/v1/static_vessel_cache?select=imo"
    if ONLY_MISSING:
        url += "&equasis_owner=is.null"

    rows = _paginate(url)
    imos = [str(row["imo"]) for row in rows if row.get("imo")]
    log("INFO", f"Fetched {len(imos)} IMOs from Supabase"
                + (" (missing owner only)" if ONLY_MISSING else " (all)"))
    return imos


def get_already_enriched_imos() -> Set[str]:
    """
    Return IMOs that already have equasis_owner set.
    Only called when ONLY_MISSING=False — avoids a redundant paginated read
    when ONLY_MISSING=True (get_imos already excludes them via the DB filter).
    Fully paginated — safe for 17k+ rows.
    """
    url = f"{SUPABASE_URL}/rest/v1/static_vessel_cache?select=imo&equasis_owner=not.is.null"
    rows = _paginate(url)
    enriched = {str(row["imo"]) for row in rows if row.get("imo")}
    log("INFO", f"Found {len(enriched)} already-enriched IMOs (will skip)")
    return enriched


def preload_vessel_owners() -> Dict[str, dict]:
    """
    Load ALL vessel_owners rows into memory once at startup.
    Eliminates the per-vessel GET in upsert_vessel_owner (N+1 → 1 call).
    Fully paginated — safe for 17k+ rows.
    """
    url  = (
        f"{SUPABASE_URL}/rest/v1/vessel_owners"
        "?select=imo,name,address,equasis_source,equasis_address,pi_club"
    )
    rows = _paginate(url)
    cache = {str(row["imo"]): row for row in rows if row.get("imo")}
    log("INFO", f"Preloaded {len(cache)} vessel_owners rows into memory")
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE — writes with retry
# ══════════════════════════════════════════════════════════════════════════════

def _supabase_upsert(table: str, row: dict, imo: str) -> bool:
    """
    POST to Supabase with merge-duplicates.
    Retries up to DB_MAX_RETRIES times on transient errors.
    """
    supa = get_supa_session()
    url  = f"{SUPABASE_URL}/rest/v1/{table}"
    hdrs = {
        "Content-Type": "application/json",
        "Prefer":       "resolution=merge-duplicates,return=minimal",
    }
    for attempt in range(DB_MAX_RETRIES):
        try:
            r = supa.post(url, json=row, headers=hdrs, timeout=15)
            if r.status_code in (200, 201, 204):
                return True
            log("WARNING", f"[{table}] IMO {imo} upsert HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:
            log("WARNING", f"[{table}] IMO {imo} upsert error "
                           f"(attempt {attempt + 1}/{DB_MAX_RETRIES}): {e}")
        if attempt < DB_MAX_RETRIES - 1:
            time.sleep(2 ** attempt)   # 1s, 2s before next attempt
    return False


def upsert_static_cache(imo: str, owner_name: str, owner_address: Optional[str]) -> bool:
    row: Dict[str, Any] = {"imo": imo, "equasis_owner": owner_name}
    if owner_address:
        row["equasis_address"] = owner_address
    return _supabase_upsert("static_vessel_cache", row, imo)


def upsert_vessel_owner(
    imo: str,
    owner_name: str,
    owner_address: Optional[str],
    owners_cache: Dict[str, dict],
) -> bool:
    """
    Upsert into vessel_owners using the pre-loaded in-memory cache
    (avoids a GET per vessel).  Updates the cache in-place after writing.
    Returns True on success or genuine no-op; False on DB write failure.
    """
    existing = owners_cache.get(str(imo))
    now      = datetime.now(timezone.utc).isoformat()

    if existing is None or existing.get("equasis_source") is True:
        # New row or existing equasis-sourced row — overwrite fully
        row: Dict[str, Any] = {
            "imo":            str(imo),
            "name":           owner_name,
            "address":        owner_address,
            "equasis_address": owner_address,
            "equasis_source": True,
            "updated_by":     "magicport",
            "updated_at":     now,
        }
        row = {k: v for k, v in row.items() if v is not None}
    else:
        # Non-equasis row — only fill empty fields, never overwrite manual data
        updates: Dict[str, Any] = {"imo": str(imo), "updated_at": now}
        if not existing.get("equasis_address") and owner_address:
            updates["equasis_address"] = owner_address
        if not existing.get("address") and owner_address:
            updates["address"] = owner_address
        if not existing.get("name") and owner_name:
            updates["name"] = owner_name
        if len(updates) == 2:   # only imo + updated_at — nothing to change
            return True          # genuine no-op; static_cache still written by caller
        row = updates

    ok = _supabase_upsert("vessel_owners", row, imo)
    if ok:
        # Keep in-memory cache consistent so subsequent IMOs see updated state
        if existing is None:
            owners_cache[str(imo)] = row
        else:
            existing.update(row)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  MAGICPORT OWNER ENRICHER — Management Data Scraper")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    if not SUPABASE_URL or not SUPABASE_KEY:
        log("CRITICAL", "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        sys.exit(1)

    # Verify heavy dependencies before any work starts
    try:
        from curl_cffi import requests as curl_requests
        from bs4 import BeautifulSoup   # noqa: F401
    except ImportError as e:
        log("CRITICAL", f"Missing dependency: {e}. "
                        "Install: pip install curl_cffi beautifulsoup4")
        sys.exit(1)

    # ── 1. Load target IMOs ─────────────────────────────────────────────────
    imos = get_imos()
    if not imos:
        log("INFO", "No IMOs to process — exiting")
        return

    # ── 2. Filter already-enriched (only needed when ONLY_MISSING=False) ───
    #       When ONLY_MISSING=True, get_imos() already excluded them via DB.
    if not ONLY_MISSING:
        already_enriched = get_already_enriched_imos()
        before = len(imos)
        imos   = [imo for imo in imos if imo not in already_enriched]
        skipped = before - len(imos)
        if skipped:
            log("INFO", f"Skipping {skipped} already-enriched IMOs")

    if not imos:
        log("INFO", "All IMOs already enriched — exiting")
        return

    log("INFO", f"Processing {len(imos)} IMOs")

    # ── 3. Preload vessel_owners once (eliminates N+1 GET per vessel) ───────
    owners_cache = preload_vessel_owners()

    # ── 4. Create MagicPort scraping session ────────────────────────────────
    session = curl_requests.Session(impersonate="chrome120")

    # ── 5. Per-vessel loop ───────────────────────────────────────────────────
    total_enriched  = 0
    total_db_error  = 0
    total_not_found = 0   # vessel page missing on MagicPort (404/410)
    total_no_data   = 0   # page found but no ownership sections parsed
    total_cf_block  = 0   # Cloudflare challenge detected
    total_no_access = 0   # management endpoint denied (non-200 after retries)

    for idx, imo in enumerate(imos, 1):
        log("INFO", f"[{idx}/{len(imos)}] IMO {imo}")

        # ── Step 1: vessel page → session context + CSRF ────────────────────
        vessel_html = fetch_vessel_page(imo, session)
        if vessel_html is None:
            log("INFO", "  → Vessel page not reachable")
            total_not_found += 1
            time.sleep(jitter_delay())
            continue

        csrf = extract_csrf_from_html(vessel_html)
        if not csrf:
            if is_cloudflare_block(vessel_html):
                log("WARNING", "  → Cloudflare block detected — pausing 120s before continuing")
                total_cf_block += 1
                time.sleep(120)
            else:
                log("INFO", "  → CSRF token not found in vessel page")
                total_no_data += 1
                time.sleep(jitter_delay())
            continue

        # ── Step 2: POST management endpoint ────────────────────────────────
        mgmt_html = fetch_management(imo, session, csrf)
        if mgmt_html is None:
            log("INFO", "  → Management endpoint denied access")
            total_no_access += 1
            time.sleep(jitter_delay())
            continue

        # ── Step 3: Parse ownership HTML ────────────────────────────────────
        parsed = parse_management_html(mgmt_html)
        if not parsed:
            log("INFO", "  → No management sections found in response")
            total_no_data += 1
            time.sleep(jitter_delay())
            continue

        owner_name, owner_address = extract_owner(parsed)
        if not owner_name:
            log("INFO", "  → No owner name extracted")
            total_no_data += 1
            time.sleep(jitter_delay())
            continue

        log("INFO",
            f"  → Owner: {owner_name[:50]} | "
            f"Addr: {owner_address[:50] if owner_address else '—'}")

        # ── Step 4: Write to both tables ────────────────────────────────────
        ok1 = upsert_static_cache(imo, owner_name, owner_address)
        ok2 = upsert_vessel_owner(imo, owner_name, owner_address, owners_cache)

        if ok1 and ok2:
            total_enriched += 1
        else:
            log("WARNING", f"  → DB write failed (static_cache={ok1}, vessel_owners={ok2})")
            total_db_error += 1

        time.sleep(jitter_delay())

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 60)
    log("DONE",
        f"Enriched: {total_enriched} | "
        f"No data: {total_no_data} | "
        f"No access: {total_no_access} | "
        f"Not found: {total_not_found} | "
        f"CF blocks: {total_cf_block} | "
        f"DB errors: {total_db_error}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
