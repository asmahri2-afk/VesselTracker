"""
equasis_enricher.py
───────────────────
Daily job: fetch all A N P vessels, enrich missing/incomplete
entries in Supabase static_vessel_cache via Oracle /equasis API.

Rules:
  - Skip IMOs already cached with equasis_owner set within CACHE_DAYS
  - If vessel exists but has gaps (no owner, no flag, etc.) → merge,
    never overwrite a field that already has a value
  - Rate limit: 3 requests / minute, no upper cap
"""

import os
import re
import sys
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ANP_URL      = "https://www.anp.org.ma/_vti_bin/WS/Service.svc/mvmnv/all"
RENDER_BASE  = os.getenv("RENDER_BASE", "").rstrip("/")
API_SECRET   = os.getenv("API_SECRET", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

BATCH_SIZE   = 3     # Equasis requests per minute
BATCH_DELAY  = 60    # seconds between batches
CACHE_DAYS   = 182    # days before a complete record is considered stale
REQUEST_GAP  = 2     # seconds between individual requests within a batch

# Fields we get from Equasis — used for merge logic
EQUASIS_FIELDS = [
    "name", "flag", "ship_type", "mmsi", "call_sign",
    "gross_tonnage", "deadweight_t", "year_of_build",
    "equasis_owner", "equasis_address", "pi_club", "class_society",
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", "").strip()) if val else None
    except Exception:
        return None

def _safe_int(val) -> Optional[int]:
    try:
        return int(str(val).strip()) if val else None
    except Exception:
        return None

def _is_valid_imo(imo: str) -> bool:
    imo = str(imo).strip()
    if not re.match(r'^\d{7}$', imo):
        return False
    try:
        total = sum(int(imo[i]) * (7 - i) for i in range(6))
        return int(imo[6]) == total % 10
    except Exception:
        return False

def log(level: str, msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# ANP — fetch all vessels
# ══════════════════════════════════════════════════════════════════════════════

def fetch_anp_vessels() -> list:
    """Fetch all vessels from ANP API regardless of port."""
    headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "fr-FR,fr;q=0.9",
        "Referer":         "https://www.anp.org.ma/",
        "Origin":          "https://www.anp.org.ma",
        "Cache-Control":   "no-cache",
    }
    for attempt in range(3):
        try:
            log("INFO", f"Fetching ANP data (attempt {attempt + 1}/3)...")
            r = requests.get(ANP_URL, headers=headers, timeout=(10, 60))
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise ValueError("ANP response is not a list")
            log("INFO", f"ANP returned {len(data)} vessel records")
            return data
        except Exception as e:
            log("WARNING", f"ANP fetch attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(5 * (2 ** attempt))
    raise RuntimeError("ANP fetch failed after 3 attempts")

def extract_imos(vessels: list) -> list:
    """Extract unique valid 7-digit IMOs from ANP vessel list."""
    seen, result = set(), []
    for v in vessels:
        raw = str(v.get("nUMERO_LLOYDField") or "").strip()
        if raw and raw not in seen:
            if _is_valid_imo(raw):
                seen.add(raw)
                result.append(raw)
            else:
                log("DEBUG", f"Skipping invalid IMO: '{raw}' ({v.get('nOM_NAVIREField', '?')})")
    log("INFO", f"Extracted {len(result)} unique valid IMOs from {len(vessels)} records")
    return result

# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE — read cache
# ══════════════════════════════════════════════════════════════════════════════

def supabase_get_cache(imos: list) -> dict:
    """
    Return dict {imo: row} for all IMOs currently in static_vessel_cache.
    Fetches in chunks of 200 to stay within URL limits.
    """
    if not SUPABASE_URL or not SUPABASE_KEY or not imos:
        return {}

    cache = {}
    chunk_size = 200
    fields = "imo,name,flag,ship_type,mmsi,call_sign,gross_tonnage,deadweight_t,year_of_build,equasis_owner,equasis_address,pi_club,class_society,equasis_updated"

    for i in range(0, len(imos), chunk_size):
        chunk = imos[i : i + chunk_size]
        imo_list = ",".join(chunk)
        url = (
            f"{SUPABASE_URL}/rest/v1/static_vessel_cache"
            f"?imo=in.({imo_list})"
            f"&select={fields}"
        )
        try:
            r = requests.get(url, headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            }, timeout=15)
            if r.ok:
                for row in r.json():
                    cache[row["imo"]] = row
            else:
                log("WARNING", f"Supabase cache fetch returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            log("WARNING", f"Supabase cache fetch error: {e}")

    log("INFO", f"Cache loaded: {len(cache)} existing records for {len(imos)} IMOs")
    return cache

def is_complete(row: dict) -> bool:
    """
    A record is 'complete' if it has equasis_owner AND
    was updated within CACHE_DAYS. If so, skip it.
    """
    if not row.get("equasis_owner"):
        return False
    updated = row.get("equasis_updated")
    if not updated:
        return False
    try:
        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - dt).days
        return age_days < CACHE_DAYS
    except Exception:
        return False

def needs_enrichment(row: dict) -> bool:
    """
    Return True if the row has any gap that Equasis can fill:
    missing owner, flag, ship_type, gross_tonnage, etc.
    """
    if is_complete(row):
        return False
    gaps = [f for f in EQUASIS_FIELDS if not row.get(f)]
    return len(gaps) > 0

# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE — write
# ══════════════════════════════════════════════════════════════════════════════

def build_upsert_row(imo: str, equasis_data: dict, existing_row: Optional[dict]) -> dict:
    """
    Build the row to upsert.
    Rule: never overwrite a field that already has a non-null value.
    Only fill gaps.
    """
    # Normalise Equasis response field names
    eq = {
        "name":            equasis_data.get("vessel_name") or equasis_data.get("name"),
        "flag":            equasis_data.get("flag")         or equasis_data.get("Flag"),
        "ship_type":       equasis_data.get("ship_type")    or equasis_data.get("Type of ship"),
        "mmsi":            equasis_data.get("mmsi")         or equasis_data.get("MMSI"),
        "call_sign":       equasis_data.get("call_sign")    or equasis_data.get("Call Sign"),
        "gross_tonnage":   _safe_float(equasis_data.get("gross_tonnage") or equasis_data.get("Gross tonnage")),
        "deadweight_t":    _safe_float(equasis_data.get("deadweight_t")  or equasis_data.get("DWT")),
        "year_of_build":   _safe_int(equasis_data.get("year_of_build")   or equasis_data.get("Year of build")),
        "equasis_owner":   equasis_data.get("equasis_owner"),
        "equasis_address": equasis_data.get("equasis_address"),
        "pi_club":         equasis_data.get("pi_club"),
        "class_society":   equasis_data.get("class_society"),
    }

    row = {"imo": str(imo), "equasis_updated": datetime.now(timezone.utc).isoformat()}

    for field, eq_value in eq.items():
        existing_value = (existing_row or {}).get(field)
        if existing_value:
            # Keep existing — never overwrite
            row[field] = existing_value
        elif eq_value is not None:
            # Fill the gap
            row[field] = eq_value
        # else: both null — omit to avoid overwriting with null

    return row

def supabase_upsert(row: dict):
    """Upsert a row into static_vessel_cache."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("WARNING", "Supabase credentials missing — skipping upsert")
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
            timeout=10,
        )
        if r.status_code in (200, 201, 204):
            return True
        log("WARNING", f"Supabase upsert {row['imo']} → {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        log("WARNING", f"Supabase upsert error for IMO {row['imo']}: {e}")
        return False


def supabase_get_owner_row(imo: str) -> Optional[dict]:
    """Fetch current vessel_owners row for this IMO."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/vessel_owners"
            f"?imo=eq.{imo}&select=imo,name,address,phone,email,equasis_source",
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


def upsert_vessel_owner(imo: str, equasis_data: dict):
    """
    Upsert into vessel_owners using Equasis data.

    Merge rules:
      - New record  → insert everything from Equasis
      - equasis_source = true (auto record)  → update all equasis fields freely
      - equasis_source = false (user record) → only fill null fields,
        NEVER touch name / address / phone / email
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return

    owner_name    = equasis_data.get("equasis_owner")
    owner_address = equasis_data.get("equasis_address")
    pi_club       = equasis_data.get("pi_club")

    # Nothing useful to write
    if not owner_name:
        return

    existing = supabase_get_owner_row(imo)
    now = datetime.now(timezone.utc).isoformat()

    if existing is None:
        # Brand new — insert full Equasis record
        row = {
            "imo":             str(imo),
            "name":            owner_name,
            "address":         owner_address,
            "equasis_address": owner_address,
            "pi_club":         pi_club,
            "equasis_source":  True,
            "updated_by":      "equasis",
            "updated_at":      now,
        }
        # Remove nulls
        row = {k: v for k, v in row.items() if v is not None}

    elif existing.get("equasis_source") is True:
        # Auto record — update equasis fields freely
        row = {
            "imo":             str(imo),
            "name":            owner_name,
            "address":         owner_address,
            "equasis_address": owner_address,
            "pi_club":         pi_club,
            "equasis_source":  True,
            "updated_by":      "equasis",
            "updated_at":      now,
        }
        row = {k: v for k, v in row.items() if v is not None}

    else:
        # User-owned record (equasis_source = false)
        # Only fill genuine nulls — never touch name/address/phone/email
        row = {"imo": str(imo), "updated_at": now}
        if not existing.get("equasis_address") and owner_address:
            row["equasis_address"] = owner_address
        if not existing.get("pi_club") and pi_club:
            row["pi_club"] = pi_club
        # name / address / phone / email → never touched
        if len(row) == 2:
            # Nothing to update
            return

    try:
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
        if r.status_code not in (200, 201, 204):
            log("WARNING", f"vessel_owners upsert {imo} → {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log("WARNING", f"vessel_owners upsert error for IMO {imo}: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# EQUASIS — fetch via Oracle API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_equasis(imo: str) -> Optional[dict]:
    """Call Oracle API /equasis/{imo}."""
    if not RENDER_BASE or not API_SECRET:
        log("ERROR", "RENDER_BASE or API_SECRET not configured")
        return None
    try:
        r = requests.get(
            f"{RENDER_BASE}/equasis/{imo}",
            headers={"X-API-Secret": API_SECRET},
            timeout=30,
        )
        if r.status_code == 404:
            log("INFO", f"IMO {imo}: not found on Equasis")
            return None
        if not r.ok:
            log("WARNING", f"IMO {imo}: HTTP {r.status_code}")
            return None
        data = r.json()
        if data.get("found") is False:
            log("INFO", f"IMO {imo}: Equasis returned found=false")
            return None
        return data
    except requests.exceptions.Timeout:
        log("WARNING", f"IMO {imo}: Equasis request timed out")
        return None
    except Exception as e:
        log("WARNING", f"IMO {imo}: Equasis fetch error — {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  EQUASIS ENRICHER — Daily Cache Builder")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Fetch all ANP vessels
    try:
        all_vessels = fetch_anp_vessels()
    except RuntimeError as e:
        log("CRITICAL", str(e))
        sys.exit(1)

    # 2. Extract valid IMOs
    imos = extract_imos(all_vessels)
    if not imos:
        log("INFO", "No valid IMOs found — nothing to do")
        return

    # 3. Load current Supabase cache for these IMOs
    cache = supabase_get_cache(imos)

    # 4. Classify each IMO
    to_fetch   = []   # not in cache at all
    to_enrich  = []   # in cache but incomplete
    skip_count = 0

    for imo in imos:
        row = cache.get(imo)
        if row is None:
            to_fetch.append((imo, None))
        elif needs_enrichment(row):
            gaps = [f for f in EQUASIS_FIELDS if not row.get(f)]
            log("DEBUG", f"IMO {imo} ({row.get('name','?')}): gaps → {gaps}")
            to_enrich.append((imo, row))
        else:
            skip_count += 1

    work = to_fetch + to_enrich
    log("INFO", f"Summary: {len(to_fetch)} new | {len(to_enrich)} need enrichment | {skip_count} complete — skipped")

    if not work:
        log("INFO", "Nothing to fetch — all IMOs are up to date")
        return

    log("INFO", f"Starting enrichment of {len(work)} IMOs ({BATCH_SIZE}/min, no cap)")
    print("-" * 60)

    done = 0
    failed = 0
    skipped_no_data = 0

    for batch_start in range(0, len(work), BATCH_SIZE):
        batch = work[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(work) + BATCH_SIZE - 1) // BATCH_SIZE
        log("INFO", f"Batch {batch_num}/{total_batches}: {[b[0] for b in batch]}")

        for imo, existing_row in batch:
            eq_data = fetch_equasis(imo)

            if eq_data is None:
                skipped_no_data += 1
                # Even if Equasis has nothing, mark it so we don't retry next run
                # unless it's a completely new record we couldn't find
                if existing_row is not None:
                    # Update timestamp to avoid re-querying too soon
                    supabase_upsert({
                        "imo": str(imo),
                        "equasis_updated": datetime.now(timezone.utc).isoformat(),
                    })
            else:
                row = build_upsert_row(imo, eq_data, existing_row)
                if supabase_upsert(row):
                    upsert_vessel_owner(imo, eq_data)
                    done += 1
                    action = "inserted" if existing_row is None else "enriched"
                    owner = row.get("equasis_owner", "—")
                    log("OK", f"IMO {imo} ({row.get('name','?')}) {action} | owner: {owner}")
                else:
                    failed += 1

            # Small gap between requests within a batch
            time.sleep(REQUEST_GAP)

        # Wait between batches (skip after last)
        if batch_start + BATCH_SIZE < len(work):
            log("INFO", f"Waiting {BATCH_DELAY}s before next batch...")
            time.sleep(BATCH_DELAY)

    print("=" * 60)
    log("DONE", f"Enriched: {done} | Not found on Equasis: {skipped_no_data} | Errors: {failed}")
    print("=" * 60)

if __name__ == "__main__":
    main()
