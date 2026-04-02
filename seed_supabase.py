"""
seed_supabase.py  ·  VesselTracker — full data import
──────────────────────────────────────────────────────
Imports every JSON data file into Supabase.
Triggered by: .github/workflows/seed_once.yml  (manual dispatch)

Tables seeded
  1. ports               ← data/ports.json
  2. tracked_imos        ← data/tracked_imos.json
  3. vessels             ← data/vessels_data.json
  4. static_vessel_cache ← data/static_vessel_cache.json
  5. shipid_map          ← data/shipid_map.json        (optional)
  6. failure_counts      ← data/failure_counts.json    (optional)
  7. sanctioned_imos     ← data/sanctioned_imos.json

Run locally:
  SUPABASE_URL=https://... SUPABASE_SERVICE_KEY=... python seed_supabase.py
"""

import json
import os
import sys
from pathlib import Path
from supabase import create_client

# ── Env ────────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL") or sys.exit("❌  SUPABASE_URL not set")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or sys.exit("❌  SUPABASE_SERVICE_KEY not set")

sb    = create_client(SUPABASE_URL, SUPABASE_KEY)
CHUNK = 400   # safe batch size (PostgREST default max_rows = 1000)

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_json(path: str):
    p = Path(path)
    if not p.exists():
        print(f"  ⚠️  File not found — skipping: {path}")
        return None
    return json.loads(p.read_text())

def upsert(table: str, rows: list, label: str = "rows"):
    if not rows:
        print(f"  ⚠️  No rows to insert into {table}.")
        return 0
    ok = 0
    for i in range(0, len(rows), CHUNK):
        batch = rows[i : i + CHUNK]
        try:
            sb.table(table).upsert(batch).execute()
            ok += len(batch)
            print(f"  {min(i + CHUNK, len(rows))} / {len(rows)} {label} …")
        except Exception as e:
            print(f"  ❌  Batch {i}–{i+len(batch)} failed: {e}")
            # Print the first offending row so you can debug schema mismatches
            print(f"      Sample row: {json.dumps(batch[0], default=str)[:300]}")
    return ok

def add_ports_depth_columns():
    """
    The base schema only has (name, lat, lon).
    Ports JSON also has anchorage_depth and cargo_pier_depth which the
    frontend uses for port compatibility checks — add them if missing.
    Safe to run multiple times (IF NOT EXISTS).
    """
    sql = """
        ALTER TABLE ports
            ADD COLUMN IF NOT EXISTS anchorage_depth    DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS cargo_pier_depth   DOUBLE PRECISION;
    """
    try:
        sb.rpc("exec_sql", {"query": sql}).execute()
        print("  ✅  Depth columns ensured on ports table.")
    except Exception:
        # exec_sql RPC may not be enabled — try the direct Postgres RPC approach
        try:
            # Fallback: use supabase postgrest schema cache endpoint isn't available;
            # tell the user to run it manually.
            print("  ⚠️  Could not auto-add depth columns via RPC.")
            print("      Run this manually in SQL Editor → New Query:")
            print()
            print("      ALTER TABLE ports")
            print("          ADD COLUMN IF NOT EXISTS anchorage_depth  DOUBLE PRECISION,")
            print("          ADD COLUMN IF NOT EXISTS cargo_pier_depth DOUBLE PRECISION;")
            print()
        except Exception as e2:
            print(f"  ⚠️  Column check failed: {e2}")

# ── 1. Ports ───────────────────────────────────────────────────────────────────
print("\n📍  Seeding ports …")
ports_raw = load_json("data/ports.json")
if ports_raw is not None:
    add_ports_depth_columns()
    rows = [
        {
            "name":              k,
            "lat":               v.get("lat"),
            "lon":               v.get("lon"),
            "anchorage_depth":   v.get("anchorage_depth"),   # None if column missing — harmless
            "cargo_pier_depth":  v.get("cargo_pier_depth"),
        }
        for k, v in ports_raw.items()
    ]
    n = upsert("ports", rows, "ports")
    print(f"✅  {n} / {len(rows)} ports inserted.")

# ── 2. Tracked IMOs ────────────────────────────────────────────────────────────
print("\n🎯  Seeding tracked_imos …")
imos_raw = load_json("data/tracked_imos.json")
if imos_raw is not None:
    rows = [{"imo": str(i)} for i in imos_raw]
    n = upsert("tracked_imos", rows, "IMOs")
    print(f"✅  {n} tracked IMOs inserted.")

# ── 3. Vessels ─────────────────────────────────────────────────────────────────
print("\n🚢  Seeding vessels …")
vessels_raw = load_json("data/vessels_data.json")
if vessels_raw is not None:
    # vessels_data.json is either a dict keyed by IMO or a list
    if isinstance(vessels_raw, dict):
        vessel_list = list(vessels_raw.values())
    else:
        vessel_list = vessels_raw

    rows = []
    for v in vessel_list:
        row = {k: val for k, val in v.items() if k != "updated_at"}
        # Ensure imo is a string
        row["imo"] = str(row["imo"])
        rows.append(row)

    n = upsert("vessels", rows, "vessels")
    print(f"✅  {n} / {len(rows)} vessels inserted.")

# ── 4. Static vessel cache ─────────────────────────────────────────────────────
# NOTE: static_vessel_cache schema has NO 'name' column.
# Name is served from the vessels table at runtime.
print("\n🗄️   Seeding static_vessel_cache …")
cache_raw = load_json("data/static_vessel_cache.json")
if cache_raw is not None:
    if isinstance(cache_raw, dict):
        items = cache_raw.items()
    else:
        items = ((r["imo"], r) for r in cache_raw)

    rows = []
    for imo, v in items:
        rows.append({
            "imo":              str(imo),
            # "name" intentionally omitted — not in DB schema
            "ship_type":        v.get("ship_type"),
            "flag":             v.get("flag"),
            "deadweight_t":     v.get("deadweight_t"),
            "gross_tonnage":    v.get("gross_tonnage"),
            "year_of_build":    v.get("year_of_build"),
            "length_overall_m": v.get("length_overall_m"),
            "beam_m":           v.get("beam_m"),
            "draught_m":        v.get("draught_m"),
        })
    n = upsert("static_vessel_cache", rows, "cached vessels")
    print(f"✅  {n} / {len(rows)} static cache rows inserted.")

# ── 5. Ship ID map (optional) ──────────────────────────────────────────────────
print("\n🔢  Seeding shipid_map …")
sm = load_json("data/shipid_map.json")
if sm is not None:
    rows = [{"imo": str(k), "shipid": int(v)} for k, v in sm.items()]
    n = upsert("shipid_map", rows, "ship IDs")
    print(f"✅  {n} ship IDs inserted.")

# ── 6. Failure counts (optional) ───────────────────────────────────────────────
print("\n⚠️   Seeding failure_counts …")
fc = load_json("data/failure_counts.json")
if fc is not None:
    rows = [{"imo": str(k), "count": int(v)} for k, v in fc.items()]
    n = upsert("failure_counts", rows, "failure records")
    print(f"✅  {n} failure count rows inserted.")

# ── 7. Sanctioned IMOs ─────────────────────────────────────────────────────────
print("\n🚨  Seeding sanctioned_imos …")
sanc_raw = load_json("data/sanctioned_imos.json")
if sanc_raw is not None:
    # Support both formats:
    #   { "entries": [ {imo, name, lists, program}, … ] }
    #   or a flat list  [ {imo, name, lists, program}, … ]
    if isinstance(sanc_raw, dict):
        entries = sanc_raw.get("entries", [])
    else:
        entries = sanc_raw

    rows = []
    seen = set()
    for e in entries:
        imo = str(e.get("imo", "")).strip()
        if not imo or imo in seen:
            continue
        seen.add(imo)
        # lists must be a Postgres TEXT[] — ensure it's a Python list
        lists_val = e.get("lists", [])
        if isinstance(lists_val, str):
            lists_val = [lists_val]
        rows.append({
            "imo":     imo,
            "name":    e.get("name", ""),
            "lists":   lists_val,
            "program": e.get("program", ""),
        })

    n = upsert("sanctioned_imos", rows, "sanctioned vessels")
    print(f"✅  {n} / {len(rows)} sanctioned vessels inserted.")

print("\n🎉  Seed complete!  Open Supabase → Table Editor to verify each table.")
