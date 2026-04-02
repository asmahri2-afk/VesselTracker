"""
seed_supabase.py
────────────────
STEP 3 — Import all JSON data into Supabase.
Place in ROOT of your repo and run via GitHub Actions.
"""

import json, os
from pathlib import Path
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

sb    = create_client(SUPABASE_URL, SUPABASE_KEY)
CHUNK = 500

def upsert(table, rows):
    if not rows:
        print(f"  (no rows to insert)")
        return
    for i in range(0, len(rows), CHUNK):
        batch = rows[i:i+CHUNK]
        sb.table(table).upsert(batch).execute()
        print(f"  {min(i+CHUNK, len(rows))}/{len(rows)} rows inserted")

# ── 1. Ports ───────────────────────────────────────────────────────────────────
print("\n📍 Seeding ports...")
ports_raw = json.loads(Path("data/ports.json").read_text())
rows = [{"name": k, "lat": v["lat"], "lon": v["lon"]} for k, v in ports_raw.items()]
upsert("ports", rows)
print(f"✅ {len(rows)} ports done.")

# ── 2. Tracked IMOs ────────────────────────────────────────────────────────────
print("\n🎯 Seeding tracked_imos...")
imos = json.loads(Path("data/tracked_imos.json").read_text())
rows = [{"imo": str(i)} for i in imos]
upsert("tracked_imos", rows)
print(f"✅ {len(rows)} tracked IMOs done.")

# ── 3. Vessels — keep ALL fields from vessels_data.json ────────────────────────
print("\n🚢 Seeding vessels...")
vessels_raw = json.loads(Path("data/vessels_data.json").read_text())
rows = []
for v in vessels_raw.values():
    row = dict(v)
    row.pop("updated_at", None)  # let Supabase set this automatically
    rows.append(row)
upsert("vessels", rows)
print(f"✅ {len(rows)} vessels done.")

# ── 4. Static vessel cache ─────────────────────────────────────────────────────
print("\n🗄️  Seeding static_vessel_cache...")
cache_raw = json.loads(Path("data/static_vessel_cache.json").read_text())
rows = []
for imo, v in cache_raw.items():
    rows.append({
        "imo":              str(imo),
        "name":             v.get("name"),
        "ship_type":        v.get("ship_type"),
        "flag":             v.get("flag"),
        "deadweight_t":     v.get("deadweight_t"),
        "gross_tonnage":    v.get("gross_tonnage"),
        "year_of_build":    v.get("year_of_build"),
        "length_overall_m": v.get("length_overall_m"),
        "beam_m":           v.get("beam_m"),
        "draught_m":        v.get("draught_m"),
    })
upsert("static_vessel_cache", rows)
print(f"✅ {len(rows)} cached vessels done.")

# ── 5. Ship ID map ─────────────────────────────────────────────────────────────
print("\n🔢 Seeding shipid_map...")
sm = json.loads(Path("data/shipid_map.json").read_text())
rows = [{"imo": k, "shipid": v} for k, v in sm.items()]
upsert("shipid_map", rows)
print(f"✅ {len(rows)} ship IDs done.")

# ── 6. Failure counts ──────────────────────────────────────────────────────────
print("\n⚠️  Seeding failure_counts...")
fc = json.loads(Path("data/failure_counts.json").read_text())
rows = [{"imo": k, "count": v} for k, v in fc.items()]
upsert("failure_counts", rows)
print(f"✅ {len(rows)} done.")

# ── 7. Sanctioned IMOs ─────────────────────────────────────────────────────────
print("\n🚨 Seeding sanctioned_imos...")
sanc_raw = json.loads(Path("data/sanctioned_imos.json").read_text())
entries  = sanc_raw.get("entries", [])
rows = [
    {
        "imo":     e["imo"],
        "name":    e.get("name", ""),
        "lists":   e.get("lists", []),
        "program": e.get("program", ""),
    }
    for e in entries if e.get("imo")
]
upsert("sanctioned_imos", rows)
print(f"✅ {len(rows)} sanctioned vessels done.")

print("\n🎉 All done! Check Supabase → Table Editor to verify.")
