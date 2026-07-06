#!/usr/bin/env python3
"""
VesselTracker — Push notification tester.

Isolates the /push/send path from the full scraper. Uses the SAME env vars
(API_SECRET, WORKER_URL) as scrape_vesselfinder.py, so a green run here proves
the shared secret and endpoint are aligned.

Env vars:
  WORKER_URL    (required) e.g. https://vesseltracker.asmahri1.workers.dev
  API_SECRET    (required) shared secret; must match the Worker's env.API_SECRET
  TEST_USER_ID  (optional) a real user UUID to actually deliver a test push to

Exit codes: 0 = auth OK (secret aligned), 1 = 403 (secret mismatch), 2 = other.
"""
import os
import sys
import json
import time

try:
    import requests
except ImportError:
    print("requests not installed — run: pip install requests")
    sys.exit(2)

WORKER_URL   = os.getenv("WORKER_URL", "").rstrip("/")
API_SECRET   = os.getenv("API_SECRET", "")
TEST_USER_ID = os.getenv("TEST_USER_ID", "").strip()


def mask(s: str) -> str:
    if not s:
        return "(empty)"
    if len(s) <= 8:
        return "****"
    return f"{s[:4]}…{s[-4:]} ({len(s)} chars)"


def line(c="─"):
    print(c * 60)


def main():
    line("═")
    print("  VesselTracker — Push tester")
    line("═")
    print(f"  WORKER_URL   : {WORKER_URL or '(empty)'}")
    print(f"  API_SECRET   : {mask(API_SECRET)}")
    print(f"  TEST_USER_ID : {TEST_USER_ID or '(none — auth-only test)'}")
    line()

    if not WORKER_URL:
        print("❌ WORKER_URL is empty. Set it in the workflow env / secrets.")
        return 2
    if not API_SECRET:
        print("❌ API_SECRET is empty. Set it in the workflow env / secrets.")
        return 2

    # Auth-only test if no real user given: use a dummy UUID. The Worker checks
    # the secret BEFORE looking up subscriptions, so:
    #   403 -> secret mismatch (the real bug)
    #   200/400/404 -> secret ACCEPTED (auth path is fine)
    user_ids = [TEST_USER_ID] if TEST_USER_ID else ["00000000-0000-0000-0000-000000000000"]
    payload = {
        "user_ids": user_ids,
        "title": "🚢 VesselTracker — test",
        "body": "Push tester — if you see this, delivery works.",
        "tag": f"vt-test-{int(time.time())}",
        "type": "test",
    }

    url = f"{WORKER_URL}/push/send"
    print(f"→ POST {url}")
    try:
        r = requests.post(
            url,
            json=payload,
            headers={"X-API-Secret": API_SECRET, "Content-Type": "application/json"},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error reaching the Worker: {e}")
        print("   → Check WORKER_URL is correct and the Worker is deployed.")
        return 2

    print(f"← HTTP {r.status_code}")
    body_preview = (r.text or "")[:300]
    print(f"  body: {body_preview}")
    line()

    if r.status_code == 403:
        print("❌ 403 FORBIDDEN — API_SECRET MISMATCH.")
        print("   The secret sent (X-API-Secret) ≠ the Worker's env.API_SECRET.")
        print("   Fix: set the SAME value in all 4 places, then redeploy the")
        print("   Worker and restart the VM service:")
        print("     • GitHub Actions secret  API_SECRET")
        print("     • Cloudflare Worker       API_SECRET  (redeploy!)")
        print("     • Cloudflare Pages        API_SECRET")
        print("     • VM /etc/vessel-api.env  API_SECRET  (systemctl restart)")
        return 1

    if r.status_code in (200, 400, 404):
        print("✅ SECRET ACCEPTED — auth path is correct.")
        if r.status_code == 200:
            try:
                data = r.json()
                sent = data.get("sent", 0)
                failed = data.get("failed", 0)
                print(f"   Delivery: {sent} sent, {failed} failed.")
                if not TEST_USER_ID:
                    print("   (dummy user → 0 delivered is expected; secret is fine)")
                elif sent == 0:
                    print("   ⚠️  0 delivered to a real user → check that user has an")
                    print("       active push subscription (enabled push in the app).")
            except json.JSONDecodeError:
                print("   (200 but non-JSON body)")
        else:
            print(f"   {r.status_code} = auth passed, no valid target (expected for dummy user).")
        return 0

    print(f"⚠️  Unexpected status {r.status_code}. Inspect the body above.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
