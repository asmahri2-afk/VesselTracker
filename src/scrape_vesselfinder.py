from datetime import datetime, timezone

def scrape_vesselfinder(imo: str) -> dict:
    """
    TEMPORARY STUB for testing.

    Returns fake but realistic AIS data so we can test:
    - workflow
    - state file
    - WhatsApp alerts
    - nearest port + distance logic
    """
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "imo": imo,
        "name": f"TEST VESSEL {imo}",
        # Somewhere off Laayoune
        "lat": 27.20,
        "lon": -13.40,
        "sog": 11.5,          # knots
        "cog": 300.0,         # degrees
        "last_pos_utc": now_utc,
        "destination": "DAKHLA",
    }
