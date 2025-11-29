import json
import re
import html
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.vesselfinder.com/vessels/details/"
ROOT = Path(__file__).resolve().parents[1]
TRACKED_IMOS_FILE = ROOT / "data" / "tracked_imos.json"
OUTPUT_FILE = ROOT / "data" / "vessels_data.json"


def load_tracked_imos():
    with open(TRACKED_IMOS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tracked_imos", [])


def fetch_html_for_imo(imo: str) -> str:
    url = f"{BASE_URL}{imo}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text


def extract_djson(html_text: str) -> dict:
    m = re.search(r'<div id="djson"[^>]*data-json=\'(.*?)\'', html_text, re.DOTALL)
    if not m:
        return {}
    raw = html.unescape(m.group(1))
    return json.loads(raw)


def parse_vessel(html_text: str, imo: str) -> dict:
    soup = BeautifulSoup(html_text, "html.parser")

    # NAME
    name_tag = soup.find("h1", class_="title")
    name = name_tag.get_text(strip=True) if name_tag else None

    # SUMMARY TEXT
    summary = soup.find("p", class_="text2")
    position_area = None
    commercial_type = None
    flag = None
    year_built = None
    age_years_text = None

    if summary:
        txt = " ".join(summary.get_text(" ", strip=True).split())

        m_area = re.search(r"is at (.+?) reported", txt)
        if m_area:
            position_area = m_area.group(1).strip()

        m_type = re.search(r"is a (.+?) built in", txt)
        if m_type:
            commercial_type = m_type.group(1).strip()

        m_flag = re.search(r"flag of (.+?)\.", txt)
        if m_flag:
            flag = m_flag.group(1).strip()

        m_year = re.search(r"built in (\d{4})", txt)
        if m_year:
            year_built = int(m_year.group(1))
            age_years_text = f"{datetime.utcnow().year - year_built} years old"

    # TABLES
    mmsi = None
    callsign = None
    ais_type_text = None
    current_draught_m = None
    last_pos_age = None
    last_pos_time_utc = None

    tables = soup.find_all("table", class_="aparams")
    for tbl in tables:
        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) != 2:
                continue

            label = tds[0].get_text(strip=True)
            value = " ".join(tds[1].get_text(" ", strip=True).split())

            if label == "IMO / MMSI":
                parts = [p.strip() for p in value.split("/") if p.strip()]
                if len(parts) == 2:
                    mmsi = parts[1]

            elif label == "Callsign":
                callsign = value or None

            elif label == "AIS Type":
                ais_type_text = value or None

            elif label == "Current draught":
                m = re.search(r"([\d\.]+)", value)
                if m:
                    current_draught_m = float(m.group(1))

            elif label == "Position received":
                span_age = tds[1].find("span", class_="red")
                if span_age:
                    last_pos_age = span_age.get_text(strip=True)

                svg = tds[1].find("svg")
                if svg and svg.has_attr("data-title"):
                    last_pos_time_utc = svg["data-title"]

    # VOYAGE DATA
    destination_port_name = None
    destination_port_code = None
    destination_flag = None
    ata_text = None
    arrival_status = None

    voyage_header = soup.find("h2", id="lim")
    if voyage_header:
        block = voyage_header.find_parent("div", class_="s0")

        if block:
            dest_flag_div = block.find("div", class_="flag-icon")
            if dest_flag_div and dest_flag_div.has_attr("title"):
                destination_flag = dest_flag_div["title"]

            dest_link = block.find("a", class_="_npNa")
            if dest_link:
                destination_port_name = dest_link.get_text(strip=True)
                href = dest_link.get("href", "")
                m = re.search(r"/ports/([^\"/?]+)", href)
                if m:
                    destination_port_code = m.group(1)

            value_div = block.find("div", class_="_value")
            if value_div:
                span1 = value_div.find("span", class_="_mcol12")
                if span1:
                    ata_text = span1.get_text(strip=True).replace("ATA: ", "")

                span_status = value_div.find("span", class_="_arrLb")
                if span_status:
                    arrival_status = span_status.get_text(strip=True)

    # LAST PORT
    last_port_name = None
    last_port_code = None
    last_port_atd_utc = None
    last_port_atd_age = None

    if voyage_header:
        block = voyage_header.find_parent("div", class_="s0")
        last_block = block.find_next("div", class_="vi__r1") if block else None
        if last_block:
            link = last_block.find("a", class_="_npNa")
            if link:
                last_port_name = link.get_text(strip=True)
                href = link.get("href", "")
                m = re.search(r"/ports/([^\"/?]+)", href)
                if m:
                    last_port_code = m.group(1)

            value_div = last_block.find("div", class_="_value")
            if value_div:
                txt = " ".join(value_div.get_text(" ", strip=True).split())

                m = re.search(r"ATD:\s+([^()]+)", txt)
                if m:
                    last_port_atd_utc = m.group(1).strip()

                m = re.search(r"\((.+?)\)", txt)
                if m:
                    last_port_atd_age = m.group(1).strip()

    # DJSON
    djson = extract_djson(html_text)
    lat = djson.get("ship_lat")
    lon = djson.get("ship_lon")
    sog_kn = djson.get("ship_sog")
    cog_deg = djson.get("ship_cog")
    sar = djson.get("sar", False)

    vessel = {
        "name": name,
        "imo": imo,
        "mmsi": mmsi,
        "callsign": callsign,
        "commercial_type": commercial_type,
        "ais_type_text": ais_type_text,
        "flag": flag,
        "year_built": year_built,
        "age_years_text": age_years_text,

        "position_area": position_area,
        "lat": lat,
        "lon": lon,
        "sog_kn": sog_kn,
        "cog_deg": cog_deg,
        "nav_status": None,
        "last_pos_age": last_pos_age,
        "last_pos_time_utc": last_pos_time_utc,
        "sar": sar,

        "current_draught_m": current_draught_m,

        "destination_port_name": destination_port_name,
        "destination_port_code": destination_port_code,
        "destination_flag": destination_flag,
        "ata_text": ata_text,
        "arrival_status": arrival_status,

        "last_port_name": last_port_name,
        "last_port_code": last_port_code,
        "last_port_atd_utc": last_port_atd_utc,
        "last_port_atd_age": last_port_atd_age
    }

    return vessel


def main():
    imos = load_tracked_imos()
    results = []

    for imo in imos:
        try:
            print(f"Fetching IMO {imo}")
            html_text = fetch_html_for_imo(imo)
            vessel_data = parse_vessel(html_text, imo)
            results.append(vessel_data)

        except Exception as e:
            print(f"Error for IMO {imo}: {e}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"vessels": results}, f, ensure_ascii=False, indent=2)

    print("Done:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
