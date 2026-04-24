from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
import json

app = FastAPI()

def scrape_name(imo):
    url = f"https://www.vesselfinder.com/vessels/details/{imo}"
    r = requests.get(url, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find("h1")
    return title.text.strip() if title else None

@app.get("/vessel/{imo}")
def vessel_name(imo: str):
    name = scrape_name(imo)
    if not name:
        return {"found": False, "name": None}
    return {"found": True, "name": name}
