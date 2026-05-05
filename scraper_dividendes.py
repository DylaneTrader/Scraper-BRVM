"""
Scraper Dividendes BRVM - Sikafinance.com
==========================================
Récupère les dividendes à venir et l'historique pluriannuel des dividendes
depuis https://www.sikafinance.com/marches/dividendes

Deux tables sur la page :
- tbdDiv  : dividendes à venir (Date détachement, Nom, Montant, Rendement)
- tblDiv2 : historique pluriannuel (Div. + Rend. pour chaque année)

Sortie : data/dividendes/
  - dividendes_a_venir.csv
  - dividendes_historique.csv
  - dividendes.json (snapshot complet avec date de scraping)
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.sikafinance.com"
DIVIDENDES_URL = BASE_URL + "/marches/dividendes"

OUTPUT_DIR = Path(__file__).parent / "data" / "dividendes"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def create_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            # Forcer le contournement des caches intermédiaires
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )
    return s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKER_RE = re.compile(r"cotation_([A-Za-z0-9\-]+\.[a-z]{2})", re.IGNORECASE)


def extract_ticker(a_tag) -> str:
    """Extrait le ticker depuis un href /marches/cotation_XXX.yy"""
    if not a_tag:
        return ""
    href = a_tag.get("href", "")
    m = TICKER_RE.search(href)
    return m.group(1) if m else ""


def clean_number(txt: str):
    """'1740,00' -> 1740.0, '-' -> None"""
    if not txt or txt.strip() in {"-", ""}:
        return None
    cleaned = txt.replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def clean_pct(txt: str):
    """'7,32 %' -> 7.32, '-' -> None"""
    if not txt or txt.strip() in {"-", ""}:
        return None
    cleaned = txt.replace("%", "").replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_date(txt: str) -> str:
    """'22/04/2026' -> '2026-04-22'"""
    parts = txt.strip().split("/")
    if len(parts) == 3:
        return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
    return txt.strip()


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------


def fetch_page(session: requests.Session) -> BeautifulSoup:
    # Cache-buster en query string pour contourner tout cache CDN
    url = f"{DIVIDENDES_URL}?_={int(time.time())}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def parse_a_venir(soup: BeautifulSoup, today: datetime) -> list[dict]:
    """Table #tbdDiv : Date détachement | Nom | Montant | Rendement.

    Ajoute un champ 'Statut' (A venir / Passé / A préciser) calculé vs aujourd'hui.
    """
    table = soup.find("table", id="tbdDiv")
    if not table:
        log.warning("Table 'tbdDiv' introuvable")
        return []

    rows = table.find_all("tr")
    data = []
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        nom_cell = tds[1]
        ticker = extract_ticker(nom_cell.find("a"))
        raw_date = tds[0].get_text(strip=True)
        date_iso = parse_date(raw_date)

        # Calcul du statut par rapport à today (00:00)
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_iso):
            d = datetime.strptime(date_iso, "%Y-%m-%d")
            statut = "Passé" if d.date() < today.date() else "A venir"
        else:
            statut = "A préciser"

        data.append(
            {
                "Date_Detachement": date_iso,
                "Statut": statut,
                "Ticker": ticker,
                "Nom": nom_cell.get_text(strip=True),
                "Montant_FCFA": clean_number(tds[2].get_text(strip=True)),
                "Rendement_Pct": clean_pct(tds[3].get_text(strip=True)),
            }
        )
    return data


def parse_historique(soup: BeautifulSoup) -> list[dict]:
    """Table #tblDiv2 : Nom | Div. YYYY | Rend. YYYY | Div. YYYY+1 | ..."""
    table = soup.find("table", id="tblDiv2")
    if not table:
        log.warning("Table 'tblDiv2' introuvable")
        return []

    rows = table.find_all("tr")
    headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]

    # Identifier les années à partir des entêtes "Div. YYYY" / "Rend. YYYY"
    year_cols: list[tuple[int, int, str]] = []  # (idx_div, idx_rend, annee)
    i = 1
    while i < len(headers):
        m_div = re.match(r"Div\.\s*(\d{4})", headers[i])
        if m_div:
            year = m_div.group(1)
            idx_rend = i + 1
            if idx_rend < len(headers) and re.match(
                rf"Rend\.\s*{year}", headers[idx_rend]
            ):
                year_cols.append((i, idx_rend, year))
                i = idx_rend + 1
                continue
        i += 1

    data = []
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if not tds:
            continue
        nom_cell = tds[0]
        ticker = extract_ticker(nom_cell.find("a"))
        nom = nom_cell.get_text(strip=True).replace("\xa0", " ")
        row = {"Ticker": ticker, "Nom": nom}
        for idx_div, idx_rend, year in year_cols:
            div_txt = tds[idx_div].get_text(strip=True) if idx_div < len(tds) else ""
            rend_txt = tds[idx_rend].get_text(strip=True) if idx_rend < len(tds) else ""
            row[f"Div_{year}"] = clean_number(div_txt)
            row[f"Rend_{year}_Pct"] = clean_pct(rend_txt)
        data.append(row)
    return data


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------


def save(a_venir: list[dict], historique: list[dict], now: datetime):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now_iso = now.strftime("%Y-%m-%d %H:%M:%S")
    stamp = now.strftime("%Y-%m-%d")

    if a_venir:
        df = pd.DataFrame(a_venir)
        df.insert(0, "Date_Scraping", stamp)
        path = OUTPUT_DIR / "dividendes_a_venir.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig", sep=";", decimal=",")
        log.info(f"-> {path} ({len(df)} lignes)")

    if historique:
        df = pd.DataFrame(historique)
        df.insert(0, "Date_Scraping", stamp)
        path = OUTPUT_DIR / "dividendes_historique.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig", sep=";", decimal=",")
        log.info(f"-> {path} ({len(df)} lignes)")

    snapshot = {
        "date_scraping": now_iso,
        "source": DIVIDENDES_URL,
        "a_venir": a_venir,
        "historique": historique,
    }
    path = OUTPUT_DIR / "dividendes.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    log.info(f"-> {path}")


def run():
    session = create_session()
    now = datetime.now()
    log.info(f"Scraping {DIVIDENDES_URL}")
    soup = fetch_page(session)

    a_venir = parse_a_venir(soup, now)
    nb_avenir = sum(1 for r in a_venir if r["Statut"] == "A venir")
    nb_passe = sum(1 for r in a_venir if r["Statut"] == "Passé")
    nb_tbd = sum(1 for r in a_venir if r["Statut"] == "A préciser")
    log.info(
        f"Calendrier dividendes : {len(a_venir)} entrées "
        f"(à venir: {nb_avenir}, passés: {nb_passe}, à préciser: {nb_tbd})"
    )

    historique = parse_historique(soup)
    log.info(f"Historique pluriannuel : {len(historique)} sociétés")

    save(a_venir, historique, now)


if __name__ == "__main__":
    start = time.time()
    run()
    log.info(f"Terminé en {time.time()-start:.1f}s")
