"""
Scraper Infos Sociétés & Indices BRVM - Sikafinance.com
========================================================
Récupère les métadonnées (hors historique de cours) :
  - Actions : fiche société + secteur + conseil sikafinance
  - Indices : nom complet + ISIN

L'historique des cours est géré séparément par scraper_brvm.py.
La liste des tickers est centralisée dans brvm_tickers.py.

Sorties :
  data/societes/{TICKER}_societe.json    (actions)
  data/indices/{TICKER}_info.json        (indices)

Utilisation :
  python scraper_info_societes.py              # Toutes les actions + indices
  python scraper_info_societes.py actions      # Actions uniquement
  python scraper_info_societes.py indices      # Indices uniquement
  python scraper_info_societes.py test [TICKER]  # Test sur un ticker (défaut SGBC.ci)
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from brvm_tickers import ACTIONS, INDICES, safe_filename

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.sikafinance.com"
SOCIETE_URL = BASE_URL + "/marches/societe/{ticker}"
SECTEUR_URL = BASE_URL + "/marches/secteur/{ticker}"
CONSEIL_URL = BASE_URL + "/analyses/conseil/{ticker}"
INDICE_URL = BASE_URL + "/marches/historiques/{ticker}"

REQUEST_DELAY = 0.5

OUTPUT_DIR = Path(__file__).parent / "data"
SOCIETE_DIR = OUTPUT_DIR / "societes"
INDICES_DIR = OUTPUT_DIR / "indices"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent / "scraper_info.log", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session HTTP
# ---------------------------------------------------------------------------


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
            "Referer": BASE_URL,
        }
    )
    try:
        s.get(BASE_URL, timeout=30)
    except Exception as e:
        log.warning(f"Impossible d'initialiser la session : {e}")
    return s


# ---------------------------------------------------------------------------
# Fiche société
# ---------------------------------------------------------------------------


def scrape_societe(session: requests.Session, ticker: str) -> dict:
    """Récupère les informations société d'une action."""
    url = SOCIETE_URL.format(ticker=ticker)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        log.error(f"  Erreur société {ticker} : {e}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    info = {"ticker": ticker}

    # Nom commercial depuis le <h1> du bloc quotebarE "XYZ, fiche société"
    quote_bar = soup.find(class_="quotebarE")
    if quote_bar:
        h1 = quote_bar.find("h1")
        if h1:
            txt = h1.get_text(" ", strip=True)
            m = re.match(r"^(.+?)\s*,\s*fiche", txt, re.IGNORECASE)
            nom = m.group(1).strip() if m else txt
            if nom:
                info["Nom"] = nom

    # Paires clé-valeur des tableaux
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) >= 2:
                key = tds[0].get_text(strip=True)
                val = tds[1].get_text(strip=True)
                if key and val and len(key) < 100:
                    clean_key = re.sub(r"[:\s]+$", "", key).strip()
                    if clean_key:
                        info[clean_key] = val

    # Paires <dt>/<dd>
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            key = dt.get_text(strip=True)
            val = dd.get_text(strip=True)
            if key and val:
                info[key] = val

    # Regex ciblées
    text = soup.get_text("\n", strip=True)
    patterns = {
        "ISIN": r"([A-Z]{2}\d{10})",
        "Nombre_Actions": r"(?:Nombre d.actions|Nombre de titres)\s*[:\-]?\s*([\d\s,.]+)",
        "Flottant_Pct": r"(?:Flottant|Free\s*float)\s*[:\-]?\s*([\d,. ]+%)",
    }
    for field, pattern in patterns.items():
        if field not in info:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info[field] = match.group(1).strip()

    # Paires <p><b>Label : </b> value</p>
    for p in soup.find_all("p"):
        b = p.find("b")
        if not b:
            continue
        label = b.get_text(strip=True).rstrip(":").strip()
        full = p.get_text(" ", strip=True)
        b_text = b.get_text(" ", strip=True)
        value = full.replace(b_text, "", 1).lstrip(": ").strip()
        if label and value and label not in info:
            info[label] = value

    # Normaliser les champs attendus
    field_map = {
        "Nombre de titres": "Nombre_Titres",
        "Flottant": "Flottant",
        "Valorisation de la société": "Valorisation",
    }
    for src, dst in field_map.items():
        if src in info and dst not in info:
            info[dst] = info[src]

    # Description : premier paragraphe long sans <b> label
    for p in soup.find_all("p"):
        if p.find("b"):
            continue
        txt = p.get_text(strip=True)
        if len(txt) > 80:
            info["Description"] = txt
            break

    return info


def scrape_secteur(session: requests.Session, ticker: str) -> dict:
    """Récupère le secteur d'activité depuis /marches/secteur/{ticker}."""
    url = SECTEUR_URL.format(ticker=ticker)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"  Erreur secteur {ticker} : {e}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    result = {}

    title = soup.find("title")
    if title:
        m = re.search(
            r"dans le secteur\s+(.+?)(?:\s+à la BRVM)?$",
            title.get_text(strip=True),
            re.IGNORECASE,
        )
        if m:
            result["Secteur"] = m.group(1).strip()

    if "Secteur" not in result:
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            m = re.search(
                r"appartient au secteur\s+([^,]+)", meta["content"], re.IGNORECASE
            )
            if m:
                result["Secteur"] = m.group(1).strip()

    return result


def scrape_conseil(session: requests.Session, ticker: str) -> dict:
    """Récupère le conseil sikafinance.com depuis /analyses/conseil/{ticker}."""
    url = CONSEIL_URL.format(ticker=ticker)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"  Erreur conseil {ticker} : {e}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    result = {}

    # Nom depuis le <h1> du bloc quotebarE : "ABIDJAN CATERING : analyses et conseils"
    nom = None
    quote_bar = soup.find(class_="quotebarE")
    if quote_bar:
        h1 = quote_bar.find("h1")
        if h1:
            txt = h1.get_text(strip=True)
            m = re.match(r"^(.+?)\s*:\s*analyses", txt, re.IGNORECASE)
            nom = m.group(1).strip() if m else txt
    if not nom:
        title = soup.find("title")
        if title:
            m = re.search(
                r"conseils sur\s+(.+?)\s+en bourse",
                title.get_text(strip=True),
                re.IGNORECASE,
            )
            if m:
                nom = m.group(1).strip()
    if nom:
        result["Nom"] = nom

    # Bloc "Le conseil sikafinance.com"
    anchor = soup.find(
        lambda tag: tag.name == "b"
        and "conseil sikafinance.com" in tag.get_text(strip=True).lower()
    )
    block = anchor.find_parent("table") if anchor else None

    if block:
        img = block.find("img", alt=lambda v: v and "conseil" in v.lower())
        if not img:
            img = block.find("img")
        if img and img.get("src"):
            src = img["src"]
            if src.startswith("/"):
                src = BASE_URL + src
            result["Conseil_Image_URL"] = src
            result["Conseil_Image_Nom"] = img["src"].rsplit("/", 1)[-1]

        if img:
            td = img.find_parent("td")
            if td:
                for i in td.find_all("img"):
                    i.decompose()
                texte = td.get_text(" ", strip=True)
                if texte:
                    result["Conseil_Texte"] = texte

        warn = block.find("p", class_="f11")
        if warn:
            result["Conseil_Avertissement"] = warn.get_text(" ", strip=True)

    return result


# ---------------------------------------------------------------------------
# Info indice
# ---------------------------------------------------------------------------


def scrape_indice_info(session: requests.Session, ticker: str) -> dict:
    """Récupère le nom complet (et l'ISIN si présent) d'un indice."""
    url = INDICE_URL.format(ticker=ticker)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"  Erreur info indice {ticker} : {e}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    result = {"Symbol": ticker}

    nom = None
    quote_bar = soup.find(class_="quotebarE")
    if quote_bar:
        h1 = quote_bar.find("h1")
        if h1:
            txt = h1.get_text(" ", strip=True)
            nom = re.sub(
                r"\s*-\s*Donn[ée]es historiques.*$", "", txt, flags=re.IGNORECASE
            ).strip()

    if not nom:
        title = soup.find("title")
        if title:
            txt = title.get_text(" ", strip=True)
            nom = re.split(
                r",\s*donn[ée]es", txt, maxsplit=1, flags=re.IGNORECASE
            )[0].strip()
            nom = re.sub(
                r"\s*-\s*Donn[ée]es historiques.*$", "", nom, flags=re.IGNORECASE
            ).strip()

    if nom:
        result["Name"] = nom

    # ISIN (BRVMxxxxxxxxx, 12 caractères)
    text = soup.get_text(" ", strip=True)
    m = re.search(r"\b(BRVM[0-9A-Z]{8})\b\s*-\s*" + re.escape(ticker), text)
    if m:
        result["ISIN"] = m.group(1)

    return result


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------


def save_societe(info: dict, ticker: str):
    if not info or len(info) <= 1:
        log.warning(f"  Rien à sauver pour {ticker}")
        return
    SOCIETE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SOCIETE_DIR / f"{safe_filename(ticker)}_societe.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    log.info(f"  -> {filepath}")


def save_indice_info(info: dict, ticker: str):
    if not info or not info.get("Name"):
        log.warning(f"  Pas de nom d'indice pour {ticker}")
        return
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = INDICES_DIR / f"{safe_filename(ticker)}_info.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    log.info(f"  -> {filepath}")


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def run_actions():
    session = create_session()
    log.info("=" * 60)
    log.info(f"INFOS SOCIÉTÉS — {len(ACTIONS)} actions")
    log.info("=" * 60)
    for i, ticker in enumerate(ACTIONS, 1):
        log.info(f"\n[{i}/{len(ACTIONS)}] {ticker}")
        try:
            info = scrape_societe(session, ticker)
            info.update(scrape_secteur(session, ticker))
            info.update(scrape_conseil(session, ticker))
            save_societe(info, ticker)
        except Exception as e:
            log.error(f"  ERREUR {ticker} : {e}")
        time.sleep(REQUEST_DELAY)


def run_indices():
    session = create_session()
    log.info("=" * 60)
    log.info(f"INFOS INDICES — {len(INDICES)} indices")
    log.info("=" * 60)
    for i, ticker in enumerate(INDICES, 1):
        log.info(f"\n[{i}/{len(INDICES)}] {ticker}")
        try:
            info = scrape_indice_info(session, ticker)
            save_indice_info(info, ticker)
        except Exception as e:
            log.error(f"  ERREUR {ticker} : {e}")
        time.sleep(REQUEST_DELAY)


def run_all():
    start = time.time()
    run_actions()
    run_indices()
    log.info(f"\nTerminé en {(time.time()-start)/60:.1f} minutes")


def run_test(ticker: str):
    session = create_session()
    log.info(f"=== TEST : {ticker} ===")
    if ticker in INDICES:
        info = scrape_indice_info(session, ticker)
        save_indice_info(info, ticker)
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        info = scrape_societe(session, ticker)
        info.update(scrape_secteur(session, ticker))
        info.update(scrape_conseil(session, ticker))
        save_societe(info, ticker)
        print(json.dumps(info, ensure_ascii=False, indent=2)[:1000])


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "actions":
            run_actions()
        elif cmd == "indices":
            run_indices()
        elif cmd == "test":
            ticker = sys.argv[2] if len(sys.argv) > 2 else "SGBC.ci"
            run_test(ticker)
        else:
            print("Usage:")
            print("  python scraper_info_societes.py              # Tout")
            print("  python scraper_info_societes.py actions")
            print("  python scraper_info_societes.py indices")
            print("  python scraper_info_societes.py test [TICKER]")
    else:
        run_all()
