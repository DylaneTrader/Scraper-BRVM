"""
Scraper BRVM - Historique de cours (Sikafinance.com)
=====================================================
Récupère les données historiques quotidiennes (OHLCV) pour toutes
les actions et indices de la BRVM.

Les métadonnées (fiche société, secteur, conseil, infos indice) sont gérées
par le script séparé scraper_info_societes.py.
La liste des tickers est centralisée dans brvm_tickers.py.

API découverte : POST /api/general/GetHistos
- Body JSON : {"ticker": "SGBC.ci", "datedeb": "YYYY-MM-DD", "datefin": "YYYY-MM-DD"}
- IMPORTANT : ne PAS inclure le champ "xperiod" sinon les dates sont ignorées
- Limite : 3 mois max par requête (91 jours), sinon erreur "toolong"
- Réponse : {"lst": [{"Date": "DD/MM/YYYY", "Open": x, "High": x, "Low": x, "Close": x, "Volume": x}], "error": ""}
- Données disponibles depuis ~2008

Stratégie : itérer par fenêtres de 90 jours en remontant dans le temps.

Utilisation :
  python scraper_brvm.py              # Scraping complet (incrémental si CSV existant)
  python scraper_brvm.py full         # Alias identique
  python scraper_brvm.py test [TICKER]  # Test sur un seul ticker (défaut SGBC.ci)
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from brvm_tickers import ACTIONS, INDICES, safe_filename

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.sikafinance.com"
API_HISTOS = BASE_URL + "/api/general/GetHistos"

# Plage temporelle
START_DATE = datetime(2008, 1, 1)
END_DATE = datetime.now()

# Fenêtre de 90 jours (< 91 jours pour éviter l'erreur "toolong")
WINDOW_DAYS = 90

# Pause entre requêtes (secondes)
REQUEST_DELAY = 1.0

# Arrêter après N fenêtres vides consécutives
MAX_EMPTY_WINDOWS = 4

# Dossiers de sortie
OUTPUT_DIR = Path(__file__).parent / "data"
ACTIONS_DIR = OUTPUT_DIR / "actions"
INDICES_DIR = OUTPUT_DIR / "indices"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "scraper.log", encoding="utf-8"),
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
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Origin": BASE_URL,
            "Referer": BASE_URL + "/marches/historiques/BRVMC",
        }
    )
    # Initialiser les cookies
    try:
        s.get(BASE_URL, timeout=30)
    except Exception as e:
        log.warning(f"Impossible d'initialiser la session : {e}")
    return s


# ---------------------------------------------------------------------------
# Scraping historique via API JSON
# ---------------------------------------------------------------------------


def fetch_window(
    session: requests.Session,
    ticker: str,
    date_start: datetime,
    date_end: datetime,
) -> list[dict]:
    """Appelle l'API GetHistos pour une fenêtre de dates."""
    payload = {
        "ticker": ticker,
        "datedeb": date_start.strftime("%Y-%m-%d"),
        "datefin": date_end.strftime("%Y-%m-%d"),
    }

    try:
        resp = session.post(API_HISTOS, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(
            f"  Erreur API pour {ticker} ({payload['datedeb']} -> {payload['datefin']}): {e}"
        )
        return []

    error = data.get("error", "")
    if error == "toolong":
        log.warning(f"  Période trop longue, réduction nécessaire")
        return []
    if error == "nodata":
        return []
    if error == "baddt":
        log.warning(f"  Dates incorrectes : {payload}")
        return []
    if error:
        log.warning(f"  Erreur inconnue : {error}")
        return []

    return data.get("lst", [])


def parse_api_date(date_str: str) -> str:
    """Convertit DD/MM/YYYY en YYYY-MM-DD."""
    parts = date_str.split("/")
    if len(parts) == 3:
        return f"{parts[2]}-{parts[1]}-{parts[0]}"
    return date_str


def load_existing_data(ticker: str, is_index: bool) -> pd.DataFrame:
    """Charge le CSV existant d'un ticker s'il existe."""
    output_dir = INDICES_DIR if is_index else ACTIONS_DIR
    filepath = output_dir / f"{safe_filename(ticker)}_historique.csv"
    if not filepath.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, sep=";", decimal=",", parse_dates=["Date"])
        log.info(
            f"  Données existantes : {len(df)} lignes jusqu'au {df['Date'].max().strftime('%Y-%m-%d')}"
        )
        return df
    except Exception as e:
        log.warning(f"  Impossible de lire {filepath} : {e}")
        return pd.DataFrame()


def scrape_history(
    session: requests.Session,
    ticker: str,
    is_index: bool = False,
    since: Optional[datetime] = None,
) -> pd.DataFrame:
    """Scrape l'historique d'un ticker par fenêtres de 90 jours.

    Si `since` est fourni, ne télécharge que les données à partir de cette date
    (mode incrémental). Sinon, télécharge tout depuis START_DATE.
    """
    start_from = since if since else START_DATE
    all_rows = []
    seen_dates = set()

    current_end = END_DATE
    empty_count = 0
    total_requests = 0

    while current_end > start_from and empty_count < MAX_EMPTY_WINDOWS:
        current_start = current_end - timedelta(days=WINDOW_DAYS)
        if current_start < start_from:
            current_start = start_from

        log.info(
            f"  {current_start.strftime('%Y-%m-%d')} -> {current_end.strftime('%Y-%m-%d')}"
        )

        raw_rows = fetch_window(session, ticker, current_start, current_end)
        total_requests += 1

        added = 0
        for r in raw_rows:
            date_iso = parse_api_date(r["Date"])
            if date_iso not in seen_dates:
                seen_dates.add(date_iso)
                row = {
                    "Date": date_iso,
                    "Ouverture": r.get("Open"),
                    "Plus_Haut": r.get("High"),
                    "Plus_Bas": r.get("Low"),
                    "Cloture": r.get("Close"),
                    "Volume_Titres": r.get("Volume"),
                }
                all_rows.append(row)
                added += 1

        if added == 0:
            empty_count += 1
        else:
            empty_count = 0
            log.info(f"    +{added} lignes")

        # Reculer la fenêtre
        current_end = current_start - timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    if not all_rows:
        if since:
            log.info(
                f"  Pas de nouvelles données pour {ticker} depuis {since.strftime('%Y-%m-%d')}"
            )
        else:
            log.warning(f"  Aucune donnée pour {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    log.info(
        f"  Nouvelles données {ticker}: {len(df)} lignes, "
        f"{df['Date'].min().strftime('%Y-%m-%d')} -> {df['Date'].max().strftime('%Y-%m-%d')} "
        f"({total_requests} requêtes)"
    )
    return df


def merge_and_finalize(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne anciennes et nouvelles données, recalcule les colonnes dérivées."""
    if existing_df.empty and new_df.empty:
        return pd.DataFrame()

    if existing_df.empty:
        combined = new_df
    elif new_df.empty:
        combined = existing_df
    else:
        cols_brutes = [
            "Date",
            "Ouverture",
            "Plus_Haut",
            "Plus_Bas",
            "Cloture",
            "Volume_Titres",
        ]
        existing_clean = existing_df[
            [c for c in cols_brutes if c in existing_df.columns]
        ].copy()
        new_clean = new_df[[c for c in cols_brutes if c in new_df.columns]].copy()
        combined = pd.concat([existing_clean, new_clean], ignore_index=True)

    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = (
        combined.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    # Colonnes dérivées
    if "Volume_Titres" in combined.columns:
        combined["Volume_FCFA"] = (
            combined["Volume_Titres"]
            * (combined["Plus_Haut"] + combined["Plus_Bas"])
            / 2
        ).round(0)

    combined["Variation_Pct"] = combined["Cloture"].pct_change() * 100
    combined["Variation_Pct"] = combined["Variation_Pct"].round(2)

    return combined


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------


def save_history(df: pd.DataFrame, ticker: str, output_dir: Path):
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{safe_filename(ticker)}_historique.csv"
    df.to_csv(filepath, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    log.info(f"  -> {filepath} ({len(df)} lignes)")


# ---------------------------------------------------------------------------
# Résumé consolidé
# ---------------------------------------------------------------------------


def build_summary(output_dir: Path):
    """Crée un fichier résumé avec les stats de chaque ticker."""
    summary = []
    for subdir in [ACTIONS_DIR, INDICES_DIR]:
        if not subdir.exists():
            continue
        for csv_file in sorted(subdir.glob("*_historique.csv")):
            try:
                df = pd.read_csv(csv_file, sep=";", decimal=",", parse_dates=["Date"])
                ticker = csv_file.stem.replace("_historique", "").replace("_", ".")
                summary.append(
                    {
                        "Ticker": ticker,
                        "Type": "Action" if subdir == ACTIONS_DIR else "Indice",
                        "Nb_Lignes": len(df),
                        "Date_Min": df["Date"].min().strftime("%Y-%m-%d"),
                        "Date_Max": df["Date"].max().strftime("%Y-%m-%d"),
                        "Dernier_Cours": df.loc[df["Date"].idxmax(), "Cloture"],
                    }
                )
            except Exception:
                continue

    if summary:
        df_summary = pd.DataFrame(summary)
        filepath = output_dir / "resume_scraping.csv"
        df_summary.to_csv(
            filepath, index=False, encoding="utf-8-sig", sep=";", decimal=","
        )
        log.info(f"\nRésumé sauvegardé : {filepath}")
        print(f"\n{'='*70}")
        print(df_summary.to_string(index=False))
        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def run_full_scrape():
    """Scrape complet (incrémental si historique existant) de toutes les actions et indices."""
    session = create_session()
    start_time = time.time()

    # ---- Actions ----
    log.info(f"{'='*60}")
    log.info(f"HISTORIQUE — {len(ACTIONS)} ACTIONS")
    log.info(f"{'='*60}")

    for i, ticker in enumerate(ACTIONS, 1):
        log.info(f"\n[{i}/{len(ACTIONS)}] {ticker}")

        existing_df = load_existing_data(ticker, is_index=False)

        if not existing_df.empty:
            since = existing_df["Date"].max() + timedelta(days=1)
            log.info(f"  Mode incrémental depuis {since.strftime('%Y-%m-%d')}")
        else:
            since = None

        try:
            new_df = scrape_history(session, ticker, is_index=False, since=since)
            final_df = merge_and_finalize(existing_df, new_df)
            save_history(final_df, ticker, ACTIONS_DIR)
        except Exception as e:
            log.error(f"  ERREUR {ticker} : {e}")

    # ---- Indices ----
    log.info(f"\n{'='*60}")
    log.info(f"HISTORIQUE — {len(INDICES)} INDICES")
    log.info(f"{'='*60}")

    for i, ticker in enumerate(INDICES, 1):
        log.info(f"\n[{i}/{len(INDICES)}] {ticker}")

        existing_df = load_existing_data(ticker, is_index=True)

        if not existing_df.empty:
            since = existing_df["Date"].max() + timedelta(days=1)
            log.info(f"  Mode incrémental depuis {since.strftime('%Y-%m-%d')}")
        else:
            since = None

        try:
            new_df = scrape_history(session, ticker, is_index=True, since=since)
            final_df = merge_and_finalize(existing_df, new_df)
            save_history(final_df, ticker, INDICES_DIR)
        except Exception as e:
            log.error(f"  ERREUR {ticker} : {e}")

    # ---- Résumé ----
    elapsed = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"SCRAPING TERMINÉ en {elapsed/60:.1f} minutes")
    log.info(f"{'='*60}")

    build_summary(OUTPUT_DIR)

    log.info(f"\nFichiers dans :")
    log.info(f"  Actions : {ACTIONS_DIR}")
    log.info(f"  Indices : {INDICES_DIR}")
    log.info(f"\n(Infos société/indice : lancer scraper_info_societes.py)")


# ---------------------------------------------------------------------------
# Mode test rapide
# ---------------------------------------------------------------------------


def run_test(ticker: str = "SGBC.ci"):
    """Test rapide sur un seul ticker."""
    session = create_session()
    log.info(f"=== TEST : {ticker} ===")

    is_index = ticker in INDICES
    df = scrape_history(session, ticker, is_index=is_index)

    if not df.empty:
        save_history(df, ticker, INDICES_DIR if is_index else ACTIONS_DIR)
        print(f"\nAperçu ({len(df)} lignes) :")
        print(df.head(10).to_string(index=False))
        print("...")
        print(df.tail(5).to_string(index=False))


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            ticker = sys.argv[2] if len(sys.argv) > 2 else "SGBC.ci"
            run_test(ticker)
        elif sys.argv[1] == "full":
            run_full_scrape()
        else:
            print("Usage:")
            print("  python scraper_brvm.py                 # Scraping complet")
            print("  python scraper_brvm.py full            # Scraping complet")
            print("  python scraper_brvm.py test [TICKER]   # Test sur un ticker")
    else:
        run_full_scrape()
