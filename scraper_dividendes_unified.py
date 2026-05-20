"""
Scraper Dividendes BRVM - Unifié (brvm.org + sikafinance.com)
=============================================================
Source unique de vérité pour les dividendes BRVM. Consomme les deux
sources et produit UN SEUL fichier fusionné.

Règles de fusion (BRVM = officiel, prioritaire) :
  - Date_Detachement / Date_Paiement : BRVM si date ISO valide, sinon Sika
  - Montant_Net_FCFA                : BRVM (warning si écart > 0.01 vs Sika)
  - Rendement_Pct                   : Sika uniquement (BRVM ne le publie pas)
  - Avis_URL                        : BRVM uniquement
  - Statut                          : recalculé après fusion

Granularité : 1 ligne = (Ticker, Exercice).

Sorties dans data/dividendes/ :
  - dividendes.csv          : tout (toutes années, tous statuts)
  - dividendes.json         : snapshot complet avec métadonnées
  - dividendes_a_venir.csv  : vue filtrée (Statut = 'A venir')
  - dividendes_passes.csv   : vue filtrée (Statut = 'Passé')

Lance simplement : python scraper_dividendes_unified.py
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib3
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Réutilisation des helpers existants
from scraper_dividendes import (
    fetch_page as fetch_sika_page,
    parse_a_venir as parse_sika_a_venir,
    parse_historique as parse_sika_historique,
    create_session as create_sika_session,
)
from scraper_dividendes_brvm import (
    fetch_page as fetch_brvm_page,
    find_dividend_table,
    parse_table as parse_brvm_table,
    detect_last_page,
    create_session as create_brvm_session,
    REQUEST_DELAY,
    MAX_PAGES,
)
from brvm_emetteur_mapping import lookup_ticker

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUTPUT_DIR = Path(__file__).parent / "data" / "dividendes"
AVIS_DIR = Path(__file__).parent / "data" / "avis_dividendes"
AVIS_DELAY = 0.5  # politesse entre téléchargements
LEGACY_FILES = [
    "dividendes_a_venir.csv",
    "dividendes_historique.csv",
    "dividendes_brvm.csv",
    "dividendes_brvm.json",
    "dividendes.json",  # ancien snapshot Sika
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collecte des sources
# ---------------------------------------------------------------------------


def collect_brvm() -> list[dict]:
    """Récupère tous les paiements depuis brvm.org (pagination complète)."""
    session = create_brvm_session()
    log.info("[BRVM] Scraping brvm.org/fr/esv/paiement-de-dividendes")
    soup = fetch_brvm_page(session, 0)
    last = min(detect_last_page(soup), MAX_PAGES)
    records: list[dict] = []
    seen: set[tuple] = set()
    for page in range(0, last + 1):
        if page > 0:
            time.sleep(REQUEST_DELAY)
            try:
                soup = fetch_brvm_page(session, page)
            except Exception as e:
                log.error(f"[BRVM] Page {page} : {e}")
                continue
        table = find_dividend_table(soup)
        if not table:
            continue
        for r in parse_brvm_table(table):
            sig = (r["Ticker"], r["Exercice"], r["Date_Paiement"], r["Montant_Net_Raw"])
            if sig in seen:
                continue
            seen.add(sig)
            records.append(r)
    log.info(f"[BRVM] {len(records)} paiements collectés")
    return records


def collect_sika() -> tuple[list[dict], list[dict]]:
    """Récupère a_venir + historique depuis sikafinance.com."""
    session = create_sika_session()
    log.info("[SIKA] Scraping sikafinance.com/marches/dividendes")
    now = datetime.now()
    soup = fetch_sika_page(session)
    a_venir = parse_sika_a_venir(soup, now)
    historique = parse_sika_historique(soup)
    log.info(f"[SIKA] À venir : {len(a_venir)} | Historique : {len(historique)} sociétés")
    return a_venir, historique


# ---------------------------------------------------------------------------
# Normalisation vers le schéma unifié
# ---------------------------------------------------------------------------


def _year_from_iso(date_iso: str) -> str:
    if isinstance(date_iso, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", date_iso):
        return date_iso[:4]
    return ""


def _is_iso(d) -> bool:
    return isinstance(d, str) and bool(re.match(r"^\d{4}-\d{2}-\d{2}$", d))


_EXERCICE_FROM_PDF = re.compile(r"exercice[_\-]?(\d{4})", re.IGNORECASE)


def _clean_exercice(val) -> str:
    """Normalise l'exercice en string sans suffixe '.0'."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_brvm(brvm_rows: list[dict]) -> list[dict]:
    """Convertit chaque ligne BRVM vers le schéma unifié."""
    out = []
    for r in brvm_rows:
        ticker = r.get("Ticker", "")
        if not ticker:
            continue  # lignes orphelines (Emetteur vide sur brvm.org)
        exercice = _clean_exercice(r.get("Exercice"))
        # Fallback : inférer l'exercice depuis l'URL du PDF Avis
        # (ex. '..._exercice_2025_-_sicable_ci.pdf' -> '2025')
        if not exercice:
            avis = r.get("Avis_URL", "") or ""
            m = _EXERCICE_FROM_PDF.search(avis)
            if m:
                exercice = m.group(1)
        # Dernier recours : déduire de la Date_Paiement (année - 1)
        if not exercice:
            dp = r.get("Date_Paiement", "")
            if _is_iso(dp):
                exercice = str(int(dp[:4]) - 1)
        out.append({
            "Ticker": ticker,
            "Nom_Canonique": r.get("Nom_Canonique") or lookup_ticker(ticker),
            "Exercice": exercice,
            "Date_Detachement": r.get("Date_Detachement", "") or "",
            "Date_Paiement": r.get("Date_Paiement", "") or "",
            "Montant_Net_FCFA": r.get("Montant_Net_FCFA"),
            "Rendement_Pct": None,
            "Avis_URL": r.get("Avis_URL", "") or "",
            "_src": {"brvm.org"},
        })
    return out


def normalize_sika(a_venir: list[dict], historique: list[dict]) -> list[dict]:
    """Convertit Sika a_venir + historique vers le schéma unifié.

    a_venir = exercice "en cours" (le plus récent dans Div_YYYY de l'historique).
    historique = lignes Div_YYYY / Rend_YYYY → on éclate en (Ticker, Exercice).
    """
    out: list[dict] = []

    # 1) a_venir : Sika n'expose pas explicitement l'exercice → on déduit
    #    via l'historique : l'exercice "à venir" correspond à l'année la plus
    #    récente où Div_YYYY est encore non renseigné OU l'année juste avant
    #    la date de détachement. Heuristique simple : si Date_Detachement
    #    est en année N, exercice = N-1 (paiement du dividende de l'exercice
    #    précédent — convention BRVM).
    for r in a_venir:
        ticker = r.get("Ticker", "")
        if not ticker:
            continue
        date_det = r.get("Date_Detachement", "")
        year_det = _year_from_iso(date_det)
        if year_det:
            exercice = str(int(year_det) - 1)
        else:
            # 'A préciser' → on suppose exercice N-1 par rapport à l'année courante
            exercice = str(datetime.now().year - 1)

        # Date_Detachement contient parfois 'A préciser' au lieu d'une ISO
        date_det_iso = date_det if re.match(r"^\d{4}-\d{2}-\d{2}$", date_det or "") else ""

        out.append({
            "Ticker": ticker,
            "Nom_Canonique": r.get("Nom_Canonique") or lookup_ticker(ticker),
            "Exercice": exercice,
            "Date_Detachement": date_det_iso,
            "Date_Paiement": "",
            "Montant_Net_FCFA": r.get("Montant_Net_FCFA"),
            "Rendement_Pct": r.get("Rendement_Pct"),
            "Avis_URL": "",
            "_src": {"sikafinance.com"},
        })

    # 2) historique : pour chaque ticker, éclater chaque Div_YYYY non null
    for r in historique:
        ticker = r.get("Ticker", "")
        if not ticker:
            continue
        for key, val in r.items():
            m = re.match(r"^Div_(\d{4})$", key)
            if not m or val in (None, "", "nan"):
                continue
            try:
                if pd.isna(val):
                    continue
            except Exception:
                pass
            year = m.group(1)
            rend = r.get(f"Rend_{year}_Pct")
            out.append({
                "Ticker": ticker,
                "Nom_Canonique": r.get("Nom_Canonique") or lookup_ticker(ticker),
                "Exercice": year,
                "Date_Detachement": "",
                "Date_Paiement": "",
                "Montant_Net_FCFA": val,
                "Rendement_Pct": rend,
                "Avis_URL": "",
                "_src": {"sikafinance.com"},
            })

    return out


# ---------------------------------------------------------------------------
# Fusion BRVM > Sika
# ---------------------------------------------------------------------------


def _coalesce_date(brvm_val: str, sika_val: str) -> str:
    if _is_iso(brvm_val):
        return brvm_val
    if _is_iso(sika_val):
        return sika_val
    return ""


def _coalesce_montant(brvm_val, sika_val, ticker: str, exercice: str) -> float | None:
    """BRVM gagne. Log warning si écart > 0.01 FCFA avec Sika."""
    if brvm_val is not None and not (isinstance(brvm_val, float) and pd.isna(brvm_val)):
        if sika_val is not None and not (isinstance(sika_val, float) and pd.isna(sika_val)):
            try:
                if abs(float(brvm_val) - float(sika_val)) > 0.01:
                    log.warning(
                        f"[FUSION] Montant divergent {ticker} exercice {exercice} : "
                        f"BRVM={brvm_val} vs Sika={sika_val} → BRVM gagne"
                    )
            except (TypeError, ValueError):
                pass
        return float(brvm_val)
    if sika_val is not None and not (isinstance(sika_val, float) and pd.isna(sika_val)):
        return float(sika_val)
    return None


def merge(rows: list[dict]) -> list[dict]:
    """Fusionne les lignes par clé (Ticker, Exercice). BRVM prioritaire."""
    bucket: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["Ticker"], r["Exercice"])
        if key not in bucket:
            bucket[key] = {**r, "_src": set(r["_src"])}
            continue
        cur = bucket[key]
        cur_is_brvm = "brvm.org" in cur["_src"]
        new_is_brvm = "brvm.org" in r["_src"]
        # Identifier qui est BRVM, qui est Sika pour la résolution des conflits
        brvm_row = cur if cur_is_brvm else (r if new_is_brvm else {})
        sika_row = cur if not cur_is_brvm else (r if not new_is_brvm else {})

        merged = {
            "Ticker": cur["Ticker"],
            "Exercice": cur["Exercice"],
            "Nom_Canonique": cur.get("Nom_Canonique") or r.get("Nom_Canonique"),
            "Date_Detachement": _coalesce_date(
                brvm_row.get("Date_Detachement", ""),
                sika_row.get("Date_Detachement", ""),
            ),
            "Date_Paiement": _coalesce_date(
                brvm_row.get("Date_Paiement", ""),
                sika_row.get("Date_Paiement", ""),
            ),
            "Montant_Net_FCFA": _coalesce_montant(
                brvm_row.get("Montant_Net_FCFA"),
                sika_row.get("Montant_Net_FCFA"),
                cur["Ticker"], cur["Exercice"],
            ),
            "Rendement_Pct": (
                sika_row.get("Rendement_Pct") if sika_row else cur.get("Rendement_Pct")
            ),
            "Avis_URL": brvm_row.get("Avis_URL", "") or cur.get("Avis_URL", ""),
            "_src": cur["_src"] | r["_src"],
        }
        bucket[key] = merged
    return list(bucket.values())


# ---------------------------------------------------------------------------
# Statut & sortie
# ---------------------------------------------------------------------------


def compute_statut(row: dict, today: datetime) -> str:
    d = row.get("Date_Detachement", "")
    if not _is_iso(d):
        return "A préciser"
    dt = datetime.strptime(d, "%Y-%m-%d").date()
    return "Passé" if dt < today.date() else "A venir"


def _avis_local_path(ticker: str, exercice: str, avis_url: str) -> Path:
    """Construit le chemin local du PDF : data/avis_dividendes/<TICKER>/<EXERCICE>_<basename>.pdf"""
    # On garde la fin du nom officiel BRVM pour traçabilité, préfixé par l'exercice
    basename = avis_url.rsplit("/", 1)[-1] or "avis.pdf"
    if not basename.lower().endswith(".pdf"):
        basename += ".pdf"
    prefix = f"{exercice}_" if exercice else ""
    return AVIS_DIR / ticker / f"{prefix}{basename}"


def download_avis(records: list[dict]) -> dict[tuple[str, str], str]:
    """Télécharge les PDF d'avis BRVM. Skip si déjà présent.

    Retourne un mapping (Ticker, Exercice) -> chemin local relatif au projet.
    """
    AVIS_DIR.mkdir(parents=True, exist_ok=True)
    session = create_brvm_session()
    project_root = Path(__file__).parent

    paths: dict[tuple[str, str], str] = {}
    n_download = n_skip = n_fail = 0

    for r in records:
        url = r.get("Avis_URL", "")
        if not url:
            continue
        ticker = r.get("Ticker", "")
        exercice = r.get("Exercice", "")
        if not ticker:
            continue

        path = _avis_local_path(ticker, exercice, url)
        rel = path.relative_to(project_root).as_posix()

        if path.exists() and path.stat().st_size > 0:
            paths[(ticker, exercice)] = rel
            n_skip += 1
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
            if path.stat().st_size == 0:
                raise RuntimeError("fichier vide")
            paths[(ticker, exercice)] = rel
            n_download += 1
            log.info(f"[AVIS] {ticker} {exercice} -> {rel}")
        except Exception as e:
            log.warning(f"[AVIS] échec {ticker} {exercice} ({url}) : {e}")
            n_fail += 1
            if path.exists() and path.stat().st_size == 0:
                path.unlink()  # ne pas laisser un fichier vide qui ferait croire au cache
        finally:
            time.sleep(AVIS_DELAY)

    log.info(f"[AVIS] téléchargés : {n_download} | cache : {n_skip} | échecs : {n_fail}")
    return paths


def to_dataframe(records: list[dict], now: datetime) -> pd.DataFrame:
    today = now
    rows = []
    for r in records:
        rows.append({
            "Ticker": r["Ticker"],
            "Nom_Canonique": r.get("Nom_Canonique", ""),
            "Exercice": r["Exercice"],
            "Statut": compute_statut(r, today),
            "Date_Detachement": r.get("Date_Detachement", ""),
            "Date_Paiement": r.get("Date_Paiement", ""),
            "Montant_Net_FCFA": r.get("Montant_Net_FCFA"),
            "Rendement_Pct": r.get("Rendement_Pct"),
            "Avis_URL": r.get("Avis_URL", ""),
            "Avis_Path": r.get("Avis_Path", ""),
            "Sources": ",".join(sorted(r.get("_src", set()))),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Tri : exercice décroissant puis date_paiement décroissante puis ticker
    df = df.sort_values(
        by=["Exercice", "Date_Paiement", "Ticker"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    df.insert(0, "Date_Scraping", now.strftime("%Y-%m-%d"))
    return df


def cleanup_legacy():
    for name in LEGACY_FILES:
        path = OUTPUT_DIR / name
        if path.exists():
            path.unlink()
            log.info(f"[CLEANUP] supprimé : {path.name}")


def save(df: pd.DataFrame, now: datetime):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_legacy()

    csv_path = OUTPUT_DIR / "dividendes.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    log.info(f"-> {csv_path} ({len(df)} lignes)")

    avenir = df[df["Statut"] == "A venir"]
    passes = df[df["Statut"] == "Passé"]
    apreciser = df[df["Statut"] == "A préciser"]

    p1 = OUTPUT_DIR / "dividendes_a_venir.csv"
    avenir.to_csv(p1, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    log.info(f"-> {p1} ({len(avenir)} à venir)")

    p2 = OUTPUT_DIR / "dividendes_passes.csv"
    passes.to_csv(p2, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    log.info(f"-> {p2} ({len(passes)} passés)")

    snapshot = {
        "date_scraping": now.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": ["brvm.org", "sikafinance.com"],
        "nb_records": len(df),
        "nb_a_venir": len(avenir),
        "nb_passes": len(passes),
        "nb_a_preciser": len(apreciser),
        "records": df.where(pd.notna(df), None).to_dict(orient="records"),
    }
    json_path = OUTPUT_DIR / "dividendes.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"-> {json_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run():
    start = time.time()
    now = datetime.now()

    brvm_raw = collect_brvm()
    sika_avenir, sika_hist = collect_sika()

    brvm_norm = normalize_brvm(brvm_raw)
    sika_norm = normalize_sika(sika_avenir, sika_hist)
    log.info(f"[NORMALISE] BRVM={len(brvm_norm)} | Sika={len(sika_norm)}")

    merged = merge(brvm_norm + sika_norm)
    log.info(f"[FUSION] {len(merged)} lignes uniques (Ticker, Exercice)")

    avis_paths = download_avis(merged)
    for r in merged:
        r["Avis_Path"] = avis_paths.get((r["Ticker"], r["Exercice"]), "")

    df = to_dataframe(merged, now)
    n_avenir = (df["Statut"] == "A venir").sum() if not df.empty else 0
    n_passe = (df["Statut"] == "Passé").sum() if not df.empty else 0
    n_tbd = (df["Statut"] == "A préciser").sum() if not df.empty else 0
    log.info(f"[STATUT] À venir : {n_avenir} | Passés : {n_passe} | À préciser : {n_tbd}")

    save(df, now)
    log.info(f"Terminé en {time.time() - start:.1f}s")


if __name__ == "__main__":
    run()
