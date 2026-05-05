"""
Scraper News BRVM - Sikafinance.com
=====================================
Récupère les actualités depuis sikafinance.com en accédant
directement à chaque article par son ID numérique.

Stratégie :
  - Chaque article a une URL : /marches/slug_ID (ex: /marches/xxx_60945)
  - Le slug n'est pas vérifié par le serveur : /marches/a_60945 fonctionne
  - On itère sur les IDs de façon décroissante (récent → ancien)
  - ID 1 = juin 2012, ID ~60 945 = avril 2026
  - Historique en CSV, mode incrémental par défaut

Utilisation :
  python scraper_news_brvm.py              # Mode incrémental (défaut)
  python scraper_news_brvm.py full         # Tout scraper depuis l'ID 1
  python scraper_news_brvm.py test         # Test sur 5 articles récents
  python scraper_news_brvm.py resume       # Reprendre un scraping full interrompu
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import logging
import re
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.sikafinance.com"
ARTICLE_URL_TEMPLATE = BASE_URL + "/marches/a_{article_id}"

# Plage d'IDs connue
MIN_ARTICLE_ID = 1  # Premier article (juin 2012)
MAX_ARTICLE_ID = 61500  # Marge au-dessus du dernier connu (~60945)

# Pause entre requêtes (secondes)
REQUEST_DELAY = 0.8

# Nombre d'IDs 404 consécutifs avant d'arrêter (certains IDs n'existent pas)
MAX_CONSECUTIVE_MISSING = 200

# Sauvegarde intermédiaire tous les N articles
SAVE_EVERY = 50

# Dossier de sortie
OUTPUT_DIR = Path(__file__).parent / "data"
NEWS_DIR = OUTPUT_DIR / "news"
HISTORY_FILE = NEWS_DIR / "actualites_brvm.csv"
PROGRESS_FILE = NEWS_DIR / "scraping_progress.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent / "scraper_news.log", encoding="utf-8"
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
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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
# Scraping d'un article par ID
# ---------------------------------------------------------------------------


def extract_json_ld(soup: BeautifulSoup) -> dict:
    """Extrait le bloc JSON-LD (données structurées) de la page article."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                for item in data:
                    if item.get("@type") in ("NewsArticle", "Article"):
                        return item
            elif isinstance(data, dict) and data.get("@type") in (
                "NewsArticle",
                "Article",
            ):
                return data
        except Exception:
            continue
    return {}


def scrape_article_by_id(session: requests.Session, article_id: int) -> dict | None:
    """
    Scrape un article par son ID numérique.
    Retourne un dict avec tous les champs, ou None si l'article n'existe pas.
    """
    url = ARTICLE_URL_TEMPLATE.format(article_id=article_id)

    try:
        resp = session.get(url, timeout=30, allow_redirects=True)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        return None
    except Exception as e:
        log.warning(f"  Erreur réseau ID {article_id} : {e}")
        return None

    # Vérifier que la page contient bien un article (pas une page d'erreur custom)
    soup = BeautifulSoup(resp.text, "lxml")
    ld = extract_json_ld(soup)

    # Si pas de JSON-LD et pas de <h1>, c'est probablement une page vide/erreur
    h1 = soup.find("h1")
    if not ld and not h1:
        return None

    # Récupérer l'URL finale (après redirection éventuelle)
    final_url = resp.url

    # ---- Titre ----
    titre = ""
    if ld.get("headline"):
        titre = ld["headline"].strip()
    if not titre and h1:
        titre = h1.get_text(strip=True)
    if not titre:
        og = soup.find("meta", property="og:title")
        if og:
            titre = og.get("content", "").strip()

    # Pas de titre = pas un vrai article
    if not titre:
        return None

    # ---- Date publication ----
    date_pub = ""
    if ld.get("datePublished"):
        date_pub = ld["datePublished"]
    if not date_pub:
        meta_date = soup.find("meta", property="article:published_time")
        if meta_date:
            date_pub = meta_date.get("content", "")

    # ---- Auteur ----
    auteur = ""
    if ld.get("author"):
        author_data = ld["author"]
        if isinstance(author_data, dict):
            auteur = author_data.get("name", "")
        elif isinstance(author_data, list) and author_data:
            auteur = author_data[0].get("name", "")
    if not auteur:
        for tag in soup.find_all(
            ["span", "div", "p"], class_=re.compile(r"auteur|author|byline", re.I)
        ):
            auteur = tag.get_text(strip=True)
            if auteur:
                break

    # ---- Catégorie ----
    categorie = ""
    if ld.get("articleSection"):
        categorie = ld["articleSection"]
    if not categorie:
        breadcrumb = soup.find(
            ["nav", "ol", "ul"], class_=re.compile(r"breadcrumb", re.I)
        )
        if breadcrumb:
            items = [li.get_text(strip=True) for li in breadcrumb.find_all(["li", "a"])]
            if len(items) >= 2:
                categorie = items[-2]
    if not categorie:
        meta_section = soup.find("meta", property="article:section")
        if meta_section:
            categorie = meta_section.get("content", "")

    # ---- Contenu complet ----
    contenu = ""

    # Stratégie 1 : JSON-LD articleBody
    if ld.get("articleBody"):
        contenu = ld["articleBody"].strip()

    # Stratégie 2 : div de contenu avec classe spécifique
    if not contenu:
        for candidate in soup.find_all(
            ["div", "article", "section"],
            class_=re.compile(
                r"article.body|article.content|content.article|news.content|post.content|entry.content|article-text|article_content",
                re.I,
            ),
        ):
            text = candidate.get_text(separator="\n", strip=True)
            if len(text) > 200:
                contenu = text
                break

    # Stratégie 3 : balise <article>
    if not contenu:
        article_tag = soup.find("article")
        if article_tag:
            for tag in article_tag.find_all(
                ["nav", "aside", "script", "style", "footer"]
            ):
                tag.decompose()
            contenu = article_tag.get_text(separator="\n", strip=True)

    # Stratégie 4 : plus grand bloc de texte
    if not contenu or len(contenu) < 100:
        best_block = ""
        for div in soup.find_all(["div", "section"]):
            classes = " ".join(div.get("class", []))
            if re.search(
                r"nav|header|footer|sidebar|menu|widget|ad|pub|social|share|comment",
                classes,
                re.I,
            ):
                continue
            text = div.get_text(separator="\n", strip=True)
            if len(text) > len(best_block):
                best_block = text
        if len(best_block) > 200:
            contenu = best_block

    contenu = re.sub(r"\n{3,}", "\n\n", contenu).strip()

    # ---- Image principale ----
    image_url = ""
    if ld.get("image"):
        img = ld["image"]
        if isinstance(img, dict):
            image_url = img.get("url", "")
        elif isinstance(img, str):
            image_url = img
        elif isinstance(img, list) and img:
            first = img[0]
            image_url = first.get("url", "") if isinstance(first, dict) else first

    return {
        "id": article_id,
        "titre": titre,
        "date_publication": date_pub,
        "auteur": auteur,
        "categorie": categorie,
        "contenu": contenu,
        "image_url": image_url,
        "url": final_url,
        "date_scraping": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Historique CSV + progression
# ---------------------------------------------------------------------------


def load_history() -> pd.DataFrame:
    """Charge le CSV d'historique s'il existe."""
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_FILE, sep=";", encoding="utf-8-sig")
        log.info(f"Historique existant : {len(df)} articles")
        return df
    except Exception as e:
        log.warning(f"Impossible de lire l'historique : {e}")
        return pd.DataFrame()


def save_history(df: pd.DataFrame):
    """Sauvegarde le DataFrame dans le CSV d'historique."""
    if df.empty:
        return
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig", sep=";")
    log.info(f"Historique sauvegardé : {len(df)} articles -> {HISTORY_FILE}")


def get_known_ids(df: pd.DataFrame) -> set:
    """Retourne l'ensemble des IDs déjà dans l'historique."""
    if df.empty or "id" not in df.columns:
        return set()
    return set(df["id"].dropna().astype(int).tolist())


def get_max_known_id(df: pd.DataFrame) -> int:
    """Retourne le plus grand ID déjà dans l'historique."""
    if df.empty or "id" not in df.columns:
        return 0
    return int(df["id"].max())


def save_progress(last_id: int, direction: str):
    """Sauvegarde la progression du scraping pour reprise."""
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(
            {
                "last_id": last_id,
                "direction": direction,
                "timestamp": datetime.now().isoformat(),
            },
            f,
        )


def load_progress() -> dict | None:
    """Charge la progression si elle existe."""
    if not PROGRESS_FILE.exists():
        return None
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def merge_and_save(existing_df: pd.DataFrame, new_articles: list[dict]):
    """Fusionne les nouveaux articles avec l'historique existant et sauvegarde."""
    if not new_articles:
        return existing_df

    new_df = pd.DataFrame(new_articles)

    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Dédupliquer par ID
    combined = combined.drop_duplicates(subset=["id"], keep="last")

    # Trier par ID décroissant (plus récent en premier)
    combined = combined.sort_values("id", ascending=False).reset_index(drop=True)

    save_history(combined)
    return combined


# ---------------------------------------------------------------------------
# Fonction de découverte du dernier ID existant
# ---------------------------------------------------------------------------


def _id_exists_robust(session: requests.Session, article_id: int, probe_width: int = 5) -> bool:
    """Teste si une zone autour de `article_id` contient au moins un article.

    Les IDs sur Sikafinance ont des trous isolés (articles supprimés/404) : un seul
    404 ne signifie pas qu'on est au-delà du dernier article. On teste donc une
    fenêtre de `probe_width` IDs voisins avant de conclure.
    """
    for offset in range(probe_width):
        art = scrape_article_by_id(session, article_id - offset)
        time.sleep(REQUEST_DELAY * 0.3)
        if art:
            return True
    return False


def find_latest_article_id(session: requests.Session, start_guess: int = 61500) -> int:
    """Cherche le dernier ID valide.

    Stratégie robuste aux trous :
      1. Probe exponentiel vers le haut depuis `start_guess` pour trouver une
         borne vraiment vide (fenêtre de 5 IDs voisins tous 404).
      2. Recherche binaire entre la dernière borne basse connue (pleine) et
         la borne haute vide, en testant la présence via `_id_exists_robust`.
      3. Balayage linéaire final vers le haut pour trouver l'ID maximal exact.
    """
    log.info("Recherche du dernier article existant...")

    # --- Phase 1 : trouver une borne haute réellement vide ---
    low = start_guess
    high = start_guess
    # Assurer que `low` est bien rempli (sinon redescendre par pas de 500)
    while low > MIN_ARTICLE_ID and not _id_exists_robust(session, low, probe_width=3):
        low = max(MIN_ARTICLE_ID, low - 500)
    # Puis monter par doublement jusqu'à trouver une zone vide
    step = 500
    while _id_exists_robust(session, high, probe_width=5):
        low = high
        high += step
        step = min(step * 2, 10000)
        log.info(f"  probe up: low={low} high={high}")

    # --- Phase 2 : recherche binaire entre low (plein) et high (vide) ---
    while low < high - 1:
        mid = (low + high) // 2
        if _id_exists_robust(session, mid, probe_width=5):
            low = mid
        else:
            high = mid

    # --- Phase 3 : balayage linéaire fin au-dessus de `low` pour trouver le max exact ---
    # `low` est un ID quelque part dans la zone peuplée. On remonte tant qu'on trouve
    # au moins un article dans les 10 IDs suivants.
    best = low
    for aid in range(low, low + 30):
        art = scrape_article_by_id(session, aid)
        time.sleep(REQUEST_DELAY * 0.3)
        if art:
            best = aid

    log.info(f"Dernier article trouvé : ID {best}")
    return best


# ---------------------------------------------------------------------------
# Pipelines de scraping
# ---------------------------------------------------------------------------


def scrape_id_range(
    session: requests.Session,
    start_id: int,
    end_id: int,
    known_ids: set,
    direction: str = "down",
) -> list[dict]:
    """
    Scrape les articles dans une plage d'IDs.

    direction="down" : de start_id vers end_id en décrémentant (récent → ancien)
    direction="up"   : de start_id vers end_id en incrémentant (ancien → récent)
    """
    all_articles = []
    consecutive_missing = 0
    scraped_count = 0
    skipped_count = 0

    if direction == "down":
        ids = range(start_id, end_id - 1, -1)
    else:
        ids = range(start_id, end_id + 1)

    total = abs(end_id - start_id) + 1
    existing_df = load_history()

    for i, article_id in enumerate(ids):
        # Skip si déjà connu
        if article_id in known_ids:
            skipped_count += 1
            continue

        article = scrape_article_by_id(session, article_id)

        if article is None:
            consecutive_missing += 1
            if consecutive_missing % 50 == 0:
                log.info(
                    f"  ID {article_id} : {consecutive_missing} IDs vides consécutifs..."
                )
            if consecutive_missing >= MAX_CONSECUTIVE_MISSING:
                log.info(
                    f"  Arrêt : {MAX_CONSECUTIVE_MISSING} IDs consécutifs sans article"
                )
                break
        else:
            consecutive_missing = 0
            all_articles.append(article)
            known_ids.add(article_id)
            scraped_count += 1

            log.info(
                f"  [{scraped_count}] ID {article_id} : "
                f"{article['titre'][:60]} ({article['date_publication'][:10]})"
            )

            # Sauvegarde intermédiaire
            if scraped_count % SAVE_EVERY == 0:
                log.info(
                    f"  --- Sauvegarde intermédiaire ({scraped_count} articles) ---"
                )
                existing_df = merge_and_save(existing_df, all_articles)
                all_articles = []
                save_progress(article_id, direction)

        time.sleep(REQUEST_DELAY)

        # Progression
        if (i + 1) % 500 == 0:
            pct = (i + 1) / total * 100
            log.info(
                f"  Progression : {i+1}/{total} IDs ({pct:.1f}%) - "
                f"{scraped_count} articles, {skipped_count} déjà connus, "
                f"{consecutive_missing} vides consécutifs"
            )

    # Sauvegarde finale
    if all_articles:
        existing_df = merge_and_save(existing_df, all_articles)

    log.info(
        f"\nBilan : {scraped_count} articles récupérés, "
        f"{skipped_count} déjà connus, {total} IDs parcourus"
    )
    return all_articles


def run_incremental():
    """Mode incrémental : scrape uniquement les nouveaux articles (IDs > max connu)."""
    log.info("=" * 60)
    log.info("MODE INCREMENTAL - News BRVM")
    log.info("=" * 60)

    session = create_session()
    existing_df = load_history()
    known_ids = get_known_ids(existing_df)
    max_known = get_max_known_id(existing_df)

    if max_known > 0:
        log.info(f"Dernier ID connu : {max_known}")
        start_id = max_known + 1
    else:
        log.info("Pas d'historique. Lancement en mode full à la place.")
        run_full()
        return

    # Chercher le dernier ID existant sur le site
    latest_id = find_latest_article_id(session, start_guess=max_known + 500)
    log.info(f"Plage à scraper : ID {start_id} -> {latest_id}")

    if start_id > latest_id:
        log.info("Aucun nouvel article.")
        return

    start_time = time.time()
    scrape_id_range(session, start_id, latest_id, known_ids, direction="up")
    elapsed = time.time() - start_time

    log.info(f"\nScraping terminé en {elapsed/60:.1f} minutes")

    # Nettoyage du fichier de progression
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


def run_full():
    """Scraping complet de tous les IDs (du plus récent au plus ancien)."""
    log.info("=" * 60)
    log.info("MODE FULL - News BRVM")
    log.info("=" * 60)

    session = create_session()
    existing_df = load_history()
    known_ids = get_known_ids(existing_df)

    # Trouver le dernier article
    latest_id = find_latest_article_id(session)
    log.info(f"Plage : ID {latest_id} -> 1")
    log.info(f"Articles déjà connus : {len(known_ids)}")

    start_time = time.time()
    scrape_id_range(session, latest_id, MIN_ARTICLE_ID, known_ids, direction="down")
    elapsed = time.time() - start_time

    log.info(f"\nScraping complet terminé en {elapsed/60:.1f} minutes")

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


def run_resume():
    """Reprendre un scraping interrompu."""
    log.info("=" * 60)
    log.info("MODE REPRISE - News BRVM")
    log.info("=" * 60)

    progress = load_progress()
    if not progress:
        log.info("Pas de progression sauvegardée. Lancement en mode full.")
        run_full()
        return

    session = create_session()
    existing_df = load_history()
    known_ids = get_known_ids(existing_df)

    last_id = progress["last_id"]
    direction = progress["direction"]

    log.info(f"Reprise depuis ID {last_id} (direction: {direction})")
    log.info(f"Articles déjà connus : {len(known_ids)}")

    start_time = time.time()

    if direction == "down":
        scrape_id_range(session, last_id, MIN_ARTICLE_ID, known_ids, direction="down")
    else:
        latest_id = find_latest_article_id(session)
        scrape_id_range(session, last_id, latest_id, known_ids, direction="up")

    elapsed = time.time() - start_time
    log.info(f"\nReprise terminée en {elapsed/60:.1f} minutes")

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


def run_test():
    """Test rapide sur 5 articles récents."""
    log.info("=" * 60)
    log.info("MODE TEST - 5 articles récents")
    log.info("=" * 60)

    session = create_session()

    # Partir du dernier ID connu et tester quelques articles
    test_ids = [60945, 60943, 60942, 60900, 60850]
    for aid in test_ids:
        article = scrape_article_by_id(session, aid)
        if article:
            log.info(f"\nID {aid}:")
            log.info(f"  Titre     : {article['titre']}")
            log.info(f"  Date      : {article['date_publication']}")
            log.info(f"  Auteur    : {article['auteur']}")
            log.info(f"  Catégorie : {article['categorie']}")
            log.info(f"  Contenu   : {len(article['contenu'])} caractères")
            log.info(f"  URL       : {article['url']}")
            log.info(f"  Aperçu    : {article['contenu'][:150]}...")
        else:
            log.info(f"\nID {aid} : article non trouvé")
        time.sleep(REQUEST_DELAY)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "full":
            run_full()
        elif cmd == "test":
            run_test()
        elif cmd == "resume":
            run_resume()
        else:
            print("Usage:")
            print("  python scraper_news_brvm.py          # Mode incrémental (défaut)")
            print(
                "  python scraper_news_brvm.py full     # Scraping complet (ID 1 -> dernier)"
            )
            print(
                "  python scraper_news_brvm.py resume   # Reprendre un scraping interrompu"
            )
            print("  python scraper_news_brvm.py test     # Test sur 5 articles")
    else:
        run_incremental()
