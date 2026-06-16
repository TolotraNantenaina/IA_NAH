import argparse
import json
import os
import random
import sys
import time

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCRAPPING_CTRL, SCRAPPING_DATA, SCRAPPING_OUTPUT

WIKI_ORIGIN = "https://mg.wikipedia.org"
BASE_URL = f"{WIKI_ORIGIN}/wiki/"
DEFAULT_TITLES_FILE = os.path.join(SCRAPPING_DATA, "mgwiki-latest-all-titles-in-ns0")

SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

def scrape_wikipedia_article(title: str) -> str:
    url = BASE_URL + title.replace(" ", "_")
    response = requests.get(url, headers=SCRAPE_HEADERS)
    if response.status_code != 200:
        raise Exception(f"Erreur {response.status_code} lors du chargement de la page.")
    soup = BeautifulSoup(response.content, "html.parser")
    content_div = soup.find("div", {"class": "mw-parser-output"})
    if not content_div:
        raise Exception("Contenu de l'article non trouvé")
    paragraphs = content_div.find_all("p")
    text = ""
    for p in paragraphs:
        cleaned = p.get_text().strip()
        if cleaned:
            text += cleaned + "\n\n"
    return text

def load_titles(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_progress(progress_file: str) -> dict:
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"completed": [], "failed": [], "last_index": 0}
    return {"completed": [], "failed": [], "last_index": 0}

def save_progress(progress_file: str, progress: dict):
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def check_article_exists(output_file: str, article_title: str) -> bool:
    if not os.path.exists(output_file):
        return False
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            motif = f"=== {article_title} ==="
            return motif in content
    except Exception:
        return False

def get_all_article_titles(limit_per_request=500, max_pages=None):
    titles = []
    url = f"{WIKI_ORIGIN}/w/api.php"
    apcontinue = None
    session = requests.Session()
    h = dict(SCRAPE_HEADERS)
    h["Accept"] = "application/json"
    session.headers.update(h)
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": limit_per_request,
            "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        batch = data.get("query", {}).get("allpages", [])
        for p in batch:
            titles.append(p["title"])
        if max_pages is not None and len(titles) >= max_pages:
            return titles[:max_pages]
        qc = data.get("continue", {})
        apcontinue = qc.get("apcontinue")
        if not apcontinue:
            break
        time.sleep(0.2)
    return titles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--titles-file",
        default=None,
        help="Liste de titres (défaut: data/mgwiki-latest-all-titles-in-ns0 si présent)",
    )
    parser.add_argument("--from-api", action="store_true", help="Titres via API MediaWiki")
    parser.add_argument("--max-titles", type=int, default=None, help="Limiter le nombre de titres")
    parser.add_argument(
        "--output",
        default="wiki_malagasy.txt",
        help="Fichier de sortie dans out-put/",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Sans confirmation avant la suite")
    args = parser.parse_args()

    os.makedirs(SCRAPPING_OUTPUT, exist_ok=True)
    os.makedirs(SCRAPPING_CTRL, exist_ok=True)

    output_file = os.path.join(SCRAPPING_OUTPUT, args.output)
    progress_file = os.path.join(SCRAPPING_CTRL, "scrapping_wiki_mg_progress.json")

    if args.from_api:
        articles = get_all_article_titles(max_pages=args.max_titles)
        print("Titres (API):", len(articles))
    else:
        path = args.titles_file
        if not path and os.path.isfile(DEFAULT_TITLES_FILE):
            path = DEFAULT_TITLES_FILE
        if not path or not os.path.isfile(path):
            print("Pas de fichier titres, bascule API.")
            articles = get_all_article_titles(max_pages=args.max_titles)
        else:
            articles = load_titles(path)
            if args.max_titles is not None:
                articles = articles[: args.max_titles]
        print("Titres:", len(articles))

    progress = load_progress(progress_file)
    completed_articles = set(progress["completed"])
    failed_articles = set(progress["failed"])
    last_index = progress["last_index"]

    total_articles = len(articles)
    print(f"Total d'articles à traiter: {total_articles}")
    print(f"Articles déjà complétés: {len(completed_articles)}")
    print(f"Articles en échec: {len(failed_articles)}")

    articles_to_process = [a for a in articles if a not in completed_articles]
    print(f"Articles restants à traiter: {len(articles_to_process)}")

    failed_to_retry = list(failed_articles)
    if failed_to_retry:
        print(f"\n=== Retraitement des articles en échec ===")
        print(f"Articles en échec à retraiter: {len(failed_to_retry)}")
        with open(output_file, "a", encoding="utf-8") as f:
            for idx, article in enumerate(failed_to_retry, 1):
                print(f"Retraitement de l'article échoué: {article} ({idx}/{len(failed_to_retry)})")
                try:
                    texte = scrape_wikipedia_article(article)
                    f.write(f"=== {article} ===\n")
                    f.write(texte + "\n\n")
                    f.flush()
                    print(f"Article {article} retraité avec succès!")
                    failed_articles.discard(article)
                    completed_articles.add(article)
                    if article in progress["failed"]:
                        progress["failed"].remove(article)
                    progress["completed"].append(article)
                    try:
                        article_index = articles.index(article)
                        if article_index > progress["last_index"]:
                            progress["last_index"] = article_index
                    except ValueError:
                        pass
                    save_progress(progress_file, progress)
                    time.sleep(random.uniform(1, 3))
                except Exception as e:
                    print(f"Échec du retraitement pour l'article '{article}': {e}")
                    time.sleep(5)

    remaining_articles = [a for a in articles_to_process if a not in failed_articles]
    print(f"\n=== Traitement des articles restants ===")
    print(f"Articles restants à traiter: {len(remaining_articles)}")

    if len(remaining_articles) > 0 and not args.yes:
        response = input(f"Continuer avec les {len(remaining_articles)} articles restants? (y/n): ")
        if response.lower() != "y":
            print("Arrêt du script.")
            return

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, article in enumerate(remaining_articles, 1):
            if not article.strip():
                print(f"Titre vide, article ignoré. ({last_index + idx}/{len(articles)})")
                continue
            if check_article_exists(output_file, article):
                print(f"Article '{article}' déjà présent dans le fichier, ignoré. ({last_index + idx}/{len(articles)})")
                completed_articles.add(article)
                progress["completed"].append(article)
                try:
                    article_index = articles.index(article)
                    if article_index > progress["last_index"]:
                        progress["last_index"] = article_index
                except ValueError:
                    pass
                save_progress(progress_file, progress)
                continue
            try:
                texte = scrape_wikipedia_article(article)
                f.write(f"=== {article} ===\n")
                f.write(texte + "\n\n")
                f.flush()
                pourcentage = ((last_index + idx) / len(articles)) * 100 if articles else 0
                print(f"Article {article} sauvegardé. ({pourcentage:.2f}% terminé / {last_index + idx}/{len(articles)})")
                completed_articles.add(article)
                progress["completed"].append(article)
                try:
                    article_index = articles.index(article)
                    if article_index > progress["last_index"]:
                        progress["last_index"] = article_index
                except ValueError:
                    pass
                save_progress(progress_file, progress)
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                pourcentage = ((last_index + idx) / len(articles)) * 100 if articles else 0
                print(f"Erreur pour l'article '{article}': {e} ({pourcentage:.2f}% terminé / {last_index + idx}/{len(articles)})")
                failed_articles.add(article)
                progress["failed"].append(article)
                save_progress(progress_file, progress)
                time.sleep(5)

    print(f"\nScraping terminé!")
    print(f"Articles réussis: {len(completed_articles)}")
    print(f"Articles en échec: {len(failed_articles)}")
    print(f"Dernier index: {progress['last_index']}")
    print(f"Sortie: {output_file}")
    print(f"Progression: {progress_file}")

if __name__ == "__main__":
    main()
