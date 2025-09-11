import requests
from bs4 import BeautifulSoup
import time
import random
import os
import json
import re
from urllib.parse import urljoin, unquote

# Conversion DOC -> TXT
from spire.doc import Document, FileFormat

PAGE_URL = "https://nybaiboly.net/Bible.htm"
ROOT_URL = "https://nybaiboly.net/"

# Headers pour simuler un navigateur
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
    'Connection': 'keep-alive',
}

def load_progress(progress_file: str) -> dict:
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"completed": [], "failed": [], "last_index": 0}
    return {"completed": [], "failed": [], "last_index": 0}

def save_progress(progress_file: str, progress: dict):
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = re.sub(r'\s+', ' ', name)
    return name

def get_liste_livres():
    """
    Récupère les livres sur https://nybaiboly.net/Bible.htm.
    Retourne une liste de dict: { 'title': 'Genesisy', 'doc_url': 'https://...' }
    """
    resp = requests.get(PAGE_URL, headers=headers, timeout=30)
    if resp.status_code != 200:
        print(f"Erreur HTTP {resp.status_code} en chargeant la page des livres.")
        return []

    soup = BeautifulSoup(resp.content, "html.parser")
    livres = []

    # Tous les liens qui pointent vers Bible/BibleMalagasyDoc-...
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "Bible/BibleMalagasyHtm-" in href:
            label = a.get_text(strip=True)
            if not label:
                continue

            # Normaliser URL absolue
            absolute = urljoin(ROOT_URL, href.replace("Bible/BibleMalagasyHtm-", "Bible/BibleMalagasyDoc-"))
            
            # Forcer l’extension .doc (au cas où ce serait .htm)
            if absolute.lower().endswith(".htm"):
                absolute = absolute[:-4] + ".doc"
            elif not absolute.lower().endswith(".doc"):
                # Certains liens peuvent déjà être .docx, on prend .doc si dispo
                absolute = absolute + ""  # laisser tel quel si .docx (rare)
            livres.append({
                "title": unquote(label),
                "doc_url": absolute
            })

    # Dédupliquer en conservant l’ordre
    seen = set()
    uniques = []
    for item in livres:
        key = (item["title"], item["doc_url"])
        if key not in seen:
            seen.add(key)
            uniques.append(item)

    return uniques

def download_doc(url: str, dest_path: str):
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def convert_doc_to_txt(src_doc: str, dst_txt: str):
    doc = Document()
    doc.LoadFromFile(src_doc)  # .doc ancien supporté
    doc.SaveToFile(dst_txt, FileFormat.Txt)
    doc.Close()

def main():
    # Chemins
    base_dir = "D:/Scrapping"
    out_dir = f"{base_dir}/out-put"
    doc_dir = f"{out_dir}/doc"
    txt_dir = f"{out_dir}/txt"
    ctrl_dir = f"{base_dir}/ctrl"
    merged_output = f"{out_dir}/baiboly_malagasy_protestant.txt"
    progress_file = f"{ctrl_dir}/baiboly_protestant_progress.json"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(ctrl_dir, exist_ok=True)

    progress = load_progress(progress_file)
    completed = set(progress.get("completed", []))
    failed = set(progress.get("failed", []))
    last_index = progress.get("last_index", 0)

    livres = get_liste_livres()
    if not livres:
        print("Erreur: Impossible de récupérer la liste des livres.")
        return

    total = len(livres)
    print(f"Total de livres à traiter: {total}")
    print(f"Livres déjà complétés: {len(completed)}")
    print(f"Livres en échec: {len(failed)}")

    # Retraitement des échecs d’abord
    failed_to_retry = [l for l in livres if l["title"] in failed]
    if failed_to_retry:
        print("\n=== Retraitement des livres en échec ===")
        with open(merged_output, "a", encoding="utf-8") as merged:
            for idx, item in enumerate(failed_to_retry, 1):
                title = item["title"]
                doc_url = item["doc_url"]
                safe = sanitize_filename(title)
                doc_path = f"{doc_dir}/{safe}.doc"
                txt_path = f"{txt_dir}/{safe}.txt"

                try:
                    print(f"Téléchargement: {title}")
                    download_doc(doc_url, doc_path)

                    print(f"Conversion en TXT: {title}")
                    convert_doc_to_txt(doc_path, txt_path)

                    with open(txt_path, "r", encoding="utf-8", errors="replace") as tf:
                        content = tf.read().strip()

                    merged.write(f"=== {title} ===\n")
                    merged.write(content)
                    merged.write("\n" + "="*80 + "\n\n")
                    merged.flush()

                    if title in failed:
                        failed.remove(title)
                    completed.add(title)
                    if title in progress.get("failed", []):
                        progress["failed"].remove(title)
                    if title not in progress.get("completed", []):
                        progress["completed"].append(title)

                    try:
                        book_index = next(i for i, v in enumerate(livres) if v["title"] == title)
                        if book_index > progress.get("last_index", 0):
                            progress["last_index"] = book_index
                    except StopIteration:
                        pass

                    save_progress(progress_file, progress)
                    time.sleep(random.uniform(1, 2.5))
                except Exception as e:
                    print(f"Échec du retraitement '{title}': {e}")
                    time.sleep(4)

    # Livres restants
    remaining = [l for l in livres if l["title"] not in completed and l["title"] not in failed]
    print("\n=== Traitement des livres restants ===")
    print(f"Livres restants à traiter: {len(remaining)}")

    if len(remaining) > 0:
        resp = input(f"Continuer avec les {len(remaining)} livres restants? (y/n): ")
        if resp.lower() != "y":
            print("Arrêt du script.")
            return

    with open(merged_output, "a", encoding="utf-8") as merged:
        for idx, item in enumerate(remaining, 1):
            title = item["title"]
            doc_url = item["doc_url"]
            safe = sanitize_filename(title)
            doc_path = f"{doc_dir}/{safe}.doc"
            txt_path = f"{txt_dir}/{safe}.txt"

            try:
                print(f"[{idx}/{len(remaining)}] Téléchargement: {title}")
                download_doc(doc_url, doc_path)

                print(f"Conversion en TXT: {title}")
                convert_doc_to_txt(doc_path, txt_path)

                with open(txt_path, "r", encoding="utf-8", errors="replace") as tf:
                    content = tf.read().strip()

                merged.write(f"=== {title} ===\n")
                merged.write(content)
                merged.write("\n" + "="*80 + "\n\n")
                merged.flush()

                completed.add(title)
                progress.setdefault("completed", []).append(title)

                try:
                    book_index = next(i for i, v in enumerate(livres) if v["title"] == title)
                    if book_index > progress.get("last_index", 0):
                        progress["last_index"] = book_index
                except StopIteration:
                    pass

                pourcentage = ((last_index + idx) / len(livres)) * 100
                print(f"Livre '{title}' sauvegardé. ({pourcentage:.2f}% terminé / {last_index + idx}/{len(livres)})")

                save_progress(progress_file, progress)
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                pourcentage = ((last_index + idx) / len(livres)) * 100
                print(f"Erreur pour '{title}': {e} ({pourcentage:.2f}% terminé / {last_index + idx}/{len(livres)})")
                failed.add(title)
                progress.setdefault("failed", []).append(title)
                save_progress(progress_file, progress)
                time.sleep(5)

    print("\nScraping terminé!")
    print(f"Livres réussis: {len(completed)}")
    print(f"Livres en échec: {len(failed)}")
    print(f"Dernier index: {progress.get('last_index', 0)}")
    print(f"Fusion: {merged_output}")

if __name__ == "__main__":
    main()