import requests
from bs4 import BeautifulSoup
import time
import random
import os
import json
from config import SCRAPPING_DIR, SCRAPPING_DATA, SCRAPPING_CTRL, SCRAPPING_OUTPUT

def scrape_wikipedia_article(title: str) -> str:
    """
    Récupère le texte d'un article Wikipédia en malagasy.
    """
    base_url = "https://mg.wikipedia.org/wiki/"
    url = base_url + title.replace(" ", "_")

    # Headers pour simuler un navigateur réel
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    response = requests.get(url, headers=headers)
    #print(f"Status code: {response.status_code}")
    if response.status_code != 200:
        raise Exception(f"Erreur {response.status_code} lors du chargement de la page.")

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Récupérer les paragraphes principaux
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
        titles = [line.strip() for line in f if line.strip()]
    return titles

def load_progress(progress_file: str) -> dict:
    """Charge le fichier de progression s'il existe"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"completed": [], "failed": [], "last_index": 0}
    return {"completed": [], "failed": [], "last_index": 0}

def save_progress(progress_file: str, progress: dict):
    """Sauvegarde la progression"""
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def check_article_exists(output_file: str, article_title: str) -> bool:
    """Vérifie si un article existe déjà dans le fichier de sortie"""
    if not os.path.exists(output_file):
        return False
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Recherche simple du titre dans le contenu
            # On vérifie si l'article est précédé de 'Ny' et suivi de 'dia' dans le fichier de sortie
            #motif = f"Ny {article_title} dia"
            motif = f"=== {article_title} ==="
            return motif in content
    except:
        return False

def main():
    # Exemple d'utilisation
    # Exemple d'utilisation
    articles = load_titles(f"{SCRAPPING_DATA}/mgwiki-latest-all-titles-in-ns0-non-traites.txt")
    # articles = ["","Madagasikara", "Fiteny_malagasy", "Andry_Rajoelina"]  # Exemples d'articles
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(SCRAPPING_OUTPUT, exist_ok=True)
    
    output_file = f"{SCRAPPING_OUTPUT}/articles_wiki.txt"
    progress_file = f"{SCRAPPING_CTRL}/scraping_progress_not_trated.json"
    
    # Charger la progression existante
    progress = load_progress(progress_file)
    completed_articles = set(progress["completed"])
    failed_articles = set(progress["failed"])
    last_index = progress["last_index"]
    
    total_articles = len(articles)
    print(f"Total d'articles à traiter: {total_articles}")
    print(f"Articles déjà complétés: {len(completed_articles)}")
    print(f"Articles en échec: {len(failed_articles)}")
    
    # Filtrer les articles déjà traités (sauf les échoués)
    articles_to_process = [article for article in articles if article not in completed_articles]
    print(f"Articles restants à traiter: {len(articles_to_process)}")
    
    # Traiter d'abord les articles en échec
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
                    f.flush()  # Forcer l'écriture immédiate
                    
                    print(f"Article {article} retraité avec succès!")
                    
                    # Retirer de la liste des échecs et ajouter aux complétés
                    failed_articles.remove(article)
                    completed_articles.add(article)
                    progress["failed"].remove(article)
                    progress["completed"].append(article)
                    
                    # Mettre à jour le last_index si nécessaire
                    try:
                        article_index = articles.index(article)
                        if article_index > progress["last_index"]:
                            progress["last_index"] = article_index
                    except ValueError:
                        pass
                    
                    save_progress(progress_file, progress)
                    
                    # Pause aléatoire entre les requêtes
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    print(f"Échec du retraitement pour l'article '{article}': {e}")
                    # L'article reste dans la liste des échecs
                    time.sleep(5)
    
    # Maintenant traiter les articles restants
    remaining_articles = [article for article in articles_to_process if article not in failed_articles]
    print(f"\n=== Traitement des articles restants ===")
    print(f"Articles restants à traiter: {len(remaining_articles)}")
    
    # Demander confirmation pour continuer
    if len(remaining_articles) > 0:
        response = input(f"Continuer avec les {len(remaining_articles)} articles restants? (y/n): ")
        if response.lower() != 'y':
            print("Arrêt du script.")
            exit()
    
    with open(output_file, "a", encoding="utf-8") as f:
        for idx, article in enumerate(remaining_articles, 1):
            if not article.strip():
                print(f"Titre vide, article ignoré. ({last_index + idx}/{len(articles)})")
                continue
                
            # Vérifier si l'article existe déjà dans le fichier
            if check_article_exists(output_file, article):
                print(f"Article '{article}' déjà présent dans le fichier, ignoré. ({last_index + idx}/{len(articles)})")
                completed_articles.add(article)
                progress["completed"].append(article)
                
                # Mettre à jour le last_index
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
                f.flush()  # Forcer l'écriture immédiate
                
                pourcentage = ((last_index + idx) / len(articles)) * 100
                print(f"Article {article} sauvegardé. ({pourcentage:.2f}% terminé / {last_index + idx}/{len(articles)})")
                
                # Mettre à jour la progression
                completed_articles.add(article)
                progress["completed"].append(article)
                
                # Mettre à jour le last_index
                try:
                    article_index = articles.index(article)
                    if article_index > progress["last_index"]:
                        progress["last_index"] = article_index
                except ValueError:
                    pass
                
                save_progress(progress_file, progress)
                
                # Pause aléatoire entre les requêtes pour éviter le rate limiting
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                pourcentage = ((last_index + idx) / len(articles)) * 100
                print(f"Erreur pour l'article '{article}': {e} ({pourcentage:.2f}% terminé / {last_index + idx}/{len(articles)})")
                
                # Mettre à jour la progression avec l'échec
                failed_articles.add(article)
                progress["failed"].append(article)
                save_progress(progress_file, progress)
                
                # Pause plus longue en cas d'erreur
                time.sleep(5)
    
    print(f"\nScraping terminé!")
    print(f"Articles réussis: {len(completed_articles)}")
    print(f"Articles en échec: {len(failed_articles)}")
    print(f"Dernier index: {progress['last_index']}")
    print(f"Progression sauvegardée dans: {progress_file}")


# Exemple d'utilisation
if __name__ == "__main__":
    main()