import json
import os
import sys

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

def load_articles_list(articles_file: str) -> list:
    """Charge la liste complète des articles"""
    try:
        with open(articles_file, "r", encoding="utf-8") as f:
            articles = [line.strip() for line in f if line.strip()]
        return articles
    except FileNotFoundError:
        print(f"Erreur: Le fichier {articles_file} n'existe pas.")
        return []

def update_progress_with_article(progress_file: str, articles_file: str, target_article: str, failed_article: str = None):
    """
    Met à jour le fichier de progression avec un article donné.
    
    Args:
        progress_file: Chemin vers le fichier de progression
        articles_file: Chemin vers le fichier contenant la liste des articles
        target_article: Article jusqu'auquel marquer comme complété
        failed_article: Article à marquer comme échoué (optionnel)
    """
    
    # Charger la liste complète des articles
    all_articles = load_articles_list(articles_file)
    if not all_articles:
        return
    
    # Charger la progression existante
    progress = load_progress(progress_file)
    
    # Trouver l'index de l'article cible
    try:
        target_index = all_articles.index(target_article)
        print(f"Article cible trouvé: '{target_article}' à l'index {target_index}")
    except ValueError:
        print(f"Erreur: L'article '{target_article}' n'a pas été trouvé dans la liste.")
        return
    
    # Mettre à jour les articles complétés
    articles_to_complete = all_articles[:target_index + 1]  # +1 pour inclure l'article cible
    
    # Filtrer les articles déjà complétés et échoués
    already_completed = set(progress["completed"])
    already_failed = set(progress["failed"])
    
    new_completed = []
    for article in articles_to_complete:
        if article not in already_completed and article not in already_failed:
            new_completed.append(article)
    
    # Ajouter les nouveaux articles complétés
    progress["completed"].extend(new_completed)
    
    # Mettre à jour l'index du dernier article
    progress["last_index"] = target_index
    
    # Gérer l'article échoué si spécifié
    if failed_article:
        if failed_article in progress["completed"]:
            progress["completed"].remove(failed_article)
        if failed_article not in progress["failed"]:
            progress["failed"].append(failed_article)
        print(f"Article marqué comme échoué: '{failed_article}'")
    
    # Sauvegarder la progression
    save_progress(progress_file, progress)
    
    # Afficher les statistiques
    print(f"\n=== Statistiques de mise à jour ===")
    print(f"Articles nouvellement marqués comme complétés: {len(new_completed)}")
    print(f"Total d'articles complétés: {len(progress['completed'])}")
    print(f"Total d'articles échoués: {len(progress['failed'])}")
    print(f"Dernier index: {progress['last_index']}")
    print(f"Progression sauvegardée dans: {progress_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python update_progress.py <article_target> [article_failed]")
        print("Exemple: python update_progress.py '(162425)_2000_EF136' '(134359)_1994_PP27'")
        return
    
    # Configuration des chemins
    directory = "D:/Scrapping"
    progress_file = f"{directory}/out-put/scraping_progress.json"
    articles_file = f"{directory}/mgwiki-latest-all-titles-in-ns0"
    
    # Paramètres
    target_article = sys.argv[1]
    failed_article = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Article cible: {target_article}")
    if failed_article:
        print(f"Article échoué: {failed_article}")
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    # Mettre à jour la progression
    update_progress_with_article(progress_file, articles_file, target_article, failed_article)

if __name__ == "__main__":
    main()
