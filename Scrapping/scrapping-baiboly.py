import requests
from bs4 import BeautifulSoup
import time
import random
import os
import json
from urllib.parse import unquote

BASE_URL = "http://baiboly.katolika.org/boky/"

# Headers pour simuler un navigateur réel
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

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

def check_livre_exists(output_file: str, livre_title: str) -> bool:
    """Vérifie si un livre existe déjà dans le fichier de sortie"""
    if not os.path.exists(output_file):
        return False
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Recherche simple du titre dans le contenu
            return f"=== {livre_title} ===" in content
    except:
        return False

def get_liste_livres():
    url = BASE_URL
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    boky_index = soup.find("div", {"class": "boky_index"})
    if not boky_index:
        return []

    livres = []
    for li in boky_index.find_all("li"):
        a_tag = li.find("a")
        if a_tag and a_tag.get("href"):
            livre = a_tag.get("href").split("/")[-1]
            # Décoder les caractères URL (espaces, accents, etc.)
            livre = unquote(livre)
            livres.append(livre)

    return livres

def get_chapitre_text(livre, chapitre):
    url = f"{BASE_URL}{livre}/{chapitre}"

    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    #contenu = soup.find("div", {"id": "boky"})
    #if not contenu:
    #    return None

    # Chercher le titre du chapitre
    titre_chapitre = soup.find("h3", class_="toko")
    titre = titre_chapitre.get_text(strip=True) if titre_chapitre else None

    if not titre:
        return None
    
    # Chercher tous les versets (sup) dans le document
    versets = soup.find_all("sup")
    
    texte = f"{titre}\n"
    
    for verset in versets:
        # Récupérer le numéro du verset
        numero_verset = verset.get_text(strip=True)
        
        # Récupérer le texte du verset (le texte qui suit le sup)
        texte_verset = ""
        element_suivant = verset.next_sibling
        
        # Parcourir les éléments suivants jusqu'au prochain sup ou fin
        while element_suivant and element_suivant.name != "sup":
            if element_suivant.name is None:  # Texte simple
                texte_verset += element_suivant.strip()
            element_suivant = element_suivant.next_sibling
        
        # Nettoyer et ajouter le verset
        texte_verset = texte_verset.strip()
        if texte_verset:
            texte += f" {numero_verset} {texte_verset}\n"
    
    return texte.strip()

def scrape_livre(livre):
    chapitre = 1
    texte_complet = ""

    while True:
        print(f"Téléchargement {livre} chapitre {chapitre}...")
        texte = get_chapitre_text(livre, chapitre)

        if not texte:
            break

        texte_complet += texte + "\n\n"
        chapitre += 1
        time.sleep(random.uniform(0.5, 1.5))  # Pause aléatoire pour être gentil avec le serveur

    return texte_complet

def main():
    # Configuration des chemins
    directory = "D:/Scrapping"
    output_directory = f"{directory}/out-put"
    ctrl_directory = f"{directory}/ctrl"
    output_file = f"{output_directory}/baiboly_malagasy.txt"
    progress_file = f"{ctrl_directory}/baiboly_progress.json"
    
    # Créer les dossiers s'ils n'existent pas
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(ctrl_directory, exist_ok=True)
    
    # Charger la progression existante
    progress = load_progress(progress_file)
    completed_livres = set(progress["completed"])
    failed_livres = set(progress["failed"])
    last_index = progress["last_index"]
    
    # Récupérer la liste des livres
    livres = get_liste_livres()
    if not livres:
        print("Erreur: Impossible de récupérer la liste des livres.")
        return
    
    total_livres = len(livres)
    print(f"Total de livres à traiter: {total_livres}")
    print(f"Livres déjà complétés: {len(completed_livres)}")
    print(f"Livres en échec: {len(failed_livres)}")
    
    # Traiter d'abord les livres en échec
    failed_to_retry = list(failed_livres)
    if failed_to_retry:
        print(f"\n=== Retraitement des livres en échec ===")
        print(f"Livres en échec à retraiter: {len(failed_to_retry)}")
        
        with open(output_file, "a", encoding="utf-8") as f:
            for idx, livre in enumerate(failed_to_retry, 1):
                print(f"Retraitement du livre échoué: {livre} ({idx}/{len(failed_to_retry)})")
                
                try:
                    texte_livre = scrape_livre(livre)
                    f.write(f"=== {livre} ===\n")
                    f.write(texte_livre)
                    f.write("\n" + "="*80 + "\n\n")
                    f.flush()  # Forcer l'écriture immédiate
                    
                    print(f"Livre {livre} retraité avec succès!")
                    
                    # Retirer de la liste des échecs et ajouter aux complétés
                    failed_livres.remove(livre)
                    completed_livres.add(livre)
                    progress["failed"].remove(livre)
                    progress["completed"].append(livre)
                    
                    # Mettre à jour le last_index si nécessaire
                    try:
                        livre_index = livres.index(livre)
                        if livre_index > progress["last_index"]:
                            progress["last_index"] = livre_index
                    except ValueError:
                        pass
                    
                    save_progress(progress_file, progress)
                    
                    # Pause aléatoire entre les requêtes
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    print(f"Échec du retraitement pour le livre '{livre}': {e}")
                    # Le livre reste dans la liste des échecs
                    time.sleep(5)
    
    # Maintenant traiter les livres restants
    remaining_livres = [livre for livre in livres if livre not in completed_livres and livre not in failed_livres]
    print(f"\n=== Traitement des livres restants ===")
    print(f"Livres restants à traiter: {len(remaining_livres)}")
    
    # Demander confirmation pour continuer
    if len(remaining_livres) > 0:
        response = input(f"Continuer avec les {len(remaining_livres)} livres restants? (y/n): ")
        if response.lower() != 'y':
            print("Arrêt du script.")
            exit()
    
    with open(output_file, "a", encoding="utf-8") as f:
        for idx, livre in enumerate(remaining_livres, 1):
            # Vérifier si le livre existe déjà dans le fichier
            if check_livre_exists(output_file, livre):
                print(f"Livre '{livre}' déjà présent dans le fichier, ignoré. ({last_index + idx}/{len(livres)})")
                completed_livres.add(livre)
                progress["completed"].append(livre)
                
                # Mettre à jour le last_index
                try:
                    livre_index = livres.index(livre)
                    if livre_index > progress["last_index"]:
                        progress["last_index"] = livre_index
                except ValueError:
                    pass
                
                save_progress(progress_file, progress)
                continue
                
            try:
                print(f"Traitement du livre: {livre} ({idx}/{len(remaining_livres)})")
                texte_livre = scrape_livre(livre)
                
                f.write(f"=== {livre} ===\n")
                f.write(texte_livre)
                f.write("\n" + "="*80 + "\n\n")
                f.flush()  # Forcer l'écriture immédiate
                
                pourcentage = ((last_index + idx) / len(livres)) * 100
                print(f"Livre {livre} sauvegardé. ({pourcentage:.2f}% terminé / {last_index + idx}/{len(livres)})")
                
                # Mettre à jour la progression
                completed_livres.add(livre)
                progress["completed"].append(livre)
                
                # Mettre à jour le last_index
                try:
                    livre_index = livres.index(livre)
                    if livre_index > progress["last_index"]:
                        progress["last_index"] = livre_index
                except ValueError:
                    pass
                
                save_progress(progress_file, progress)
                
                # Pause aléatoire entre les requêtes
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                pourcentage = ((last_index + idx) / len(livres)) * 100
                print(f"Erreur pour le livre '{livre}': {e} ({pourcentage:.2f}% terminé / {last_index + idx}/{len(livres)})")
                
                # Mettre à jour la progression avec l'échec
                failed_livres.add(livre)
                progress["failed"].append(livre)
                save_progress(progress_file, progress)
                
                # Pause plus longue en cas d'erreur
                time.sleep(5)
    
    print(f"\nScraping terminé!")
    print(f"Livres réussis: {len(completed_livres)}")
    print(f"Livres en échec: {len(failed_livres)}")
    print(f"Dernier index: {progress['last_index']}")
    print(f"Progression sauvegardée dans: {progress_file}")

if __name__ == "__main__":
    main()
