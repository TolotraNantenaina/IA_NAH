import os
import json
# Trouver les articles composés uniquement de 5 chiffres avant la virgule
import re

mois_mg = [
    "Janoary", "Febroary", "Martsa", "Aprily", "Mey", "Jona",
    "Jolay", "Aogositra", "Septambra", "Oktobra", "Novambra", "Desambra"
]

def load_titles(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip()]
    return titles

def main():
    # Exemple d'utilisation
    directory = "D:/Scrapping"
    articles = load_titles(f"{directory}/mgwiki-latest-all-titles-in-ns0")

    articles_not_trated = []
    # Nombre entier strictement compris entre 1 et 99999
    motif = re.compile(r"^(?:[1-9]\d{0,4})$")    

    # Si l'article doit être exactement du type "3_janoary"
    motif_numerique_exact = re.compile(r"^\d_(?:" + "|".join(mois_mg) + r")$", re.IGNORECASE)

    for article in articles:
        if motif_numerique_exact.match(article) or motif.match(article):
            articles_not_trated.append(article)

    # Écrire les articles non traités dans un fichier
    output_file = os.path.join(directory, "mgwiki-latest-all-titles-in-ns0-non-traites.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for art in articles_not_trated:
            f.write(art + "\n")

    print(f"{len(articles_not_trated)} articles non traités ont été sauvegardés dans {output_file}")

if __name__ == "__main__":
    main()