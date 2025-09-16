# IA_NAH

## Présentation

**IA_NAH** est un projet Python pour l'entraînement d'un IA en langue malagasy : "IA_NAH". Elle est dédié à la collecte, au nettoyage et à l’entraînement de modèles de langage sur des textes en malgache. Il inclut :
- Le scraping de textes bibliques et d’articles wiki en malgache.
- Le nettoyage et la normalisation des textes.
- L’entraînement d’un modèle de génération de texte basé sur PyTorch.

## Structure du projet

```
Cleaning/         Scripts de nettoyage et de normalisation des textes
Scrapping/        Scripts de scraping, données brutes et résultats intermédiaires
Train/            Scripts et données pour l’entraînement du modèle de langage
```

## Fonctionnalités principales

- **Scrapping** : Extraction automatisée de textes bibliques et d’articles wiki en malgache.
- **Nettoyage** : Normalisation linguistique, conversion des nombres en ordinaux, suppression des incohérences.
- **Entraînement** : Modèle de langage de type Bigram/Transformer entraîné sur corpus malgache.
- **Gestion de la progression** : Sauvegarde de l’état d’avancement du scraping et de l’entraînement.

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone <url_du_repo>
   cd IA_NAH
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurer les chemins**  
   Adapter les chemins dans les scripts si besoin (par défaut, les chemins sont relatifs à la structure du projet).

## Utilisation

### 1. Scraping des textes

- Lancer le scraping de la Bible :
  ```bash
  python Scrapping/scrapping-baiboly.py
  ```
- Les textes extraits sont sauvegardés dans `Scrapping/out-put/baiboly_malagasy.txt`.

### 2. Nettoyage des textes

- Exemple de nettoyage :
  ```bash
  python Cleaning/cleanning_ch_txt.py
  ```
- Le texte nettoyé est sauvegardé dans le même fichier ou un nouveau fichier selon le script.

### 3. Entraînement du modèle

- Lancer l’entraînement :
  ```bash
  python Train/train_malagasy.py
  ```
- Les checkpoints et le modèle final sont sauvegardés dans `Train/out-put/`.

## Exemples de scripts

- `Scrapping/scrapping-baiboly.py` : Scraping de la Bible malgache.
- `Cleaning/cleanning_ch_txt.py` : Nettoyage et conversion des nombres en ordinaux.
- `Train/train_malagasy.py` : Entraînement du modèle de génération de texte.

## Dépendances principales

- `requests`, `beautifulsoup4` : Pour le scraping web.
- `torch`, `pandas` : Pour l’entraînement du modèle.
- `python >= 3.8`

## Auteurs

- [Ton nom ou pseudo]

## Licence

Ce projet est sous licence MIT.
