# CLAUDE.md

Ce fichier fournit des indications à Claude Code (claude.ai/code) pour travailler dans ce dépôt.

## Présentation du projet

IA_NAH est un modèle de langage Transformer entraîné de zéro sur des textes malgaches collectés par scraping (Bible, Wikipédia, CC100). Le pipeline est : scraping → nettoyage → entraînement → inférence.

## Commandes

**Installation (première fois) :**
```bash
pip install -r requirements.txt
```

Ou via le script shell qui crée automatiquement un environnement virtuel :
```bash
./ia_nah.sh
```

**Entraîner le modèle :**
```bash
python main.py --mode train --model-name modele_malagasy.pth
# Ou directement :
python Train/train_malagasy.py
```

**Fine-tuning sur les discours malgaches :**
```bash
# Via main.py
python main.py --mode finetune --model-name modele_malagasy.pth --epochs 80 --lr 2e-5
# Directement avec plus d'options
python Train/finetune_discours.py --pretrained Train/out-put/modele_malagasy.pth --epochs 80
# Mode interactif après fine-tuning
python Train/finetune_discours.py --interactive
```

**Inférence / génération de texte :**
```bash
python main.py --mode test --prompt "Ny teny" --tokens 200 --temperature 1.0
# Mode interactif :
python main.py --mode test --interactive
# Ou directement avec plus d'options :
python Train/inference.py --model Train/out-put/modele_malagasy.pth --prompt "Ny teny" --top-k 40
```

**Scraping des sources de texte :**
```bash
python Scrapping/scrapping-baiboly.py          # Bible malgache
python Scrapping/scrapping_wiki_mg.py          # Wikipédia malgache
python Scrapping/fetch_cc100_mg.py             # Jeu de données CC100
```

**Nettoyage des textes collectés :**
```bash
python Cleaning/cleanning_ch_txt.py
```

## Architecture

### Configuration des chemins
`config.py` à la racine du projet définit tous les chemins de répertoires (`TRAIN_DIR`, `SCRAPPING_OUTPUT`, `TRAIN_OUTPUT`, `TRAIN_DATA`, etc.). Tous les scripts l'importent — ne jamais coder les chemins en dur.

### Points d'entrée
- `main.py` — CLI unifié qui délègue `--mode train` ou `--mode test` en appelant les scripts de `Train/` via subprocess. Transmet la variable d'environnement `MODEL_OUTPUT_PATH` pour personnaliser le chemin de sauvegarde du modèle.
- `ia_nah.sh` — crée un `venv/`, installe les dépendances et lance `Train/train_malagasy.py`.

### Modèle (`Train/model.py`)
`BigramLanguageModel` est un Transformer décodeur uniquement avec :
- `RotaryEmbedding` (RoPE) appliqué à l'intérieur de chaque tête `CausalSelfAttention` à la place des embeddings positionnels appris
- Normalisation avant les couches (*pre-norm*) : `LayerNorm` avant l'attention et le feedforward, pas après
- Activation `GELU` dans `FeedForward` (plus adaptée aux transformers que `ReLU`)
- Initialisation des poids via `_init_weights` (`normal(0, 0.02)` pour les linéaires et embeddings)
- Partage des poids entre `token_embedding_table` et `lm_head` (réduit les paramètres)
- Masque causal enregistré comme buffer, non recalculé à chaque passe
- `generate()` supporte la température et l'échantillonnage top-k

### Entraînement (`Train/train_malagasy.py`)
- Les hyperparamètres sont centralisés dans `get_hyperparameters()` — c'est là qu'il faut modifier l'architecture ou le planning d'apprentissage
- Schedule LR : warmup linéaire (`warmup_iters=400`) puis décroissance cosinus vers `eta_min` — implémenté manuellement dans `get_lr()`
- Gradient clipping : `clip_grad_norm_(model.parameters(), max_grad_norm)` après chaque backward
- Tokenisation : BPE via SentencePiece (`use_bpe=True` par défaut) — entraîne `malagasy_bpe.model` à la première exécution et le réutilise ensuite ; fallback en mode caractère
- Détection automatique du device : MPS (Apple Silicon) → CUDA → CPU
- Précision mixte (`torch.autocast`) activée sur MPS et CUDA
- Accumulation de gradients (`accumulation_steps=4`) ; batch effectif = `batch_size × accumulation_steps`
- Checkpoints sauvegardés dans `Train/out-put/checkpoints/checkpoint.pt` ; historique des pertes dans `loss_history.csv` dans le même dossier
- Modèle final sauvegardé dans `Train/out-put/modele_malagasy.pth` (ou via la variable `MODEL_OUTPUT_PATH`)

### Fine-tuning discours (`Train/finetune_discours.py`)
- Charge un modèle pré-entraîné (`--pretrained`), reconstruit le tokenizer depuis les métadonnées du checkpoint
- Concatène `Train/data/discours_malagasy.txt` et `Train/data/discours_ex_malagasy.txt`
- Stratégie adaptée aux petits corpus : entraînement par époques (passage complet sur toutes les données)
- Seuls les blocs attention/FFN et `ln_f` sont entraînés ; les embeddings sont figés pour préserver le vocabulaire
- LR très faible (défaut `2e-5`) avec warmup 10 % + cosine ; gradient clipping `1.0`
- Sauvegarde dans `Train/out-put/modele_discours.pth` (ou `--output`)
- Accepte `--interactive` pour tester directement après fine-tuning

### Inférence (`Train/inference.py`)
Charge un checkpoint `.pth`, reconstruit le tokenizer (chemin BPE stocké dans le checkpoint, avec recherche de secours près du fichier modèle) et génère du texte. Supporte le mode batch (`--prompt`) ou interactif (commandes à la volée : `temp X`, `topk X`, `tokens X`).

### Pipeline de données
1. **Scrapping/** — les scrapers écrivent le texte brut dans `Scrapping/out-put/` et suivent leur progression dans des fichiers JSON sous `Scrapping/ctrl/`
2. **Cleaning/** — `cleanning_ch_txt.py` normalise le texte et convertit les références numériques de chapitres en ordinaux malgaches (ex. `faha-3` → `fahatelo`)
3. **Train/data/** — corpus nettoyé consommé par l'entraînement ; fichier par défaut : `tantara_malagasy.txt`

### Fichiers du modèle BPE
`Train/malagasy_bpe.model` et `.vocab` sont des artefacts SentencePiece pré-entraînés. Le script d'entraînement ignore la phase BPE si le fichier `.model` existe déjà.
