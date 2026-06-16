Parfait, voici un plan d'optimisation complet sur les 3 axes :

---

## 🧠 1. Améliorer le modèle Transformer

### Architecture — points à vérifier/upgrader

```python
# Configurations recommandées pour votre corpus malgache
config = {
    "vocab_size":   8000,    # BPE tokenizer sur malgache
    "n_embd":       256,     # Embedding size (léger pour M2 8Go)
    "n_head":       8,       # Attention heads
    "n_layer":      6,       # Transformer blocks
    "block_size":   256,     # Context window
    "dropout":      0.1,
}
```

### Améliorations clés à implémenter

**a) Remplacer le Bigram par un vrai tokenizer BPE**
```python
# Installer SentencePiece (idéal pour le malgache)
pip install sentencepiece

import sentencepiece as spm

# Entraîner sur votre corpus
spm.SentencePieceTrainer.train(
    input='Scrapping/out-put/baiboly_malagasy.txt',
    model_prefix='Train/malagasy_bpe',
    vocab_size=8000,
    character_coverage=0.9999,  # Important pour les accents malgaches
    model_type='bpe'
)
```

**b) Ajouter RoPE (Rotary Position Embedding)**
```python
# Meilleure gestion du contexte que le positional embedding classique
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)
```

**c) Learning Rate Scheduler cosine**
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

# Dans la boucle d'entraînement
scheduler.step()
```

---

## 📚 2. Plus de données malgaches

Votre corpus actuel (Bible + Wiki) est un bon début mais insuffisant. Voici des sources à scraper :

### Sources à ajouter

| Source | Contenu | Volume estimé |
|---|---|---|
| **Wikipedia malgache** `mg.wikipedia.org` | Articles généraux | ~50k articles |
| **Teny Malagasy** `tenymalagasy.mg` | Dictionnaire, grammaire | Moyen |
| **Radio Antsiva / MBS** | Articles news | Bon |
| **Bible + autres livres religieux** | Déjà fait | ✅ |
| **Common Crawl (malgache filtré)** | Web crawl général | Très grand |

### Script scraper Wikipedia malgache

```python
# Scrapping/scrapping_wiki_mg.py
import requests
from bs4 import BeautifulSoup
import time, json

BASE_URL = "https://mg.wikipedia.org"

def get_all_article_links(limit=500):
    url = f"{BASE_URL}/w/api.php"
    params = {
        "action": "query",
        "list": "allpages",
        "aplimit": 500,
        "format": "json"
    }
    response = requests.get(url, params=params).json()
    pages = response["query"]["allpages"]
    return [p["title"] for p in pages]

def scrape_article(title):
    url = f"{BASE_URL}/wiki/{title.replace(' ', '_')}"
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if content:
        return " ".join(p.get_text() for p in content.find_all("p"))
    return ""

titles = get_all_article_links()
with open("Scrapping/out-put/wiki_malagasy.txt", "w") as f:
    for title in titles:
        text = scrape_article(title)
        if len(text) > 100:
            f.write(text + "\n")
        time.sleep(0.5)  # Respecter le serveur
```

### Common Crawl — corpus malgache filtré
```bash
# Utiliser le dataset HuggingFace déjà filtré
pip install datasets

python -c "
from datasets import load_dataset
ds = load_dataset('cc100', lang='mg', split='train', streaming=True)
with open('Scrapping/out-put/cc100_malagasy.txt', 'w') as f:
    for i, row in enumerate(ds):
        f.write(row['text'] + '\n')
        if i > 50000: break  # Adapter selon votre espace disque
"
```

---

## ⚡ 3. Optimiser pour Apple M2 (MPS)

C'est le point **le plus impactant** — PyTorch sur MPS peut être 3x plus rapide que CPU sur M2.

### Activer MPS dans votre script d'entraînement

```python
# Train/train_malagasy.py — remplacer votre device setup

import torch

def get_device():
    if torch.backends.mps.is_available():
        print("✅ Apple MPS activé (GPU M2)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ CUDA activé")
        return torch.device("cuda")
    else:
        print("⚠️  CPU uniquement")
        return torch.device("cpu")

device = get_device()
model = model.to(device)
```

### Optimisations spécifiques MPS

```python
# 1. Batch size adapté à 8 Go RAM unifiée
BATCH_SIZE = 32       # Pas plus sur 8 Go
BLOCK_SIZE = 256      # Context window raisonnable

# 2. Mixed precision (économise la mémoire)
from torch.amp import autocast

with autocast(device_type='cpu'):  # MPS utilise 'cpu' pour autocast
    logits, loss = model(x, y)

# 3. Gradient accumulation (simule un grand batch)
ACCUMULATION_STEPS = 4  # Simule batch_size * 4

optimizer.zero_grad()
for i, (x, y) in enumerate(dataloader):
    logits, loss = model(x, y)
    loss = loss / ACCUMULATION_STEPS
    loss.backward()
    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Alternative — Passer à MLX (Apple natif)

```bash
pip install mlx mlx-lm
```

```python
# MLX est souvent 2x plus rapide que PyTorch/MPS sur Apple Silicon
import mlx.core as mx
import mlx.nn as nn

# MLX utilise le GPU M2 nativement sans configuration
# Votre logique PyTorch est facilement portable
```

---

## 🗺️ Roadmap suggérée

```
Étape 1 (cette semaine)
  └── Activer MPS dans train_malagasy.py
  └── Mesurer le gain de vitesse

Étape 2 (semaine 2)
  └── Scraper Wikipedia malgache + CC100
  └── Nettoyer et fusionner les corpus

Étape 3 (semaine 3)
  └── Remplacer tokenizer par SentencePiece BPE
  └── Ré-entraîner avec le nouveau corpus

Étape 4 (semaine 4)
  └── Implémenter RoPE + cosine scheduler
  └── Fine-tuner et évaluer (perplexité)
```

Voulez-vous que je vous aide à adapter un de ces scripts directement à votre code existant ? Partagez votre `train_malagasy.py` et je l'optimise pour vous ! 🚀