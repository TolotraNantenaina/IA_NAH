"""
Fine-tuning du modèle malagasy sur des discours conversationnels.

Le corpus de fine-tuning utilise le fichier fusionné :
  - Train/data/discours_merged.txt  (généré par merge_discours.py)
    contient : dialogue Rakoto/Rasoa + 49 kabary/proverbes JSON + 219 Q&A CSV

Usage :
    python Train/finetune_discours.py
    python Train/finetune_discours.py --pretrained Train/out-put/mon_modele.pth --epochs 100 --lr 5e-5
    python Train/finetune_discours.py --interactive   (génération après fine-tuning)
"""

import argparse
import math
import os
import sys
from contextlib import nullcontext

import torch
from torch.nn.utils import clip_grad_norm_

from model import BigramLanguageModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_DATA, TRAIN_DIR, TRAIN_OUTPUT

DISCOURS_MERGED = os.path.join(TRAIN_DATA, "discours_merged.txt")
# Fallback sur les fichiers séparés si le fichier fusionné n'existe pas
DISCOURS_FILES = [
    os.path.join(TRAIN_DATA, "discours_malagasy.txt"),
    os.path.join(TRAIN_DATA, "discours_ex_malagasy.txt"),
]

DEFAULT_PRETRAINED = os.path.join(TRAIN_OUTPUT, "modele_malagasy.pth")
DEFAULT_OUTPUT = os.path.join(TRAIN_OUTPUT, "modele_discours.pth")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(device_arg="auto"):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def amp_context(device, use_amp):
    if use_amp and device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.bfloat16)
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

def load_discourse_corpus():
    """Charge le corpus de discours fusionné, ou les fichiers séparés en fallback."""
    if os.path.isfile(DISCOURS_MERGED):
        with open(DISCOURS_MERGED, "r", encoding="utf-8") as f:
            content = f.read().strip()
        print(f"  discours_merged.txt : {len(content):,} caractères")
        return content
    # Fallback : ancienne méthode avec fichiers séparés
    print("Avertissement : discours_merged.txt introuvable, utilisation des fichiers séparés.")
    parts = []
    for path in DISCOURS_FILES:
        if not os.path.isfile(path):
            print(f"Fichier introuvable (ignoré) : {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        parts.append(content)
        print(f"  {os.path.basename(path)} : {len(content):,} caractères")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Modèle pré-entraîné
# ---------------------------------------------------------------------------

def load_pretrained(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Modèle introuvable : {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = BigramLanguageModel(
        vocab_size=ckpt["vocab_size"],
        n_embd=ckpt.get("n_embd", 256),
        n_head=ckpt.get("n_head", 8),
        n_layer=ckpt.get("n_layer", 6),
        block_size=ckpt.get("block_size", 256),
        dropout=ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def build_tokenizer(ckpt):
    """Reconstruit l'encodeur/décodeur à partir des métadonnées du checkpoint."""
    tokenizer_type = ckpt.get("tokenizer_type", "char")
    stoi = ckpt.get("stoi")
    itos = ckpt.get("itos")
    spm_model_path = ckpt.get("spm_model_path")

    if tokenizer_type == "bpe":
        import sentencepiece as spm

        # Cherche le fichier .model si le chemin absolu n'est plus valide
        if not spm_model_path or not os.path.isfile(spm_model_path):
            candidates = [
                os.path.join(TRAIN_DIR, "malagasy_bpe.model"),
                os.path.join(TRAIN_OUTPUT, "malagasy_bpe.model"),
            ]
            spm_model_path = next((c for c in candidates if os.path.isfile(c)), None)
        if not spm_model_path:
            raise FileNotFoundError(
                "Modèle BPE introuvable. Entraînez d'abord le modèle principal."
            )
        sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        encode = lambda s: sp.encode(s, out_type=int)
        decode = lambda ids: sp.decode(ids)
    else:
        if stoi is None or itos is None:
            raise ValueError("Checkpoint char sans vocabulaire stoi/itos.")
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda ids: "".join(itos[i] for i in ids)

    return encode, decode, tokenizer_type, spm_model_path, stoi, itos


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def build_batches(tokens, block_size, batch_size):
    """
    Découpe le tenseur de tokens en séquences non-chevauchantes,
    puis regroupe en batchs.
    """
    seqs = []
    for i in range(0, len(tokens) - block_size, block_size):
        x = tokens[i: i + block_size]
        y = tokens[i + 1: i + block_size + 1]
        seqs.append((x, y))

    batches = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i: i + batch_size]
        batches.append(
            (torch.stack([c[0] for c in chunk]), torch.stack([c[1] for c in chunk]))
        )
    return batches


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step, total_steps, peak_lr, eta_min, warmup_ratio=0.1):
    """Warmup linéaire (warmup_ratio * total_steps) puis cosine vers eta_min."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return eta_min + (peak_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(model, tokens, block_size, batch_size, epochs, learning_rate,
             eta_min, max_grad_norm, device, use_amp):
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Modèle : {n_params:.1f}M paramètres | device : {device}")

    batches = build_batches(tokens, block_size, batch_size)
    if not batches:
        raise ValueError(
            f"Corpus trop court pour block_size={block_size}. "
            "Réduisez --block-size ou enrichissez les données de discours."
        )

    total_steps = epochs * len(batches)
    print(f"{len(batches)} batch(es)/époque × {epochs} époques = {total_steps} steps")
    print(f"LR peak={learning_rate} → min={eta_min} (cosine + warmup 10 %)")

    # Seuls les paramètres des couches attention/FFN sont mis à jour
    # (les embeddings sont figés pour préserver le vocabulaire)
    trainable = [
        {"params": model.blocks.parameters(), "lr": learning_rate},
        {"params": model.ln_f.parameters(), "lr": learning_rate},
        {"params": model.lm_head.parameters(), "lr": learning_rate * 0.1},
    ]
    optimizer = torch.optim.AdamW(trainable, weight_decay=0.01)

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xs, ys in batches:
            xs, ys = xs.to(device), ys.to(device)

            lr = get_lr(global_step, total_steps, learning_rate, eta_min)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, use_amp):
                _, loss = model(xs, ys)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

        avg = epoch_loss / len(batches)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"  Époque {epoch:>4}/{epochs} | loss : {avg:.4f} | lr : {lr:.2e}")

    print("Fine-tuning terminé.")
    return model


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_model(model, ckpt, output_path, tokenizer_type, spm_model_path, stoi, itos):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "vocab_size": model.vocab_size,
        "n_embd": ckpt.get("n_embd", 256),
        "n_head": ckpt.get("n_head", 8),
        "n_layer": ckpt.get("n_layer", 6),
        "block_size": ckpt.get("block_size", 256),
        "dropout": ckpt.get("dropout", 0.1),
        "tokenizer_type": tokenizer_type,
        "finetuned_on": "discours_malagasy",
    }
    if tokenizer_type == "char" and stoi is not None:
        payload["stoi"] = stoi
        payload["itos"] = itos
    if spm_model_path:
        payload["spm_model_path"] = os.path.abspath(spm_model_path)
    torch.save(payload, output_path)
    print(f"Modèle sauvegardé : {output_path}")


# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------

def generate(model, encode, decode, prompt, max_new_tokens, temperature, top_k, device):
    model.eval()
    if prompt:
        ids = encode(prompt)
        ids = ids if ids else [0]
    else:
        ids = [0]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()
    return decode(out)


def interactive_mode(model, encode, decode, device):
    print("\nMode interactif — 'quit' pour quitter")
    print("Commandes : temp X | topk X | tokens X")
    temperature, top_k, max_tokens = 0.8, 40, 200
    while True:
        try:
            user_input = input("\nPrompt : ").strip()
            if user_input.lower() == "quit":
                break
            if user_input.startswith("temp "):
                temperature = float(user_input.split()[1])
                print(f"Température → {temperature}")
                continue
            if user_input.startswith("topk "):
                top_k = int(user_input.split()[1])
                print(f"Top-k → {top_k}")
                continue
            if user_input.startswith("tokens "):
                max_tokens = int(user_input.split()[1])
                print(f"Tokens → {max_tokens}")
                continue
            result = generate(model, encode, decode, user_input, max_tokens, temperature, top_k, device)
            print("=" * 50)
            print(result)
            print("=" * 50)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erreur : {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning du modèle malagasy sur des discours conversationnels"
    )
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED,
                        help="Chemin du modèle pré-entraîné (.pth)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Chemin de sauvegarde du modèle fine-tuné")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Nombre d'époques (défaut : 80)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate de pointe (défaut : 2e-5)")
    parser.add_argument("--eta-min", type=float, default=1e-7,
                        help="LR minimum cosine (défaut : 1e-7)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Taille de batch (défaut : 4)")
    parser.add_argument("--block-size", type=int, default=None,
                        help="Longueur de contexte (héritée du modèle si omis)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Clipping des gradients (défaut : 1.0)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--interactive", action="store_true",
                        help="Lance le mode interactif après le fine-tuning")
    parser.add_argument("--prompt", default="",
                        help="Prompt pour la génération (sans --interactive)")
    parser.add_argument("--tokens", type=int, default=200,
                        help="Tokens à générer (défaut : 200)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = get_device(args.device)
    use_amp = device.type in ("mps", "cuda")

    print("=" * 60)
    print("Fine-tuning sur discours malgaches")
    print("=" * 60)

    # 1. Charger le modèle pré-entraîné
    print(f"\nChargement : {args.pretrained}")
    model, ckpt = load_pretrained(args.pretrained, device)
    encode, decode, tokenizer_type, spm_model_path, stoi, itos = build_tokenizer(ckpt)
    block_size = args.block_size or ckpt.get("block_size", 256)
    print(f"Tokenizer : {tokenizer_type} | block_size : {block_size}")

    # 2. Charger le corpus de discours
    print("\nCorpus de discours :")
    corpus = load_discourse_corpus()
    if not corpus.strip():
        print("Aucun texte de discours trouvé. Vérifiez Train/data/.")
        return 1
    print(f"Total : {len(corpus):,} caractères")

    tokens = torch.tensor(encode(corpus), dtype=torch.long)
    print(f"Total : {len(tokens):,} tokens")

    if len(tokens) <= block_size:
        print(
            f"\nCorpus trop court ({len(tokens)} tokens ≤ block_size={block_size}).\n"
            f"Réduisez --block-size à {len(tokens) // 2} ou moins."
        )
        return 1

    # 3. Fine-tuning
    print(f"\nFine-tuning : {args.epochs} époques | lr={args.lr} | batch={args.batch_size}")
    print("-" * 40)
    model = finetune(
        model, tokens,
        block_size=block_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        eta_min=args.eta_min,
        max_grad_norm=args.max_grad_norm,
        device=device,
        use_amp=use_amp,
    )

    # 4. Sauvegarde
    save_model(model, ckpt, args.output, tokenizer_type, spm_model_path, stoi, itos)

    # 5. Génération
    if args.interactive:
        interactive_mode(model, encode, decode, device)
    elif args.prompt or True:
        print("\nExemple de génération :")
        print("=" * 50)
        sample = generate(
            model, encode, decode, args.prompt,
            args.tokens, args.temperature, args.top_k, device
        )
        print(sample)
        print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
