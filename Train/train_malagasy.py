import csv
import math
import os
import sys
from contextlib import nullcontext

import torch
from torch.nn.utils import clip_grad_norm_

from model import BigramLanguageModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_DIR, TRAIN_OUTPUT, TRAIN_DATA


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_hyperparameters():
    return {
        "batch_size": 32,
        "block_size": 256,
        "max_iters": 20000,
        "eval_interval": 500,
        "learning_rate": 3e-4,
        "eta_min": 1e-5,
        "warmup_iters": 400,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "eval_iters": 200,
        "n_embd": 256,
        "n_head": 8,
        "n_layer": 6,
        "dropout": 0.1,
        "checkpoint_dir": None,
        "checkpoint_interval": 1000,
        "resume_from_checkpoint": True,
        "accumulation_steps": 4,
        "use_bpe": True,
        "bpe_vocab_size": 8000,
        "bpe_model_prefix": os.path.join(TRAIN_DIR, "malagasy_bpe"),
    }


def get_lr(step, warmup_iters, max_iters, peak_lr, eta_min):
    """Warmup linéaire puis décroissance cosinus."""
    if step < warmup_iters:
        return peak_lr * step / max(1, warmup_iters)
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    return eta_min + (peak_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_data(use_csv=False, txt_file=None, csv_file=None, csv_column="text"):
    if use_csv:
        parts = []
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parts.append(str(row.get(csv_column, "") or ""))
        text = " ".join(parts)
    else:
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
    print(f"Corpus chargé : {len(text):,} caractères")
    return text


def build_vocabulary(text):
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"Vocabulaire caractères : {vocab_size} symboles")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)
    return vocab_size, stoi, itos, encode, decode


def ensure_bpe_model(corpus_path, model_prefix, vocab_size):
    model_file = model_prefix + ".model"
    if os.path.isfile(model_file):
        return model_file
    import sentencepiece as spm
    print("Entraînement du tokenizer BPE...")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9999,
        model_type="bpe",
    )
    return model_file


def setup_tokenizer(hyperparams, txt_file):
    import sentencepiece as spm
    prefix = hyperparams["bpe_model_prefix"]
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    model_file = ensure_bpe_model(txt_file, prefix, hyperparams["bpe_vocab_size"])
    sp = spm.SentencePieceProcessor(model_file=model_file)
    vocab_size = sp.get_piece_size()
    encode = lambda s: sp.encode(s, out_type=int)
    decode = lambda ids: sp.decode(ids)
    print(f"Tokenizer BPE : vocab_size={vocab_size} | {model_file}")
    return vocab_size, encode, decode, None, None, model_file


def prepare_data(text, encode, train_split=0.9):
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_split * len(data))
    return data[:n], data[n:]


def get_batch(split, train_data, val_data, batch_size, block_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def amp_context(device, use_amp):
    if use_amp and device.type in ("mps", "cuda"):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters, use_amp):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, batch_size, block_size, device)
            with amp_context(device, use_amp):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_model(model, train_data, val_data, hyperparams):
    device = hyperparams["device"]
    batch_size = hyperparams["batch_size"]
    block_size = hyperparams["block_size"]
    max_iters = hyperparams["max_iters"]
    eval_interval = hyperparams["eval_interval"]
    peak_lr = hyperparams["learning_rate"]
    eta_min = hyperparams.get("eta_min", 1e-5)
    warmup_iters = hyperparams.get("warmup_iters", 400)
    weight_decay = hyperparams.get("weight_decay", 0.1)
    max_grad_norm = hyperparams.get("max_grad_norm", 1.0)
    eval_iters = hyperparams["eval_iters"]
    accum_steps = max(1, int(hyperparams.get("accumulation_steps", 4)))
    use_amp = hyperparams.get("use_amp", True) and device.type in ("mps", "cuda")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Modèle : {n_params:.1f}M paramètres | device : {device}")
    print(f"LR : {peak_lr} (warmup {warmup_iters} steps) -> {eta_min} (cosine)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, weight_decay=weight_decay
    )

    checkpoint_dir = hyperparams.get("checkpoint_dir") or "."
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    checkpoint_interval = int(hyperparams.get("checkpoint_interval", 0) or 0)
    resume = bool(hyperparams.get("resume_from_checkpoint", False))

    # Historique des pertes (CSV)
    loss_log_path = os.path.join(checkpoint_dir, "loss_history.csv")
    loss_log_exists = os.path.isfile(loss_log_path)
    loss_log = open(loss_log_path, "a", newline="", encoding="utf-8")
    loss_writer = csv.writer(loss_log)
    if not loss_log_exists:
        loss_writer.writerow(["step", "train_loss", "val_loss", "lr"])

    start_iter = 0
    if resume and os.path.isfile(checkpoint_path):
        try:
            print(f"Reprise depuis : {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_iter = int(ckpt.get("iter", 0)) + 1
            print(f"Reprise à l'itération {start_iter}")
        except Exception as e:
            print(f"Impossible de charger le checkpoint : {e}. Nouvel entraînement.")

    try:
        for step in range(start_iter, max_iters):
            # Mise à jour du learning rate
            lr = get_lr(step, warmup_iters, max_iters, peak_lr, eta_min)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Évaluation périodique
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(
                    model, train_data, val_data, batch_size, block_size, device, eval_iters, use_amp
                )
                print(
                    f"step {step:>6} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e}"
                )
                loss_writer.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.2e}"])
                loss_log.flush()

            # Passe forward + backward avec accumulation de gradients
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum_steps):
                xb, yb = get_batch("train", train_data, val_data, batch_size, block_size, device)
                with amp_context(device, use_amp):
                    _, loss = model(xb, yb)
                (loss / accum_steps).backward()

            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Sauvegarde du checkpoint
            if checkpoint_interval and step > 0 and step % checkpoint_interval == 0:
                try:
                    torch.save(
                        {
                            "iter": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "hyperparams": {k: v for k, v in hyperparams.items() if k != "device"},
                        },
                        checkpoint_path,
                    )
                    print(f"Checkpoint sauvegardé (step {step})")
                except Exception as e:
                    print(f"Échec sauvegarde checkpoint : {e}")
    finally:
        loss_log.close()

    print("Entraînement terminé.")
    return model


def generate_sample_text(model, decode, device, max_new_tokens=300):
    print("\nÉchantillon généré :")
    print("=" * 50)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        generated = decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
    print(generated)
    print("=" * 50)
    return generated


def save_model(model, stoi, itos, hyperparams, output_path, tokenizer_type, spm_model_path=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "vocab_size": model.vocab_size,
        "n_embd": hyperparams["n_embd"],
        "n_head": hyperparams["n_head"],
        "n_layer": hyperparams["n_layer"],
        "block_size": hyperparams["block_size"],
        "dropout": hyperparams["dropout"],
        "tokenizer_type": tokenizer_type,
    }
    if tokenizer_type == "char" and stoi is not None:
        payload["stoi"] = stoi
        payload["itos"] = itos
    if spm_model_path:
        payload["spm_model_path"] = os.path.abspath(spm_model_path)
    torch.save(payload, output_path)
    print(f"Modèle sauvegardé : {output_path}")


def main():
    print("Entraînement du modèle de langue malagasy")
    print("=" * 60)

    torch.manual_seed(1337)

    hyperparams = get_hyperparameters()
    device = get_device()
    hyperparams["device"] = device
    hyperparams["use_amp"] = True

    txt_file = os.path.join(TRAIN_DATA, "tantara_malagasy.txt")

    print("Chargement des données...")
    text = load_data(txt_file=txt_file)

    print("Tokenisation...")
    if hyperparams.get("use_bpe", True):
        vocab_size, encode, decode, stoi, itos, spm_model_path = setup_tokenizer(
            hyperparams, txt_file
        )
    else:
        vocab_size, stoi, itos, encode, decode = build_vocabulary(text)
        spm_model_path = None

    train_data, val_data = prepare_data(text, encode)
    print(f"Train : {len(train_data):,} tokens | Val : {len(val_data):,} tokens")

    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=hyperparams["n_embd"],
        n_head=hyperparams["n_head"],
        n_layer=hyperparams["n_layer"],
        block_size=hyperparams["block_size"],
        dropout=hyperparams["dropout"],
    ).to(device)

    hyperparams["checkpoint_dir"] = os.path.join(TRAIN_OUTPUT, "checkpoints")
    model = train_model(model, train_data, val_data, hyperparams)

    generate_sample_text(model, decode, device)

    model_path = os.environ.get("MODEL_OUTPUT_PATH") or os.path.join(
        TRAIN_OUTPUT, "modele_malagasy.pth"
    )
    tok_type = "bpe" if hyperparams.get("use_bpe", True) else "char"
    save_model(model, stoi, itos, hyperparams, model_path, tok_type, spm_model_path)
    print(f"\nModèle final : {model_path}")


if __name__ == "__main__":
    main()
