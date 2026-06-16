import argparse
import os
import sys

import torch

from model import BigramLanguageModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_OUTPUT

def get_device_arg(device_arg):
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)

def load_model(checkpoint_path, device="cpu"):
    print(f"🔄 Chargement du modèle depuis {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Le fichier {checkpoint_path} n'existe pas!")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint["vocab_size"]
    n_embd = checkpoint.get("n_embd", 256)
    n_head = checkpoint.get("n_head", 8)
    n_layer = checkpoint.get("n_layer", 6)
    block_size = checkpoint.get("block_size", 256)
    dropout = checkpoint.get("dropout", 0.1)
    tokenizer_type = checkpoint.get("tokenizer_type", "char")
    sp = None
    stoi = checkpoint.get("stoi")
    itos = checkpoint.get("itos")
    if tokenizer_type == "bpe":
        import sentencepiece as spm
        spm_path = checkpoint.get("spm_model_path")
        if not spm_path or not os.path.isfile(spm_path):
            base = os.path.dirname(os.path.abspath(checkpoint_path))
            cand = os.path.join(base, "malagasy_bpe.model")
            if os.path.isfile(cand):
                spm_path = cand
            else:
                cand2 = os.path.join(os.path.dirname(base), "malagasy_bpe.model")
                if os.path.isfile(cand2):
                    spm_path = cand2
        if not spm_path or not os.path.isfile(spm_path):
            raise FileNotFoundError(
                "Modèle BPE introuvable. Vérifiez spm_model_path dans le checkpoint."
            )
        sp = spm.SentencePieceProcessor(model_file=spm_path)
        decode = lambda ids: sp.decode(ids)
        encode_prompt = lambda p: sp.encode(p, out_type=int)
    else:
        decode = lambda l: "".join([itos[i] for i in l])
        encode_prompt = lambda p: [stoi[c] for c in p if c in stoi]
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✅ Modèle chargé avec {sum(p.numel() for p in model.parameters())/1e6:.1f}M paramètres")
    print(f"📊 Vocabulaire: {vocab_size} ({tokenizer_type})")
    print(f"🔧 Architecture: {n_layer} couches, {n_head} têtes, {n_embd} dims")
    return model, decode, encode_prompt, tokenizer_type

def generate_text(
    model,
    decode,
    encode_prompt,
    tokenizer_type,
    prompt="",
    max_new_tokens=500,
    temperature=1.0,
    top_k=None,
    device="cpu",
):
    print(f"🎯 Génération avec température={temperature}, top_k={top_k}")
    if prompt:
        ids = encode_prompt(prompt)
        if not ids:
            ids = [0]
        context = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()
    return decode(out)

def interactive_mode(model, decode, encode_prompt, tokenizer_type, device="cpu"):
    print("\n🎮 Mode interactif - Tapez 'quit' pour quitter")
    print("Commandes: temp X, topk X, tokens X")
    temperature = 1.0
    top_k = None
    max_tokens = 200
    while True:
        try:
            user_input = input("\n💬 Votre prompt: ").strip()
            if user_input.lower() == "quit":
                print("👋 Au revoir!")
                break
            if user_input.startswith("temp "):
                temperature = float(user_input.split()[1])
                print(f"🌡️ Température changée à {temperature}")
                continue
            if user_input.startswith("topk "):
                top_k = int(user_input.split()[1])
                print(f"🔝 Top-k changé à {top_k}")
                continue
            if user_input.startswith("tokens "):
                max_tokens = int(user_input.split()[1])
                print(f"🔢 Nombre de tokens changé à {max_tokens}")
                continue
            print(f"\n🤖 Génération en cours...")
            generated = generate_text(
                model,
                decode,
                encode_prompt,
                tokenizer_type,
                user_input,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )
            print(f"\n📝 Texte généré:")
            print("=" * 50)
            print(generated)
            print("=" * 50)
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test et génération avec le modèle malagasy")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(TRAIN_OUTPUT, "modele_malagasy.pth"),
        help="Chemin vers le fichier de modèle (.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device à utiliser",
    )
    parser.add_argument("--prompt", type=str, default="", help="Texte de départ")
    parser.add_argument("--tokens", type=int, default=500, help="Nombre de tokens à générer")
    parser.add_argument("--temperature", type=float, default=1.0, help="Température")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif")
    args = parser.parse_args()
    device = get_device_arg(args.device)
    print(f"🖥️ Utilisation du device: {device}")
    try:
        model, decode, encode_prompt, tokenizer_type = load_model(args.model, device)
        if args.interactive:
            interactive_mode(model, decode, encode_prompt, tokenizer_type, device)
        else:
            print(f"\n🎯 Génération: prompt={args.prompt!r}, tokens={args.tokens}")
            generated = generate_text(
                model,
                decode,
                encode_prompt,
                tokenizer_type,
                args.prompt,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )
            print(f"\n📝 Texte généré:")
            print("=" * 60)
            print(generated)
            print("=" * 60)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
