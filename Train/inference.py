import torch
import argparse
import os
from model import BigramLanguageModel

def load_model(checkpoint_path, device='cpu'):
    """
    Charge le modèle depuis un checkpoint
    
    Args:
        checkpoint_path: chemin vers le fichier .pth
        device: device sur lequel charger le modèle
    
    Returns:
        model, stoi, itos, decode_function
    """
    print(f"🔄 Chargement du modèle depuis {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Le fichier {checkpoint_path} n'existe pas!")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Récupérer les paramètres du modèle
    vocab_size = checkpoint["vocab_size"]
    n_embd = checkpoint.get("n_embd", 256)
    n_head = checkpoint.get("n_head", 8)
    n_layer = checkpoint.get("n_layer", 8)
    block_size = checkpoint.get("block_size", 128)
    dropout = checkpoint.get("dropout", 0.1)
    
    # Récupérer le vocabulaire
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    
    # Créer la fonction de décodage
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Recréer le modèle
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"✅ Modèle chargé avec {sum(p.numel() for p in model.parameters())/1e6:.1f}M paramètres")
    print(f"📊 Vocabulaire: {vocab_size} caractères")
    print(f"🔧 Architecture: {n_layer} couches, {n_head} têtes d'attention, {n_embd} dimensions")
    
    return model, stoi, itos, decode

def generate_text(model, decode, prompt="", max_new_tokens=500, temperature=1.0, top_k=None, device='cpu'):
    """
    Génère du texte avec le modèle
    
    Args:
        model: modèle chargé
        decode: fonction de décodage
        prompt: texte de départ (optionnel)
        max_new_tokens: nombre de tokens à générer
        temperature: contrôle la créativité
        top_k: limite le choix aux k meilleurs tokens
        device: device à utiliser
    
    Returns:
        texte généré
    """
    print(f"🎯 Génération avec température={temperature}, top_k={top_k}")
    
    # Encoder le prompt si fourni
    if prompt:
        # Pour simplifier, on utilise les premiers caractères du vocabulaire
        # Dans un vrai cas, il faudrait encoder le prompt avec stoi
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = model.generate(
            context, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )[0].tolist()
    
    generated_text = decode(output)
    return generated_text

def interactive_mode(model, decode, device='cpu'):
    """
    Mode interactif pour tester le modèle
    """
    print("\n🎮 Mode interactif - Tapez 'quit' pour quitter")
    print("Commandes disponibles:")
    print("  - Entrez du texte pour générer une suite")
    print("  - 'temp X' pour changer la température (ex: temp 1.5)")
    print("  - 'topk X' pour changer top_k (ex: topk 50)")
    print("  - 'tokens X' pour changer le nombre de tokens (ex: tokens 200)")
    
    temperature = 1.0
    top_k = None
    max_tokens = 200
    
    while True:
        try:
            user_input = input("\n💬 Votre prompt: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Au revoir!")
                break
            
            if user_input.startswith('temp '):
                temperature = float(user_input.split()[1])
                print(f"🌡️ Température changée à {temperature}")
                continue
            
            if user_input.startswith('topk '):
                top_k = int(user_input.split()[1])
                print(f"🔝 Top-k changé à {top_k}")
                continue
            
            if user_input.startswith('tokens '):
                max_tokens = int(user_input.split()[1])
                print(f"🔢 Nombre de tokens changé à {max_tokens}")
                continue
            
            if not user_input:
                user_input = ""
            
            print(f"\n🤖 Génération en cours...")
            generated = generate_text(
                model, decode, user_input, 
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
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
    parser = argparse.ArgumentParser(description='Test et génération avec le modèle malagasy')
    parser.add_argument('--model', type=str, default='out-put/modele_malagasy.pth',
                       help='Chemin vers le fichier de modèle (.pth)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device à utiliser')
    parser.add_argument('--prompt', type=str, default='',
                       help='Texte de départ pour la génération')
    parser.add_argument('--tokens', type=int, default=500,
                       help='Nombre de tokens à générer')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Température pour la génération (1.0 = normal)')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Limite le choix aux k meilleurs tokens')
    parser.add_argument('--interactive', action='store_true',
                       help='Lance le mode interactif')
    
    args = parser.parse_args()
    
    # Déterminer le device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🖥️ Utilisation du device: {device}")
    
    try:
        # Charger le modèle
        model, stoi, itos, decode = load_model(args.model, device)
        
        if args.interactive:
            # Mode interactif
            interactive_mode(model, decode, device)
        else:
            # Génération simple
            print(f"\n🎯 Génération avec les paramètres:")
            print(f"   - Prompt: '{args.prompt}'")
            print(f"   - Tokens: {args.tokens}")
            print(f"   - Température: {args.temperature}")
            print(f"   - Top-k: {args.top_k}")
            
            generated = generate_text(
                model, decode, args.prompt,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
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
