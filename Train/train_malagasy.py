import torch
import os
import pandas as pd
from model import BigramLanguageModel

# ======================
#   HYPERPARAMÈTRES
# ======================
# Conseils pour un apprentissage approfondi sur un Core i5 8ème génération, 16Go RAM, Windows 11 :
# - Augmenter block_size pour capturer plus de contexte (attention à la RAM)
# - Augmenter n_embd, n_head, n_layer pour un modèle plus puissant (dans la limite de la RAM)
# - Réduire batch_size si la RAM est saturée, sinon l'augmenter pour accélérer l'entraînement
# - max_iters plus élevé pour un apprentissage plus long
# - learning_rate plus bas pour une convergence plus fine si tu entraînes longtemps

def get_hyperparameters():
    """Retourne les hyperparamètres d'entraînement"""
    return {
        'batch_size': 8,       # Réduit pour éviter l'overflow mémoire, tu peux essayer 16 si ça passe
        'block_size': 128,     # Plus de contexte, mais attention à la RAM (tu peux tester 256 si ça passe)
        'max_iters': 20000,    # Plus d'itérations pour un apprentissage approfondi
        'eval_interval': 500,  # Garde la même valeur pour surveiller la perte régulièrement
        'learning_rate': 5e-4, # Plus bas pour un apprentissage plus stable sur le long terme
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'eval_iters': 200,
        'n_embd': 256,         # Embedding plus large pour plus de capacité
        'n_head': 8,           # Plus de têtes d'attention pour mieux modéliser les dépendances
        'n_layer': 8,          # Plus de couches pour un modèle plus profond
        'dropout': 0.1,        # Ajoute du dropout pour éviter l'overfitting
        # Checkpoints
        'checkpoint_dir': None,           # Défini dans main en fonction du projet
        'checkpoint_interval': 1000,      # Sauvegarder toutes les N itérations
        'resume_from_checkpoint': True,   # Reprendre automatiquement si un checkpoint existe
    }

def load_data(use_csv=False, txt_file=None, csv_file=None, csv_column="text"):
    """
    Charge les données d'entraînement depuis un fichier texte ou CSV
    
    Args:
        use_csv: bool, utiliser un fichier CSV ou texte
        txt_file: str, chemin vers le fichier texte
        csv_file: str, chemin vers le fichier CSV
        csv_column: str, nom de la colonne contenant le texte
    
    Returns:
        text: str, texte chargé
    """
    if use_csv:
        df = pd.read_csv(csv_file)
        text = " ".join(df[csv_column].astype(str).tolist())
    else:
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    print("Nombre total de caractères :", len(text))
    return text

def build_vocabulary(text):
    """
    Construit le vocabulaire à partir du texte
    
    Args:
        text: str, texte d'entraînement
    
    Returns:
        vocab_size, stoi, itos, encode, decode
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("Taille du vocabulaire :", vocab_size)
    
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return vocab_size, stoi, itos, encode, decode

def prepare_data(text, encode, train_split=0.9):
    """
    Prépare les données d'entraînement et de validation
    
    Args:
        text: str, texte d'entraînement
        encode: fonction d'encodage
        train_split: float, proportion des données pour l'entraînement
    
    Returns:
        train_data, val_data
    """
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data

def get_batch(split, train_data, val_data, batch_size, block_size, device):
    """
    Génère un batch de données pour l'entraînement ou la validation
    
    Args:
        split: str, 'train' ou 'val'
        train_data: tensor, données d'entraînement
        val_data: tensor, données de validation
        batch_size: int, taille du batch
        block_size: int, taille du contexte
        device: str, device à utiliser
    
    Returns:
        x, y: tensors, batch d'entrée et de sortie
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters):
    """
    Estime la perte sur les données d'entraînement et de validation
    
    Args:
        model: modèle à évaluer
        train_data: tensor, données d'entraînement
        val_data: tensor, données de validation
        batch_size: int, taille du batch
        block_size: int, taille du contexte
        device: str, device à utiliser
        eval_iters: int, nombre d'itérations pour l'évaluation
    
    Returns:
        dict: pertes d'entraînement et de validation
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model, train_data, val_data, hyperparams):
    """
    Entraîne le modèle avec les données fournies
    
    Args:
        model: modèle à entraîner
        train_data: tensor, données d'entraînement
        val_data: tensor, données de validation
        hyperparams: dict, hyperparamètres d'entraînement
    
    Returns:
        model: modèle entraîné
    """
    device = hyperparams['device']
    batch_size = hyperparams['batch_size']
    block_size = hyperparams['block_size']
    max_iters = hyperparams['max_iters']
    eval_interval = hyperparams['eval_interval']
    learning_rate = hyperparams['learning_rate']
    eval_iters = hyperparams['eval_iters']
    
    print(f"🚀 Début de l'entraînement sur {device}")
    print(f"📊 Modèle avec {sum(p.numel() for p in model.parameters())/1e6:.1f}M paramètres")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Préparation du checkpoint
    checkpoint_dir = hyperparams.get('checkpoint_dir') or '.'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    checkpoint_interval = int(hyperparams.get('checkpoint_interval', 0) or 0)
    resume_from_checkpoint = bool(hyperparams.get('resume_from_checkpoint', False))

    start_iter = 0
    if resume_from_checkpoint and os.path.isfile(checkpoint_path):
        try:
            print(f"🔁 Reprise depuis le checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_iter = int(ckpt.get('iter', 0)) + 1
            print(f"➡️  Reprise à l'itération {start_iter}")
        except Exception as e:
            print(f"⚠️  Impossible de charger le checkpoint: {e}. Nouvel entraînement.")
    
    for iter in range(start_iter, max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters)
            print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        
        xb, yb = get_batch('train', train_data, val_data, batch_size, block_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Sauvegarde périodique du checkpoint
        if checkpoint_interval and iter > 0 and (iter % checkpoint_interval == 0):
            try:
                torch.save({
                    'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hyperparams': hyperparams,
                }, checkpoint_path)
                print(f"💾 Checkpoint sauvegardé à l'itération {iter} -> {checkpoint_path}")
            except Exception as e:
                print(f"⚠️  Échec de la sauvegarde du checkpoint: {e}")
    
    print("✅ Entraînement terminé!")
    return model

def generate_sample_text(model, decode, device, max_new_tokens=500):
    """
    Génère un échantillon de texte avec le modèle entraîné
    
    Args:
        model: modèle entraîné
        decode: fonction de décodage
        device: device à utiliser
        max_new_tokens: nombre de tokens à générer
    
    Returns:
        str: texte généré
    """
    print("\n🎯 Génération d'un échantillon de texte...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
    
    print("\n=== TEXTE GÉNÉRÉ ===\n")
    print(generated)
    print("\n" + "="*50)
    
    return generated

def save_model(model, stoi, itos, hyperparams, output_path):
    """
    Sauvegarde le modèle entraîné avec tous les paramètres nécessaires
    
    Args:
        model: modèle entraîné
        stoi: dictionnaire string-to-index
        itos: dictionnaire index-to-string
        hyperparams: dict, hyperparamètres utilisés
        output_path: str, chemin de sauvegarde
    """
    print(f"💾 Sauvegarde du modèle dans {output_path}...")
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sauvegarder le modèle avec tous les paramètres
    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'vocab_size': len(stoi),
        'n_embd': hyperparams['n_embd'],
        'n_head': hyperparams['n_head'],
        'n_layer': hyperparams['n_layer'],
        'block_size': hyperparams['block_size'],
        'dropout': hyperparams['dropout'],
    }, output_path)
    
    print("✅ Modèle sauvegardé avec succès!")

def main():
    """
    Fonction principale qui orchestre tout le processus d'entraînement
    """
    print("🎯 Démarrage de l'entraînement du modèle malagasy")
    print("="*60)
    
    # Configuration des chemins
    directory = "D:/Train"
    output_directory = f"{directory}/out-put"
    data_directory = f"{directory}/data"
    data_scrapping_directory = "D:/Scrapping"
    scrapping_out_directory = f"{data_scrapping_directory}/out-put"
    
    # Configuration des données
    use_csv = False
    data_output_file = f"{data_directory}/tantara_malagasy.txt"
    txt_file = data_output_file
    csv_file = "donnees_malagasy.csv"
    csv_column = "text"
    
    # Initialiser le seed pour la reproductibilité
    torch.manual_seed(1337)
    
    # Récupérer les hyperparamètres
    hyperparams = get_hyperparameters()
    print(f"🔧 Device utilisé: {hyperparams['device']}")
    
    # Charger et préparer les données
    print("\n📂 Chargement des données...")
    text = load_data(use_csv, txt_file, csv_file, csv_column)
    
    print("\n🔤 Construction du vocabulaire...")
    vocab_size, stoi, itos, encode, decode = build_vocabulary(text)
    
    print("\n📊 Préparation des données d'entraînement...")
    train_data, val_data = prepare_data(text, encode)
    
    # Créer le modèle
    print("\n🏗️ Création du modèle...")
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=hyperparams['n_embd'],
        n_head=hyperparams['n_head'],
        n_layer=hyperparams['n_layer'],
        block_size=hyperparams['block_size'],
        dropout=hyperparams['dropout']
    ).to(hyperparams['device'])
    
    # Entraîner le modèle
    print("\n🚀 Début de l'entraînement...")
    # Configurer le dossier de checkpoints dans l'output du projet
    hyperparams['checkpoint_dir'] = f"{output_directory}/checkpoints"
    model = train_model(model, train_data, val_data, hyperparams)
    
    # Générer un échantillon de texte
    generate_sample_text(model, decode, hyperparams['device'])
    
    # Sauvegarder le modèle
    model_path = f"{output_directory}/modele_malagasy.pth"
    save_model(model, stoi, itos, hyperparams, model_path)
    
    print("\n🎉 Processus d'entraînement terminé avec succès!")
    print(f"📁 Modèle sauvegardé: {model_path}")

if __name__ == "__main__":
    main()