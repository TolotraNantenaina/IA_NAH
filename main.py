import argparse
import os
import sys
import subprocess

from config import TRAIN_DIR
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Définir les chemins des scripts
TRAIN_SCRIPT = os.path.join(TRAIN_DIR, "train_malagasy.py")
INFER_SCRIPT = os.path.join(TRAIN_DIR, "inference.py")


def main():
    parser = argparse.ArgumentParser(description="Application principale : entraînement ou test d'un modèle")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help="Mode à utiliser : 'train' pour entraîner, 'test' pour tester/générer")
    parser.add_argument('--model-name', type=str, default="modele_malagasy.pth",
                        help="Nom du fichier modèle à sauvegarder ou charger (ex: modele_malagasy.pth)")
    parser.add_argument('--prompt', type=str, default='',
                        help="Prompt de départ pour la génération (mode test)")
    parser.add_argument('--tokens', type=int, default=500,
                        help="Nombre de tokens à générer (mode test)")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="Température pour la génération (mode test)")
    parser.add_argument('--top-k', type=int, default=None,
                        help="Top-k sampling (mode test)")
    parser.add_argument('--interactive', action='store_true',
                        help="Mode interactif pour le test")
    args = parser.parse_args()

    #  python main.py --mode train --model-name mon_modele.pth
    if args.mode == 'train':
        # Lancer l'entraînement avec le nom de modèle choisi
        output_path = os.path.join('Train', 'out-put', args.model_name) if not os.path.isabs(args.model_name) else args.model_name
        print(f"\n🚀 Entraînement du modèle, il sera sauvegardé sous : {output_path}\n")
        # On passe le nom du modèle en variable d'environnement pour train_malagasy.py
        env = os.environ.copy()
        env['MODEL_OUTPUT_PATH'] = output_path
        subprocess.run([sys.executable, TRAIN_SCRIPT], env=env)

    #  python main.py --mode test --model-name mon_modele.pth --prompt "Votre texte" --tokens 200 --temperature 1.2
    elif args.mode == 'test':
        # Lancer le test/inférence avec le modèle choisi
        model_path = os.path.join('Train', 'out-put', args.model_name) if not os.path.isabs(args.model_name) else args.model_name
        print(f"\n🧪 Test du modèle : {model_path}\n")
        cmd = [
            sys.executable, INFER_SCRIPT,
            '--model', model_path,
            '--prompt', args.prompt,
            '--tokens', str(args.tokens),
            '--temperature', str(args.temperature)
        ]
        if args.top_k is not None:
            cmd += ['--top-k', str(args.top_k)]
        if args.interactive:
            cmd += ['--interactive']
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
