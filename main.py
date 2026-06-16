import argparse
import os
import sys
import subprocess

from config import TRAIN_DIR, TRAIN_OUTPUT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_SCRIPT = os.path.join(TRAIN_DIR, "train_malagasy.py")
INFER_SCRIPT = os.path.join(TRAIN_DIR, "inference.py")
FINETUNE_SCRIPT = os.path.join(TRAIN_DIR, "finetune_discours.py")


def main():
    parser = argparse.ArgumentParser(
        description="IA_NAH — entraînement, test et fine-tuning du modèle malagasy"
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['train', 'test', 'finetune'],
        help="'train' : pré-entraînement | 'test' : génération | 'finetune' : fine-tuning discours"
    )
    parser.add_argument('--model-name', type=str, default="modele_malagasy.pth",
                        help="Nom du fichier modèle source (train/test/finetune)")
    parser.add_argument('--output-name', type=str, default=None,
                        help="Nom du fichier de sortie (finetune, défaut : modele_discours.pth)")
    parser.add_argument('--prompt', type=str, default='',
                        help="Prompt de départ (modes test et finetune)")
    parser.add_argument('--tokens', type=int, default=500,
                        help="Nombre de tokens à générer")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="Température pour la génération")
    parser.add_argument('--top-k', type=int, default=None, help="Top-k sampling")
    parser.add_argument('--interactive', action='store_true', help="Mode interactif")
    parser.add_argument('--epochs', type=int, default=80,
                        help="Nombre d'époques (mode finetune)")
    parser.add_argument('--lr', type=float, default=2e-5,
                        help="Learning rate de pointe (mode finetune)")
    args = parser.parse_args()

    model_path = (
        args.model_name if os.path.isabs(args.model_name)
        else os.path.join(TRAIN_OUTPUT, args.model_name)
    )

    if args.mode == 'train':
        print(f"\nEntraînement — modèle sauvegardé : {model_path}\n")
        env = os.environ.copy()
        env['MODEL_OUTPUT_PATH'] = model_path
        subprocess.run([sys.executable, TRAIN_SCRIPT], env=env)

    elif args.mode == 'test':
        print(f"\nInférence — modèle : {model_path}\n")
        cmd = [
            sys.executable, INFER_SCRIPT,
            '--model', model_path,
            '--prompt', args.prompt,
            '--tokens', str(args.tokens),
            '--temperature', str(args.temperature),
        ]
        if args.top_k is not None:
            cmd += ['--top-k', str(args.top_k)]
        if args.interactive:
            cmd += ['--interactive']
        subprocess.run(cmd)

    elif args.mode == 'finetune':
        output_name = args.output_name or "modele_discours.pth"
        output_path = (
            output_name if os.path.isabs(output_name)
            else os.path.join(TRAIN_OUTPUT, output_name)
        )
        print(f"\nFine-tuning discours — source : {model_path} → sortie : {output_path}\n")
        cmd = [
            sys.executable, FINETUNE_SCRIPT,
            '--pretrained', model_path,
            '--output', output_path,
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--prompt', args.prompt,
            '--tokens', str(args.tokens),
            '--temperature', str(args.temperature),
        ]
        if args.top_k is not None:
            cmd += ['--top-k', str(args.top_k)]
        if args.interactive:
            cmd += ['--interactive']
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
