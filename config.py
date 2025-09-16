import os

# Racine du projet (chemin absolu)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dossiers principaux
SCRAPPING_DIR = os.path.join(PROJECT_ROOT, "Scrapping")
CLEANING_DIR = os.path.join(PROJECT_ROOT, "Cleaning")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "Train")

# Sous-dossiers utiles
SCRAPPING_OUTPUT = os.path.join(SCRAPPING_DIR, "out-put")
SCRAPPING_CTRL = os.path.join(SCRAPPING_DIR, "ctrl")
SCRAPPING_DATA = os.path.join(SCRAPPING_DIR, "data")
SCRAPPING_TXT = os.path.join(SCRAPPING_OUTPUT, "txt")
SCRAPPING_DOC = os.path.join(SCRAPPING_OUTPUT, "doc")
SCRAPPING_CSV = os.path.join(SCRAPPING_OUTPUT, "csv")
TRAIN_OUTPUT = os.path.join(TRAIN_DIR, "out-put")
TRAIN_DATA = os.path.join(TRAIN_DIR, "data")
