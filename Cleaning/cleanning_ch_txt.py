import re
from config import SCRAPPING_OUTPUT

ORDINALS = {
    1: "voalohany",
    2: "faharoa",
    3: "fahatelo",
    4: "fahefatra",
    5: "fahefadimy",
    6: "fahenina",
    7: "fahafito",
    8: "fahavalo",
    9: "fahasivy",
    10: "fahafolo",
    11: "fahairaika ambin’ny folo",
    12: "faharoa ambin’ny folo",
    13: "fahatelo ambin’ny folo",
    14: "fahefatra ambin’ny folo",
    15: "fahefadimy ambin’ny folo",
    16: "fahenina ambin’ny folo",
    17: "fahafito ambin’ny folo",
    18: "fahavalo ambin’ny folo",
    19: "fahasivy ambin’ny folo",
    20: "faharoapolo",
    21: "fahairaika amby roapolo",
    22: "faharoa amby roapolo",
    23: "fahatelo amby roapolo",
    24: "fahefatra amby roapolo",
    25: "fahefadimy amby roapolo",
    26: "fahenina amby roapolo",
    27: "fahafito amby roapolo",
    28: "fahavalo amby roapolo",
    29: "fahasivy amby roapolo",
    30: "fahatelo-polo",
    31: "fahairaika amby telopolo",
    32: "faharoa amby telopolo",
    33: "fahatelo amby telopolo",
    34: "fahefatra amby telopolo",
    35: "fahefadimy amby telopolo",
    36: "fahenina amby telopolo",
    37: "fahafito amby telopolo",
    38: "fahavalo amby telopolo",
    39: "fahasivy amby telopolo",
    40: "fahefat-polo",
    41: "fahairaika amby efa-polo",
    42: "faharoa amby efa-polo",
    43: "fahatelo amby efa-polo",
    44: "fahefatra amby efa-polo",
    45: "fahefadimy amby efa-polo",
    46: "fahenina amby efa-polo",
    47: "fahafito amby efa-polo",
    48: "fahavalo amby efa-polo",
    49: "fahasivy amby efa-polo",
    50: "fahefa-dimpolo",
    51: "fahairaika amby dimampolo" ,
  52: "faharoa amby dimampolo" ,
  53: "fahatelo amby dimampolo" ,
  54: "fahefatra amby dimampolo" ,
  55: "fahefadimy amby dimampolo" ,
  56: "fahenina amby dimampolo" ,
  57: "fahafito amby dimampolo" ,
  58: "fahavalo amby dimampolo" ,
  59: "fahasivy amby dimampolo" ,
  60: "fahenimpolo" ,

  61: "fahairaika amby enimpolo" ,
  62: "faharoa amby enimpolo" ,
  63: "fahatelo amby enimpolo" ,
  64: "fahefatra amby enimpolo" ,
  65: "fahefadimy amby enimpolo" ,
  66: "fahenina amby enimpolo" ,
  67: "fahafito amby enimpolo" ,
  68: "fahavalo amby enimpolo" ,
  69: "fahasivy amby enimpolo" ,
  70: "fahafito-polo" ,

  71: "fahairaika amby fitopolo" ,
  72: "faharoa amby fitopolo" ,
  73: "fahatelo amby fitopolo" ,
  74: "fahefatra amby fitopolo" ,
  75: "fahefadimy amby fitopolo" ,
  76: "fahenina amby fitopolo" ,
  77: "fahafito amby fitopolo" ,
  78: "fahavalo amby fitopolo" ,
  79: "fahasivy amby fitopolo" ,
  80: "fahavalopolo" ,

  81: "fahairaika amby valopolo" ,
  82: "faharoa amby valopolo" ,
  83: "fahatelo amby valopolo" ,
  84: "fahefatra amby valopolo" ,
  85: "fahefadimy amby valopolo" ,
  86: "fahenina amby valopolo" ,
  87: "fahafito amby valopolo" ,
  88: "fahavalo amby valopolo" ,
  89: "fahasivy amby valopolo" ,
  90: "sivifolo" ,

  91: "fahairaika amby sivifolo" ,
  92: "faharoa amby sivifolo" ,
  93: "fahatelo amby sivifolo" ,
  94: "fahefatra amby sivifolo" ,
  95: "fahefadimy amby sivifolo" ,
  96: "fahenina amby sivifolo" ,
  97: "fahafito amby sivifolo" ,
  98: "fahavalo amby sivifolo" ,
  99: "fahasivy amby sivifolo" ,
  100: "faha-jato" ,

  101: "fahairaika amby zato" ,
  102: "faharoa amby zato" ,
  103: "fahatelo amby zato" ,
  104: "fahefatra amby zato" ,
  105: "fahefadimy amby zato" ,
  106: "fahenina amby zato" ,
  107: "fahafito amby zato" ,
  108: "fahavalo amby zato" ,
  109: "fahasivy amby zato" ,
  110: "fahafolo amby zato" ,

  111: "fahairaika ambin’ny folo amby zato" ,
  112: "faharoa ambin’ny folo amby zato" ,
  113: "fahatelo ambin’ny folo amby zato" ,
  114: "fahaefatra ambin’ny folo amby zato" ,
  115: "fahadimy ambin’ny folo amby zato" ,
  116: "fahaenina ambin’ny folo amby zato" ,
  117: "fahafito ambin’ny folo amby zato" ,
  118: "fahavalo ambin’ny folo amby zato" ,
  119: "fahasivy ambin’ny folo amby zato" ,
  120: "faharoapolo amby zato" ,

  121: "fahairaika amby roapolo amby zato" ,
  122: "faharoa amby roapolo amby zato" ,
  123: "fahatelo amby roapolo amby zato" ,
  124: "fahaefatra amby roapolo amby zato" ,
  125: "fahadimy amby roapolo amby zato"
}

# Regex:
#  - faha-<nombre>  -> remplacer par l’ordinal (1–50)
#  - <nombre> seul  -> remplacer par l’ordinal (1–50)
RE_FAHA_NUMBER = re.compile(r"\bfaha-\s*(\d{1,3})\b", flags=re.IGNORECASE)
RE_NUMBER_ALONE = re.compile(r"\b(\d{1,3})\b")

def to_ordinal(n: int) -> str:
    return ORDINALS.get(n, str(n))  # laisse tel quel si >50 ou non mappé

def replace_faha(match: re.Match) -> str:
    n = int(match.group(1))
    return to_ordinal(n)

def replace_number(match: re.Match) -> str:
    n = int(match.group(1))
    return to_ordinal(n) + ","

def clean_and_convert(text: str) -> str:
    # 1) traiter d’abord les formes avec préfixe 'faha-'
    text = RE_FAHA_NUMBER.sub(replace_faha, text)
    # 2) puis les nombres isolés
    text = RE_NUMBER_ALONE.sub(replace_number, text)
    # 3) normalisations simples d’espaces
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text

# Exemple
if __name__ == "__main__":
    # Ouvrir le fichier pour lire son contenu
    chemin_fichier = f"{SCRAPPING_OUTPUT}\cleaning-baiboly-malagasy.txt"
    with open(chemin_fichier, 'r', encoding='utf-8') as f:
        texte = f.read()

    # Nettoyer et convertir le texte
    texte_converti = clean_and_convert(texte)

    # Écrire le texte converti dans le même fichier (remplacement)
    with open(chemin_fichier, 'w', encoding='utf-8') as f:
        f.write(texte_converti)