import csv
import re
import unicodedata
from pathlib import Path

INPUT_DIR = Path(r"D:\Scrapping\csv")
OUTPUT_TXT = Path(r"D:\Scrapping\out-put\dict_merged_clean.txt")
OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

GRAMMAR_TAGS = [
    r"\ba\.\b", r"\ba\.,", r"\bp\.\?\s*t\.\?\b", r"\bft\.\b", r"\bfr\.\b",
    r"\bmt\.\?\s*at\.\?\b", r"\bmt\.\?\s*ih\.\?\b", r"\bmt\.\?\s*fit\.\?\b",
    r"\bt\.i\.f\.\b", r"\bt\.?\s*i\.?\s*f\.?\b", r"\bp\.?\s*t\.?\b", r"\bt\.?\s*v\.?\s*r\.?\b"
]
GRAMMAR_TAGS_RE = re.compile(r"^(?:" + r"|".join(GRAMMAR_TAGS) + r")\s*:?[\s\-–—]*", re.IGNORECASE)

# Supprimeurs globaux pour tokens éditoriaux et abréviations où qu'ils soient
GLOBAL_EDITORIAL_TOKENS = re.compile(
    r"\b(?:mt\.?\s*at\.?|mt\.?\s*ih\.?|mt\.?\s*fit\.?|t\.?\s*i\.?\s*f\.?|t\.?\s*v\.?\s*r\.?|p\.?\s*t\.?|a\.|ft\.|fr\.|jar\.|jer\.)\b\s*:?",
    re.IGNORECASE,
)

BULLETS_TABLE = str.maketrans({
    "•": " ",
    "¶": " ",
    "~": " ",
    "—": "-",
    "–": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "\u00A0": " ",  # nbsp
})

WHITESPACE_RE = re.compile(r"\s+")
LEADING_PUNCT_RE = re.compile(r"^[\s,;:.\-–—/\\|]+")
ONLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
SENTENCE_RE = re.compile(r"([A-ZÀ-ÖØ-Þ][^.!?]*[.!?])")

def normalize_unicode(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return text.translate(BULLETS_TABLE)

def strip_editorial_prefixes(text: str) -> str:
    # Retire quelques labels éditoriaux fréquents en début de définition (conservateur)
    changed = True
    s = text
    while changed:
        new = GRAMMAR_TAGS_RE.sub("", s)
        changed = (new != s)
        s = new
    return s

def basic_cleanup(text: str) -> str:
    s = normalize_unicode(text)
    s = s.replace("\r", " ").replace("\n", " ")
    s = LEADING_PUNCT_RE.sub("", s)
    # Espaces autour des deux-points/points-virgules/virgules
    s = re.sub(r"\s*([:;,])\s*", r"\1 ", s)
    # Collapses
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s

def clean_word(word: str) -> str:
    s = basic_cleanup(word)
    # On évite de forcer la casse (certains mots malagasy commencent par majuscule)
    return s

def clean_definition(defn: str) -> str:
    s = basic_cleanup(defn)
    s = strip_editorial_prefixes(s)
    # Normalise "t. i. f." -> "t.i.f." pour faciliter la suppression globale
    s = re.sub(r"(?i)\b([a-z])\.\s+([a-z])\.\s+([a-z])\.\b", r"\1.\2.\3.", s)
    s = re.sub(r"(?i)\b([a-z])\.\s+([a-z])\.\b", r"\1.\2.", s)
    # Supprime tokens éditoriaux restants où qu'ils soient
    try:
        s = GLOBAL_EDITORIAL_TOKENS.sub("", s)
    except NameError:
        pass
    # Supprime petites parenthèses éditoriales (ex: (mt. at.), (fr.), (jar.))
    s = re.sub(r"\(\s*[^)]{1,30}\s*\)", "", s)
    # Retire doubles ponctuations répétées
    s = re.sub(r"([!?.,:;])\1{1,}", r"\1", s)
    # Corrige doubles guillemets issus du CSV
    s = s.replace('""', '"')
    # Corrige quelques artefacts OCR
    s = re.sub(r"\s+\?\s+", "? ", s)
    s = re.sub(r"\s+\.\s+", ". ", s)
    s = re.sub(r"\s+\-\s+", " - ", s)
    # Nettoie espaces avant la ponctuation
    s = re.sub(r"\s+([!?,.;:])", r"\1", s)
    # Ré-insère un espace après ponctuation si collée
    s = re.sub(r"([!?,.;:])([^\s])", r"\1 \2", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def extract_sentences(text: str):
    if not text:
        return []
    candidates = SENTENCE_RE.findall(text)
    sentences = []
    seen = set()
    for cand in candidates:
        s = cand.strip()
        if not s:
            continue
        # Nettoie ponctuation résiduelle et énumérations en début de phrase
        s = re.sub(r"^[\s\-–—,.;:]+", "", s)
        s = re.sub(r"^[a-z]\)\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\d+[.)]\s*", "", s)
        s = s.strip()
        if not s:
            continue
        # Assure majuscule initiale
        s = s[0].upper() + s[1:]
        # Normalisations légères d'espaces
        s = re.sub(r"\s+([!?,.;:])", r"\1", s)
        s = re.sub(r"([!?,.;:])([^\s])", r"\1 \2", s)
        s = WHITESPACE_RE.sub(" ", s).strip()
        if s and s not in seen:
            seen.add(s)
            sentences.append(s)
    return sentences

def is_valid_entry(word: str, definition: str) -> bool:
    if not word or not definition:
        return False
    if len(definition) < 3:
        return False
    if ONLY_PUNCT_RE.match(definition):
        return False
    return True

def find_columns(header):
    # Tente de repérer 'word' et 'definition' sans sensibilité à la casse
    lower = [h.lower().strip() for h in header]
    word_idx = None
    def_idx = None
    for i, h in enumerate(lower):
        if h in {"word", "lemma", "mot", "entry"}:
            word_idx = i if word_idx is None else word_idx
        if h in {"definition", "def", "sense", "gloss", "meaning"}:
            def_idx = i if def_idx is None else def_idx
    # Fallback: si exactement 2 colonnes, on suppose (word, definition)
    if word_idx is None or def_idx is None:
        if len(header) >= 2:
            word_idx, def_idx = 0, 1
    return word_idx, def_idx

def read_csv_file(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sniffer = csv.Sniffer()
        sample = f.read(4096)
        f.seek(0)
        dialect = None
        try:
            dialect = sniffer.sniff(sample)
        except csv.Error:
            pass
        reader = csv.reader(f, dialect) if dialect else csv.reader(f)
        rows = list(reader)
    if not rows:
        return []
    word_idx, def_idx = find_columns(rows[0])
    start = 1 if word_idx is not None and def_idx is not None and rows[0][word_idx].lower() in ("word", "lemma", "mot", "entry") else 0
    data = []
    for r in rows[start:]:
        if not r:
            continue
        # Étire sécurité si lignes tronquées
        if word_idx is None or def_idx is None or max(word_idx, def_idx) >= len(r):
            if len(r) >= 2:
                w, d = r[0], r[1]
            else:
                continue
        else:
            w, d = r[word_idx], r[def_idx]
        data.append((w, d))
    return data

def merge_definitions(definitions):
    # Fusionne définitions uniques, conserve l'ordre d’apparition
    seen = set()
    result = []
    for d in definitions:
        if d not in seen:
            seen.add(d)
            result.append(d)
    return " | ".join(result)

def main():
    csv_files = sorted([p for p in INPUT_DIR.glob("*.csv") if p.is_file()])
    if not csv_files:
        print(f"Aucun CSV trouvé dans: {INPUT_DIR}")
        return

    word_to_defs = {}
    pair_seen = set()

    for path in csv_files:
        rows = read_csv_file(path)
        for w_raw, d_raw in rows:
            w = clean_word(w_raw)
            d = clean_definition(d_raw)
            if not is_valid_entry(w, d):
                continue
            pair_key = (w, d)
            if pair_key in pair_seen:
                continue
            pair_seen.add(pair_key)
            word_to_defs.setdefault(w, []).append(d)

    # Écrit un .txt: word  :  phrases bien formées
    with OUTPUT_TXT.open("w", encoding="utf-8", newline="\n") as out:
        for w in sorted(word_to_defs.keys(), key=lambda x: x.casefold()):
            all_defs = word_to_defs[w]
            all_sentences = []
            seen_sent = set()
            for d in all_defs:
                for s in extract_sentences(d):
                    if s not in seen_sent:
                        seen_sent.add(s)
                        all_sentences.append(s)
            # Fallback si rien d'extrait: fabrique une phrase minimale
            if not all_sentences:
                base = all_defs[0] if all_defs else ""
                base = base.strip()
                if base:
                    # Capitalise et termine par point si besoin
                    base = base[0].upper() + base[1:]
                    if not re.search(r"[.!?]$", base):
                        base += "."
                    all_sentences = [base]
            line = f"{w}  :  {' '.join(all_sentences)}\n"
            out.write(line)

    print(f"Terminé. Sortie: {OUTPUT_TXT} (mots: {len(word_to_defs)})")

if __name__ == "__main__":
    main()