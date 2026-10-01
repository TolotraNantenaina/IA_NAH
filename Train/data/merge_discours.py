"""
Fusionne toutes les sources de discours malgaches en un corpus propre.

Sources :
- discours_malagasy.txt       : dialogue Rakoto/Rasoa (référence, bonne qualité)
- discours_ex_malagasy.txt    : duplicata partiel avec phrase tronquée → ignoré
- discours_malagasy.json      : 49 kabary/proverbes format instruction
- csv/question_answer_malagasy.csv : 219 paires Q&A factuelles
"""
import csv
import json
import os
import re

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(DATA_DIR, "discours_merged.txt")

sections = []


def add(text):
    text = text.strip()
    if text:
        sections.append(text)


# --- 1. Dialogue principal (référence) ---
with open(os.path.join(DATA_DIR, "discours_malagasy.txt"), encoding="utf-8") as f:
    add(f.read())

# --- 2. JSON : extraire les conversations kabary/proverbes ---
with open(os.path.join(DATA_DIR, "discours_malagasy.json"), encoding="utf-8") as f:
    entries = json.load(f)

for entry in entries:
    msgs = entry.get("messages", [])
    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    if user_msg and asst_msg:
        # Nettoie les phrases tronquées ou trop courtes (< 20 chars)
        if len(asst_msg) < 20:
            continue
        block = f"Fanontaniana: {user_msg}\nValiny: {asst_msg}"
        add(block)

# --- 3. CSV : paires Q&A factuelles ---
csv_path = os.path.join(DATA_DIR, "csv", "question_answer_malagasy.csv")
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a:
            continue
        # Ignore les réponses tronquées (se terminent au milieu d'un mot)
        if len(a) < 10 or a.endswith(("Mala", "ka", "sy")):
            continue
        block = f"Fanontaniana: {q}\nValiny: {a}"
        add(block)

# --- Écriture du fichier fusionné ---
merged = "\n\n".join(sections)

# Nettoyage : double espaces, lignes vides excessives
merged = re.sub(r" {2,}", " ", merged)
merged = re.sub(r"\n{3,}", "\n\n", merged)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(merged)

total_chars = len(merged)
total_sections = len(sections)
print(f"Fusion terminée : {total_sections} blocs | {total_chars:,} caractères")
print(f"Fichier : {OUT_FILE}")
