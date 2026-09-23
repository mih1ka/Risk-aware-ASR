import json, re
from pathlib import Path

GLOSSARY_PATH = Path("data/glossary/tiered_glossary.csv")
ENCOUNTERS_PATH = Path("data/processed/encounter_texts.jsonl")

glossary_terms = set()
with open(GLOSSARY_PATH) as f:
    next(f)  # skip header
    for line in f:
        term, tier, category = line.strip().split(",")
        glossary_terms.add(term.lower())

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

hits = []
with open(ENCOUNTERS_PATH) as f:
    for line in f:
        rec = json.loads(line)
        if rec["dataset"] != "fareez":
            continue
        tokens = set(tokenize(rec["full_text"]))
        matched = tokens & glossary_terms
        if matched:
            hits.append((rec["encounter_id"], len(matched), matched))

hits.sort(key=lambda x: -x[1])
for eid, n, matched in hits[:5]:
    print(eid, n, matched)
