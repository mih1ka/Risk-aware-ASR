import json
import pandas as pd

with open("/usr/share/dict/words") as f:
    DICTIONARY = set(w.strip().lower() for w in f if w.strip().isalpha())

with open("data/glossary/rxnorm_raw/displaynames.json") as f:
    rx_data = json.load(f)
RXNORM_TERMS = set(t.strip().lower() for t in rx_data["displayTermsList"]["term"])

def is_common_word(word):
    if word in DICTIONARY:
        return True
    stems = set()
    if word.endswith("ies"):
        stems.add(word[:-3] + "y")
    if word.endswith("es"):
        stems.add(word[:-2]); stems.add(word[:-1])
    if word.endswith("s") and not word.endswith("ss"):
        stems.add(word[:-1])
    if word.endswith("ing"):
        stems.add(word[:-3]); stems.add(word[:-3] + "e")
    if word.endswith("ed"):
        stems.add(word[:-2]); stems.add(word[:-1]); stems.add(word[:-2] + "e")
    return any(s in DICTIONARY for s in stems)

# Real drugs RxNorm's exact-string match missed — confirmed by hand
TIER3_EXTRA = {"flexeril", "zofran"}

# Curated from manual review of the Tier-2 candidate list
TIER2_TERMS = {
    "echocardiogram", "ultrasound", "lipid", "musculoskeletal", "strep",
    "autoimmune", "rehab", "pedis", "rhonchi", "midline", "followup",
    "gallbladder", "mammogram", "workup", "paraspinal", "statin",
    "debridement", "neurologic", "nontender", "nsaid", "nsaids",
    "epicondylitis", "weightbearing", "tendinopathy", "chemo", "lisfranc",
    "nonhealing", "hyperlipidemia", "lumpectomy", "mellitus", "mcmurray",
    "midfoot", "nonpitting", "radiculopathy", "corticosteroid",
    "immobilizer", "drusen", "neovascular", "lachman", "satting",
    "impingement", "shingles", "adenopathy", "lymphadenopathy",
    "neurovascular", "dorsalis", "malleolus", "epidural", "herniation",
    "herniated", "olecranon", "meniscus", "epicondyle", "sternoclavicular",
    "esophagitis",
}

candidates = set()
with open("data/processed/turns.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        for token in rec["text"].split():
            word = token.strip(".,?!'\"").lower()
            if word.isalpha() and len(word) >= 5 and not is_common_word(word):
                candidates.add(word)

rows = []
for word in candidates:
    if word in RXNORM_TERMS or word in TIER3_EXTRA:
        rows.append({"term": word, "tier": 3, "category": "drug_dosage"})
    elif word in TIER2_TERMS:
        rows.append({"term": word, "tier": 2, "category": "clinical_term"})

df = pd.DataFrame(rows).drop_duplicates(subset="term").sort_values(["tier", "term"])
df.to_csv("data/glossary/tiered_glossary.csv", index=False)

print(df["tier"].value_counts())
print(f"\nSaved {len(df)} entries to data/glossary/tiered_glossary.csv")