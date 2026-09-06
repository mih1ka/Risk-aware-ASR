import json

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
        stems.add(word[:-2])
        stems.add(word[:-1])
    if word.endswith("s") and not word.endswith("ss"):
        stems.add(word[:-1])
    if word.endswith("ing"):
        stems.add(word[:-3])
        stems.add(word[:-3] + "e")
    if word.endswith("ed"):
        stems.add(word[:-2])
        stems.add(word[:-1])
        stems.add(word[:-2] + "e")
    return any(s in DICTIONARY for s in stems)

candidates = {}
with open("data/processed/turns.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        for token in rec["text"].split():
            word = token.strip(".,?!'\"").lower()
            if not word.isalpha() or len(word) < 5:
                continue
            if is_common_word(word):
                continue
            candidates[word] = candidates.get(word, 0) + 1

tier3, tier2_candidates = {}, {}
for word, count in candidates.items():
    if word in RXNORM_TERMS:
        tier3[word] = count
    else:
        tier2_candidates[word] = count

print(f"Tier 3 (matched RxNorm — drug names): {len(tier3)} unique terms")
for w, c in sorted(tier3.items(), key=lambda x: -x[1])[:25]:
    print(f"  {c:3d}  {w}")

print(f"\nTier 2 candidates: {len(tier2_candidates)} unique terms")
for w, c in sorted(tier2_candidates.items(), key=lambda x: -x[1])[:80]:
    print(f"  {c:3d}  {w}")