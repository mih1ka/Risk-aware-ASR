import json
import wordninja

with open("/usr/share/dict/words") as f:
    DICTIONARY = set(w.strip().lower() for w in f if w.strip().isalpha())

COMMON_PREFIXES = ["doing","going","being","and","of","so","to","with","for",
                    "on","in","at","but","the","that","this","have","has",
                    "not","was","were","are","is"]

path = "data/processed/turns.jsonl"
suspects = {}

with open(path) as f:
    for line in f:
        rec = json.loads(line)
        for token in rec["text"].split():
            word = token.strip(".,?!'\"").lower()
            if not word.isalpha() or len(word) < 5:
                continue
            if word in DICTIONARY:
                continue  # already a real single word — not a fusion
            for prefix in COMMON_PREFIXES:
                if word.startswith(prefix) and len(word) >= len(prefix) + 2:
                    remainder = word[len(prefix):]
                    if remainder in DICTIONARY:
                        suspects[word] = suspects.get(word, 0) + 1
                        break

for word, count in sorted(suspects.items(), key=lambda x: -x[1])[:40]:
    print(f"{count:3d}  {word}")