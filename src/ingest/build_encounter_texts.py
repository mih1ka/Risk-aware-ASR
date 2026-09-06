import json
from collections import defaultdict

grouped = defaultdict(list)

with open("data/processed/turns.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        key = (rec["split"], rec["dataset"], rec["encounter_id"])
        grouped[key].append(rec["text"])

records = []
for (split, dataset, encounter_id), texts in grouped.items():
    full_text = " ".join(texts)
    records.append({
        "split": split,
        "dataset": dataset,
        "encounter_id": encounter_id,
        "full_text": full_text,
        "num_turns": len(texts),
        "char_length": len(full_text),
    })

with open("data/processed/encounter_texts.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"Built {len(records)} encounter-level texts")
lengths = [r["char_length"] for r in records]
print(f"Length range: {min(lengths)}–{max(lengths)} characters (avg {sum(lengths)//len(lengths)})")