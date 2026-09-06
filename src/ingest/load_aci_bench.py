import pandas as pd
import re
import json
from pathlib import Path

RAW_DIR = Path("data/raw_aci_bench/data/challenge_data")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILES = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test1": "clinicalnlp_taskB_test1.csv",
    "test2": "clinicalnlp_taskC_test2.csv",
    "test3": "clef_taskC_test3.csv",
}

TURN_PATTERN = re.compile(r"\[(doctor|patient)\]\s*(.*)")

def parse_dialogue(dialogue_text):
    turns = []
    for line in dialogue_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = TURN_PATTERN.match(line)
        if m:
            speaker, text = m.groups()
            turns.append({"speaker": speaker, "text": text.strip()})
        elif turns:
            turns[-1]["text"] += " " + line
    return turns

all_records = []
for split_name, filename in SPLIT_FILES.items():
    path = RAW_DIR / filename
    if not path.exists():
        print(f"Skipping {split_name}: {filename} not found")
        continue
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        turns = parse_dialogue(row["dialogue"])
        for i, turn in enumerate(turns):
            all_records.append({
                "split": split_name,
                "dataset": row["dataset"],
                "encounter_id": row["encounter_id"],
                "turn_idx": i,
                "speaker": turn["speaker"],
                "text": turn["text"],
            })

out_path = OUT_DIR / "turns.jsonl"
with open(out_path, "w") as f:
    for rec in all_records:
        f.write(json.dumps(rec) + "\n")

print(f"Parsed {len(all_records)} turns across {len(SPLIT_FILES)} splits")
print(f"Saved to {out_path}")

first_id = all_records[0]["encounter_id"]
print(f"\nSample encounter: {first_id}")
for rec in all_records:
    if rec["encounter_id"] == first_id:
        print(f"[{rec['speaker']}] {rec['text']}")