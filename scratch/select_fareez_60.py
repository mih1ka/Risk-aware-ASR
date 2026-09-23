"""
Throwaway diagnostic — not part of the reproducible pipeline.

Selects ~60 fareez encounters for the full real-ASR test run:
prioritizes encounters containing >=1 Tier 2/3 glossary term, fills
remaining slots with an evenly-spread sample of Tier-1-only encounters.
Excludes the 3 already piloted (MSK0006, MSK0049, MSK0007).

Writes scratch/fareez_60_gold_text.json (encounter_id -> full_text)
and prints the chosen encounter_ids. Does not touch align_and_label.py,
its stored results, or the glossary.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "eval"))
from align_and_label import load_glossary, normalize, tokenize  # noqa: E402

ENCOUNTER_TEXTS_PATH = ROOT / "data" / "processed" / "encounter_texts.jsonl"
OUT_PATH = Path(__file__).resolve().parent / "fareez_60_gold_text.json"

ALREADY_PILOTED = {"MSK0006", "MSK0049", "MSK0007"}
TARGET_TOTAL = 60


def load_fareez_encounters():
    encounters = []
    with open(ENCOUNTER_TEXTS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec["dataset"] == "fareez" and rec["encounter_id"] not in ALREADY_PILOTED:
                encounters.append(rec)
    return encounters


def distinct_tier23_terms(full_text, glossary):
    terms = set()
    for token in tokenize(full_text):
        norm = normalize(token)
        tier = glossary.get(norm, 1)
        if tier in (2, 3):
            terms.add(norm)
    return terms


def main():
    glossary = load_glossary()
    encounters = load_fareez_encounters()
    print(f"Eligible fareez encounters (excl. 3 piloted): {len(encounters)}")

    tier23 = []
    for rec in encounters:
        terms = distinct_tier23_terms(rec["full_text"], glossary)
        if terms:
            tier23.append((rec, len(terms)))

    print(f"Tier 2/3-containing encounters: {len(tier23)}")

    # Most distinct Tier 2/3 terms first; ties broken by encounter_id for determinism.
    tier23.sort(key=lambda pair: (-pair[1], pair[0]["encounter_id"]))
    selected = [rec for rec, _ in tier23[:TARGET_TOTAL]]
    selected_counts = {rec["encounter_id"]: n for rec, n in tier23[:TARGET_TOTAL]}

    selected.sort(key=lambda r: r["encounter_id"])

    gold_text_map = {r["encounter_id"]: r["full_text"] for r in selected}
    with open(OUT_PATH, "w") as f:
        json.dump(gold_text_map, f, indent=2)

    print(f"\nSelected: {len(selected)} (all Tier 2/3-containing, ranked by distinct-term count, "
          f"range {min(selected_counts.values())}-{max(selected_counts.values())} distinct terms)")
    print(f"Written to {OUT_PATH.relative_to(ROOT)}")

    print(f"\nChosen encounter_ids ({len(selected)}):")
    for r in selected:
        print(r["encounter_id"])


if __name__ == "__main__":
    main()
