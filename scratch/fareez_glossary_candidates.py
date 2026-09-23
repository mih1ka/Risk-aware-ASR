"""
Throwaway diagnostic — not part of the reproducible pipeline.

Mines fareez gold text for frequent, non-glossary, non-common-English
tokens that look clinically relevant — candidates for manual glossary
tiering, same workflow as the original extract_medical_candidates.py step.
Produces a review list only; does not write to tiered_glossary.csv.
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "eval"))

from align_and_label import load_glossary  # noqa: E402

ENCOUNTER_TEXTS_PATH = ROOT / "data" / "processed" / "encounter_texts.jsonl"

# Inlined from src/glossary/extract_medical_candidates.py (not imported —
# that module runs its own analysis over turns.jsonl as top-level code on
# import, which would pollute this script's output).
with open("/usr/share/dict/words") as f:
    DICTIONARY = set(w.strip().lower() for w in f if w.strip().isalpha())


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

MIN_WORD_LEN = 5
MIN_ENCOUNTERS = 3
TOP_N = 50
MAX_EXAMPLES = 3

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_fareez_encounters():
    encounters = []
    with open(ENCOUNTER_TEXTS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec["dataset"] == "fareez":
                encounters.append(rec)
    return encounters


def main():
    glossary = load_glossary()
    glossary_terms = set(glossary.keys())
    print(f"Loaded glossary: {len(glossary_terms)} terms (excluded from candidates)")

    encounters = load_fareez_encounters()
    print(f"Scanning {len(encounters)} fareez encounters")

    token_counts = defaultdict(int)
    token_encounters = defaultdict(set)
    token_examples = defaultdict(list)

    for rec in encounters:
        eid = rec["encounter_id"]
        full_text = rec["full_text"]
        sentences = SENTENCE_SPLIT.split(full_text)

        for token in full_text.split():
            word = token.strip(".,?!'\";:()").lower()
            if not word.isalpha() or len(word) < MIN_WORD_LEN:
                continue
            if word in glossary_terms:
                continue
            if is_common_word(word):
                continue

            token_counts[word] += 1
            token_encounters[word].add(eid)

            if len(token_examples[word]) < MAX_EXAMPLES:
                pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
                for sent in sentences:
                    if pattern.search(sent) and sent.strip() not in token_examples[word]:
                        token_examples[word].append(sent.strip())
                        if len(token_examples[word]) >= MAX_EXAMPLES:
                            break

    candidates = [
        (word, token_counts[word], len(token_encounters[word]))
        for word in token_counts
        if len(token_encounters[word]) >= MIN_ENCOUNTERS
    ]
    candidates.sort(key=lambda x: -x[1])

    print(f"\n{len(candidates)} candidate terms appear in >= {MIN_ENCOUNTERS} distinct encounters "
          f"(showing top {min(TOP_N, len(candidates))})\n")

    for word, count, n_enc in candidates[:TOP_N]:
        print(f"{count:4d}x  ({n_enc} encounters)  {word}")
        for ex in token_examples[word]:
            print(f"          - {ex}")
        print()


if __name__ == "__main__":
    main()
