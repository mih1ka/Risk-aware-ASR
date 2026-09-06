import re
from pathlib import Path

import pandas as pd
import jellyfish

GLOSSARY_PATH = Path("data/glossary/tiered_glossary.csv")
LABELS_PATH = Path("data/processed/word_level_labels.csv")


def normalize(word):
    return re.sub(r"[^a-z0-9']", "", str(word).lower())


def load_glossary():
    df = pd.read_csv(GLOSSARY_PATH)
    return dict(zip(df["term"].str.lower(), df["tier"]))


def candidate_similarity(word, term):
    word_n, term_n = normalize(word), normalize(term)
    if not word_n or not term_n:
        return 0.0
    max_len = max(len(word_n), len(term_n))
    edit_sim = 1 - (jellyfish.levenshtein_distance(word_n, term_n) / max_len)
    phonetic_bonus = 0.5 if jellyfish.metaphone(word_n) == jellyfish.metaphone(term_n) else 0.0
    return edit_sim + phonetic_bonus


def classify_error(row, glossary):
    error_type = row["error_type"]

    if error_type == "correct":
        return "correct"
    if error_type == "deletion":
        return "omission"
    if error_type == "insertion":
        return "insertion"

    gold, asr_word = row["gold_word"], row["whisper_word"]
    if pd.isna(gold) or pd.isna(asr_word):
        return "insertion" if pd.isna(gold) else "omission"

    if candidate_similarity(asr_word, gold) >= 0.5:
        return "phonetic_substitution"

    asr_norm = normalize(asr_word)
    if asr_norm in glossary and asr_norm != normalize(gold):
        return "plausible_but_wrong"

    return "unrelated_substitution"


def main():
    glossary = load_glossary()
    df = pd.read_csv(LABELS_PATH)
    df["error_category"] = df.apply(lambda r: classify_error(r, glossary), axis=1)

    print("=== Error taxonomy — all words ===")
    print(df["error_category"].value_counts())
    print()

    critical = df[df["tier"].isin([2, 3])]
    print("=== Error taxonomy — critical (Tier 2/3) words only ===")
    print(critical["error_category"].value_counts())
    print()

    print("=== Proportions among critical ERRORS only (excludes 'correct') ===")
    critical_errors = critical[critical["error_category"] != "correct"]
    print((critical_errors["error_category"].value_counts(normalize=True) * 100).round(1))

    out_path = Path("data/processed/word_level_labels_taxonomy.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved full taxonomy-labeled dataset to {out_path}")


if __name__ == "__main__":
    main()