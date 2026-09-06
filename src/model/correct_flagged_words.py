import re
from pathlib import Path

import pandas as pd
import jellyfish

GLOSSARY_PATH = Path("data/glossary/tiered_glossary.csv")
RISK_SCORES_PATH = Path("data/processed/risk_scores_test.csv")
TOP_K = 3


def normalize(word):
    return re.sub(r"[^a-z0-9']", "", str(word).lower())


def load_glossary_terms():
    df = pd.read_csv(GLOSSARY_PATH)
    return df["term"].str.lower().tolist()


def candidate_similarity(word, term):
    word_n, term_n = normalize(word), normalize(term)
    if not word_n or not term_n:
        return 0.0
    max_len = max(len(word_n), len(term_n))
    edit_sim = 1 - (jellyfish.levenshtein_distance(word_n, term_n) / max_len)
    phonetic_bonus = 0.5 if jellyfish.metaphone(word_n) == jellyfish.metaphone(term_n) else 0.0
    return edit_sim + phonetic_bonus


def suggest_corrections(word, candidates, top_k=TOP_K):
    scored = [(term, candidate_similarity(word, term)) for term in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _ in scored[:top_k]]

CONFIDENCE_THRESHOLD = 0.5

def suggest_corrections_gated(word, candidates, top_k=TOP_K, threshold=CONFIDENCE_THRESHOLD):
    scored = [(term, candidate_similarity(word, term)) for term in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored or scored[0][1] < threshold:
        return None  # not confident enough — escalate to mandatory human review
    return [term for term, _ in scored[:top_k]]

def main():
    candidates = load_glossary_terms()
    df = pd.read_csv(RISK_SCORES_PATH)

    target = df[(df["error_type"] == "substitution") & (df["tier"].isin([2, 3]))].copy()
    print(f"Evaluating correction on {len(target)} critical substitution errors\n")

    top1_hits = top3_hits = 0
    near_miss_hits1 = near_miss_hits3 = near_miss_total = 0
    dropout_hits1 = dropout_hits3 = dropout_total = 0
    examples = []

    for _, row in target.iterrows():
        gold = normalize(row["gold_word"])
        whisper_word = row["whisper_word"]
        suggestions = suggest_corrections(whisper_word, candidates)

        hit1 = bool(suggestions) and normalize(suggestions[0]) == gold
        hit3 = any(normalize(s) == gold for s in suggestions)
        top1_hits += hit1
        top3_hits += hit3

        # Was Whisper's output even phonetically related to the true word?
        # A similarity-based corrector can only ever succeed in this case.
        near_miss = candidate_similarity(whisper_word, row["gold_word"]) >= 0.5
        if near_miss:
            near_miss_total += 1
            near_miss_hits1 += hit1
            near_miss_hits3 += hit3
        else:
            dropout_total += 1
            dropout_hits1 += hit1
            dropout_hits3 += hit3

        if len(examples) < 15:
            examples.append((row["gold_word"], whisper_word, suggestions, near_miss))

    n = len(target)
    print(f"Overall — Top-1: {top1_hits/n:.3f}  Top-3: {top3_hits/n:.3f}")
    print(f"(Random-guess top-1 baseline: {1/len(candidates):.3f})\n")

    print(f"Near-miss errors (Whisper output phonetically related to gold): {near_miss_total}")
    if near_miss_total:
        print(f"  Top-1: {near_miss_hits1/near_miss_total:.3f}  Top-3: {near_miss_hits3/near_miss_total:.3f}")
    print(f"Catastrophic dropouts (Whisper output unrelated to gold): {dropout_total}")
    if dropout_total:
        print(f"  Top-1: {dropout_hits1/dropout_total:.3f}  Top-3: {dropout_hits3/dropout_total:.3f}\n")

    print("Sample corrections (gold_word | whisper_word | suggestions | near_miss?):")
    for gold, whisper_word, suggestions, near_miss in examples:
        print(f"  {gold:20s} | {whisper_word:20s} | {suggestions}  [{near_miss}]")
if __name__ == "__main__":
    main()