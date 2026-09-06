import re
from pathlib import Path

import pandas as pd
import jiwer
import jellyfish

DATA_DIR = Path("data/raw_aci_bench/data/src_experiment_data")
GLOSSARY_PATH = Path("data/glossary/tiered_glossary.csv")

# train_* excluded — no matching human-corrected file exists for train in
# either collection mode.
FILE_PAIRS = [
    (DATA_DIR / f"{split}_virtscribe_asr.csv", DATA_DIR / f"{split}_virtscribe_humantrans.csv")
    for split in ["valid", "test1", "test2", "test3"]
] + [
    (DATA_DIR / f"{split}_aci_asr.csv", DATA_DIR / f"{split}_aci_asrcorr.csv")
    for split in ["valid", "test1", "test2", "test3"]
]


def load_glossary():
    df = pd.read_csv(GLOSSARY_PATH)
    return dict(zip(df["term"].str.lower(), df["tier"]))


def extract_text(dialogue):
    parts = re.split(r"\[(?:doctor|patient)\]", str(dialogue))
    return " ".join(p.strip() for p in parts if p.strip())


def normalize(word):
    return re.sub(r"[^a-z0-9']", "", str(word).lower())


def tokenize(text):
    return [w for w in text.split() if re.search(r"[a-zA-Z0-9]", w)]


def candidate_similarity(word, term):
    word_n, term_n = normalize(word), normalize(term)
    if not word_n or not term_n:
        return 0.0
    max_len = max(len(word_n), len(term_n))
    edit_sim = 1 - (jellyfish.levenshtein_distance(word_n, term_n) / max_len)
    phonetic_bonus = 0.5 if jellyfish.metaphone(word_n) == jellyfish.metaphone(term_n) else 0.0
    return edit_sim + phonetic_bonus


def suggest_corrections(word, candidates, top_k=3):
    scored = [(term, candidate_similarity(word, term)) for term in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _ in scored[:top_k]]


def align_pair(eid, human_raw, asr_raw, human_norm, asr_norm, glossary, rows, source):
    ref_text = " ".join(human_norm) if human_norm else "blank"
    hyp_text = " ".join(asr_norm) if asr_norm else "blank"
    out = jiwer.process_words(ref_text, hyp_text)
    alignment = out.alignments[0]

    for chunk in alignment:
        if chunk.type == "equal":
            for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ref_i = chunk.ref_start_idx + offset
                hyp_i = chunk.hyp_start_idx + offset
                rows.append({"encounter_id": eid, "source": source, "gold_word": human_raw[ref_i],
                             "asr_word": asr_raw[hyp_i], "error_type": "correct",
                             "is_error": 0, "tier": glossary.get(human_norm[ref_i], 1)})
        elif chunk.type == "substitute":
            ref_n = chunk.ref_end_idx - chunk.ref_start_idx
            hyp_n = chunk.hyp_end_idx - chunk.hyp_start_idx
            for offset in range(max(ref_n, hyp_n)):
                ref_i = chunk.ref_start_idx + offset if offset < ref_n else None
                hyp_i = chunk.hyp_start_idx + offset if offset < hyp_n else None
                rows.append({"encounter_id": eid, "source": source,
                             "gold_word": human_raw[ref_i] if ref_i is not None else None,
                             "asr_word": asr_raw[hyp_i] if hyp_i is not None else None,
                             "error_type": "substitution", "is_error": 1,
                             "tier": glossary.get(human_norm[ref_i], 1) if ref_i is not None else None})
        elif chunk.type == "delete":
            for ref_i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                rows.append({"encounter_id": eid, "source": source, "gold_word": human_raw[ref_i],
                             "asr_word": None, "error_type": "deletion", "is_error": 1,
                             "tier": glossary.get(human_norm[ref_i], 1)})
        elif chunk.type == "insert":
            for hyp_i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                rows.append({"encounter_id": eid, "source": source, "gold_word": None,
                             "asr_word": asr_raw[hyp_i], "error_type": "insertion",
                             "is_error": 1, "tier": None})


def main():
    glossary = load_glossary()
    candidates = list(glossary.keys())
    rows = []
    total_matched = 0

    for asr_path, human_path in FILE_PAIRS:
        if not asr_path.exists() or not human_path.exists():
            print(f"Skipping missing pair: {asr_path.name} / {human_path.name}")
            continue

        asr_df = pd.read_csv(asr_path)
        human_df = pd.read_csv(human_path)
        merged = asr_df.merge(human_df, on="id", suffixes=("_asr", "_human"))
        total_matched += len(merged)
        source = asr_path.stem
        print(f"{source}: matched {len(merged)} encounters")

        for _, row in merged.iterrows():
            eid = f"{source}_{row['id']}"
            asr_raw = tokenize(extract_text(row["dialogue_asr"]))
            human_raw = tokenize(extract_text(row["dialogue_human"]))
            asr_norm = [normalize(w) for w in asr_raw]
            human_norm = [normalize(w) for w in human_raw]
            align_pair(eid, human_raw, asr_raw, human_norm, asr_norm, glossary, rows, source)

    print(f"\nTotal matched encounters across all real ASR/human pairs: {total_matched}\n")

    df = pd.DataFrame(rows)
    out_path = Path("data/processed/real_asr_word_labels.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} word-level rows to {out_path}\n")

    gold_rows = df[df["gold_word"].notna()]
    tier_summary = gold_rows.groupby("tier")["is_error"].agg(["mean", "count"])
    print("Error rate by glossary tier on REAL ASR data (all splits/modes combined):")
    print(tier_summary)
    print()

    target = df[(df["error_type"] == "substitution") & (df["tier"].isin([2, 3]))].copy()
    print(f"Evaluating correction on {len(target)} real critical substitution errors\n")
    if len(target) == 0:
        print("No critical substitution errors found — nothing to evaluate.")
        return

    top1_hits = top3_hits = 0
    near_miss_hits1 = near_miss_total = 0
    dropout_hits1 = dropout_total = 0

    for _, row in target.iterrows():
        gold = normalize(row["gold_word"])
        asr_word = row["asr_word"]
        suggestions = suggest_corrections(asr_word, candidates)
        hit1 = bool(suggestions) and normalize(suggestions[0]) == gold
        hit3 = any(normalize(s) == gold for s in suggestions)
        top1_hits += hit1
        top3_hits += hit3

        near_miss = candidate_similarity(asr_word, row["gold_word"]) >= 0.5
        if near_miss:
            near_miss_total += 1
            near_miss_hits1 += hit1
        else:
            dropout_total += 1
            dropout_hits1 += hit1

    n = len(target)
    print(f"Overall — Top-1: {top1_hits/n:.3f}  Top-3: {top3_hits/n:.3f}")
    if near_miss_total:
        print(f"Near-miss (n={near_miss_total}): Top-1 {near_miss_hits1/near_miss_total:.3f}")
    if dropout_total:
        print(f"Dropout (n={dropout_total}): Top-1 {dropout_hits1/dropout_total:.3f}")


if __name__ == "__main__":
    main()