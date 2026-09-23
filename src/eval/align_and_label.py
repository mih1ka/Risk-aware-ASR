import json
import re
from pathlib import Path

import pandas as pd
import jiwer

GLOSSARY_PATH = Path("data/glossary/tiered_glossary.csv")
WHISPER_DIR = Path("data/whisper_out")
OUT_PATH = Path("data/processed/word_level_labels.csv")


def load_glossary():
    df = pd.read_csv(GLOSSARY_PATH)
    return dict(zip(df["term"].str.lower(), df["tier"]))


def normalize(word):
    stripped = re.sub(r"[^a-z0-9']", "", word.lower())
    if stripped == "okay":
        return "ok"
    return stripped


def tokenize(text):
    return [w for w in text.split() if re.search(r"[a-zA-Z0-9]", w)]


def align_files(json_files, glossary):
    rows = []
    per_encounter_wer = []

    for jf in json_files:
        data = json.loads(jf.read_text())
        eid = data["encounter_id"]

        gold_raw = tokenize(data["original_text"])
        gold_norm = [normalize(w) for w in gold_raw]

        hyp_words = data["words"]
        hyp_raw = [w["word"].strip() for w in hyp_words]
        hyp_conf = [w["confidence"] for w in hyp_words]
        hyp_norm = [normalize(w) or "blank" for w in hyp_raw]

        ref_text = " ".join(gold_norm) if gold_norm else "blank"
        hyp_text = " ".join(hyp_norm) if hyp_norm else "blank"

        out = jiwer.process_words(ref_text, hyp_text)
        alignment = out.alignments[0]

        for chunk in alignment:
            if chunk.type == "equal":
                for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    ref_i = chunk.ref_start_idx + offset
                    hyp_i = chunk.hyp_start_idx + offset
                    rows.append({
                        "encounter_id": eid,
                        "gold_word": gold_raw[ref_i],
                        "whisper_word": hyp_raw[hyp_i],
                        "confidence": hyp_conf[hyp_i],
                        "error_type": "correct",
                        "is_error": 0,
                        "tier": glossary.get(gold_norm[ref_i], 1),
                    })
            elif chunk.type == "substitute":
                ref_n = chunk.ref_end_idx - chunk.ref_start_idx
                hyp_n = chunk.hyp_end_idx - chunk.hyp_start_idx
                for offset in range(max(ref_n, hyp_n)):
                    ref_i = chunk.ref_start_idx + offset if offset < ref_n else None
                    hyp_i = chunk.hyp_start_idx + offset if offset < hyp_n else None
                    rows.append({
                        "encounter_id": eid,
                        "gold_word": gold_raw[ref_i] if ref_i is not None else None,
                        "whisper_word": hyp_raw[hyp_i] if hyp_i is not None else None,
                        "confidence": hyp_conf[hyp_i] if hyp_i is not None else None,
                        "error_type": "substitution",
                        "is_error": 1,
                        "tier": glossary.get(gold_norm[ref_i], 1) if ref_i is not None else None,
                    })
            elif chunk.type == "delete":
                for ref_i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    rows.append({
                        "encounter_id": eid,
                        "gold_word": gold_raw[ref_i],
                        "whisper_word": None,
                        "confidence": None,
                        "error_type": "deletion",
                        "is_error": 1,
                        "tier": glossary.get(gold_norm[ref_i], 1),
                    })
            elif chunk.type == "insert":
                for hyp_i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    rows.append({
                        "encounter_id": eid,
                        "gold_word": None,
                        "whisper_word": hyp_raw[hyp_i],
                        "confidence": hyp_conf[hyp_i],
                        "error_type": "insertion",
                        "is_error": 1,
                        "tier": None,
                    })

        per_encounter_wer.append({"encounter_id": eid, "wer": out.wer})

    return pd.DataFrame(rows), pd.DataFrame(per_encounter_wer)


def main():
    glossary = load_glossary()

    json_files = sorted(WHISPER_DIR.glob("*.json"))
    print(f"Found {len(json_files)} encounter files")

    df, wer_df = align_files(json_files, glossary)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} word-level rows to {OUT_PATH}")

    print(f"Mean per-encounter WER: {wer_df['wer'].mean():.4f}")

    gold_rows = df[df["gold_word"].notna()]
    tier_summary = gold_rows.groupby("tier")["is_error"].agg(["mean", "count"])
    print("\nError rate by glossary tier (1=everyday/unlisted, 2=clinical term, 3=drug/dosage):")
    print(tier_summary)


if __name__ == "__main__":
    main()