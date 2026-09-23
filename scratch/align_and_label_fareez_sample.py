"""
Throwaway diagnostic — not part of the reproducible pipeline.

Runs align_and_label.py's exact alignment logic (imported, not reimplemented)
against only the 3 fareez whisper_out files (MSK0006, MSK0049, MSK0007) to
sanity-check real-audio Whisper output before scaling up. Does NOT touch
data/processed/word_level_labels.csv — prints a report only.
"""
import json
import sys
from pathlib import Path

import jiwer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "eval"))
from align_and_label import load_glossary, normalize, tokenize  # noqa: E402

WHISPER_DIR = ROOT / "data" / "whisper_out"
TARGET_IDS = ["MSK0006", "MSK0049", "MSK0007"]
MAX_EXAMPLES = 10


FILLERS = {"uh", "um"}


def analyze_encounter(eid, glossary, exclude_fillers=False):
    data = json.loads((WHISPER_DIR / f"{eid}.json").read_text())
    assert data["encounter_id"] == eid

    gold_raw = tokenize(data["original_text"])
    gold_norm = [normalize(w) for w in gold_raw]

    hyp_words = data["words"]
    hyp_raw = [w["word"].strip() for w in hyp_words]
    hyp_conf = [w["confidence"] for w in hyp_words]
    hyp_norm = [normalize(w) or "blank" for w in hyp_raw]

    if exclude_fillers:
        keep = [i for i, n in enumerate(gold_norm) if n not in FILLERS]
        gold_raw = [gold_raw[i] for i in keep]
        gold_norm = [gold_norm[i] for i in keep]
        keep_h = [i for i, n in enumerate(hyp_norm) if n not in FILLERS]
        hyp_raw = [hyp_raw[i] for i in keep_h]
        hyp_conf = [hyp_conf[i] for i in keep_h]
        hyp_norm = [hyp_norm[i] for i in keep_h]

    ref_text = " ".join(gold_norm) if gold_norm else "blank"
    hyp_text = " ".join(hyp_norm) if hyp_norm else "blank"

    out = jiwer.process_words(ref_text, hyp_text)
    alignment = out.alignments[0]

    rows = []
    for chunk in alignment:
        if chunk.type == "equal":
            for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ref_i = chunk.ref_start_idx + offset
                hyp_i = chunk.hyp_start_idx + offset
                rows.append({
                    "gold_word": gold_raw[ref_i], "whisper_word": hyp_raw[hyp_i],
                    "confidence": hyp_conf[hyp_i], "error_type": "correct", "is_error": 0,
                    "tier": glossary.get(gold_norm[ref_i], 1),
                })
        elif chunk.type == "substitute":
            ref_n = chunk.ref_end_idx - chunk.ref_start_idx
            hyp_n = chunk.hyp_end_idx - chunk.hyp_start_idx
            for offset in range(max(ref_n, hyp_n)):
                ref_i = chunk.ref_start_idx + offset if offset < ref_n else None
                hyp_i = chunk.hyp_start_idx + offset if offset < hyp_n else None
                rows.append({
                    "gold_word": gold_raw[ref_i] if ref_i is not None else None,
                    "whisper_word": hyp_raw[hyp_i] if hyp_i is not None else None,
                    "confidence": hyp_conf[hyp_i] if hyp_i is not None else None,
                    "error_type": "substitution", "is_error": 1,
                    "tier": glossary.get(gold_norm[ref_i], 1) if ref_i is not None else None,
                })
        elif chunk.type == "delete":
            for ref_i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                rows.append({
                    "gold_word": gold_raw[ref_i], "whisper_word": None, "confidence": None,
                    "error_type": "deletion", "is_error": 1,
                    "tier": glossary.get(gold_norm[ref_i], 1),
                })
        elif chunk.type == "insert":
            for hyp_i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                rows.append({
                    "gold_word": None, "whisper_word": hyp_raw[hyp_i],
                    "confidence": hyp_conf[hyp_i], "error_type": "insertion", "is_error": 1,
                    "tier": None,
                })

    return out.wer, rows


def main():
    glossary = load_glossary()
    print(f"Glossary: {len(glossary)} terms\n")

    for exclude_fillers in (False, True):
        label = "WITH fillers excluded (uh/um dropped from gold+hyp)" if exclude_fillers else "AS-IS (fillers included, ok/okay normalize fix applied)"
        print("#" * 80)
        print(f"# {label}")
        print("#" * 80)
        run_report(glossary, exclude_fillers)


def run_report(glossary, exclude_fillers):
    for eid in TARGET_IDS:
        wer, rows = analyze_encounter(eid, glossary, exclude_fillers)
        print("=" * 80)
        print(f"{eid}  —  WER: {wer:.4f} ({wer*100:.2f}%)")

        tier_total = {1: 0, 2: 0, 3: 0}
        tier_errors = {1: 0, 2: 0, 3: 0}
        for r in rows:
            if r["tier"] is None or r["gold_word"] is None:
                continue
            tier_total[r["tier"]] += 1
            tier_errors[r["tier"]] += r["is_error"]

        print("Tier breakdown (gold-word-anchored rows only, i.e. excludes pure insertions):")
        for t in (1, 2, 3):
            total = tier_total[t]
            errs = tier_errors[t]
            rate = 100.0 * errs / total if total else 0.0
            print(f"  Tier {t}: {errs}/{total} errors ({rate:.2f}%)")

        mismatches = [r for r in rows if r["is_error"] == 1]
        print(f"\nTotal error rows: {len(mismatches)} / {len(rows)} total rows")
        print(f"Example mismatches (up to {MAX_EXAMPLES}):")
        for r in mismatches[:MAX_EXAMPLES]:
            print(f"  [{r['error_type']:12s}] gold={r['gold_word']!r:20s} whisper={r['whisper_word']!r:20s} "
                  f"conf={r['confidence']!r} tier={r['tier']}")
        print()


if __name__ == "__main__":
    main()
