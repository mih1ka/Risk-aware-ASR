"""
Copy of scratch/run_align_fareez272.py for blocker R1, changed only to use
the 176-term glossary (data/glossary/tiered_glossary_176.csv, the 190-term
glossary minus the 14 terms added from Fareez transcripts) and to write a
NEW labels file under experiments/blockers/private/ (git-ignored, it holds
transcript words). Does NOT touch data/processed/word_level_labels_fareez.csv.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "eval"))
import align_and_label  # noqa: E402
from align_and_label import align_files, load_glossary, normalize  # noqa: E402

align_and_label.GLOSSARY_PATH = ROOT / "data" / "glossary" / "tiered_glossary_176.csv"

WHISPER_DIR = ROOT / "data" / "whisper_out"
OUT_PATH = Path(__file__).resolve().parent / "private" / "word_level_labels_fareez_176.csv"
SELECTION_PATH = ROOT / "scratch" / "fareez_272_gold_text.json"


def main():
    selected_ids = set(json.loads(SELECTION_PATH.read_text()).keys())

    json_files = sorted(
        jf for jf in WHISPER_DIR.glob("*.json") if jf.stem in selected_ids
    )
    missing = selected_ids - {jf.stem for jf in json_files}

    print(f"Selected encounters: {len(selected_ids)}")
    print(f"Found in whisper_out/: {len(json_files)}")
    if missing:
        print(f"MISSING (not yet uploaded/run in Colab, excluded from this run): {sorted(missing)}")
    print()

    # Confirm the ok/okay normalize fix is active.
    assert normalize("Okay.") == "ok" and normalize("OK") == "ok", "ok/okay fix not active!"
    print("Confirmed: normalize('Okay.') == normalize('OK') == 'ok' (fix is active in this run)\n")

    glossary = load_glossary()
    print(f"Glossary: {len(glossary)} terms")

    df, wer_df = align_files(json_files, glossary)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} word-level rows to {OUT_PATH.relative_to(ROOT)}")

    # Overall (pooled, not mean-of-means) WER across all encounters combined.
    total_errors = df["is_error"].sum()
    total_ref_words = df["gold_word"].notna().sum()
    overall_wer = total_errors / total_ref_words if total_ref_words else float("nan")
    print(f"\nMean per-encounter WER: {wer_df['wer'].mean():.4f}")
    print(f"Overall pooled WER (total errors / total gold words) across all {len(json_files)} encounters: "
          f"{overall_wer:.4f} ({overall_wer*100:.2f}%)")

    gold_rows = df[df["gold_word"].notna()]
    tier_summary = gold_rows.groupby("tier")["is_error"].agg(["mean", "count", "sum"])
    tier_summary = tier_summary.rename(columns={"mean": "error_rate", "count": "total_tokens", "sum": "error_count"})
    print("\nError rate by glossary tier (1=everyday/unlisted, 2=clinical term, 3=drug/dosage):")
    print(tier_summary)


if __name__ == "__main__":
    main()
