"""
Throwaway diagnostic — not part of the reproducible pipeline.

Reruns align_and_label.py's actual align_files() logic (imported, not
reimplemented) against the 207 original aci-bench whisper_out files
only (fareez-prefixed files excluded), with the patched normalize()
(ok/okay fix) now live in align_and_label.py. Writes to a scratch
output, does NOT touch data/processed/word_level_labels.csv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "eval"))
from align_and_label import align_files, load_glossary  # noqa: E402

WHISPER_DIR = ROOT / "data" / "whisper_out"
OUT_PATH = Path(__file__).resolve().parent / "word_level_labels_aci_bench_renormalized.csv"

FAREEZ_PREFIXES = ("MSK", "RES", "CAR", "DER", "GAS")


def main():
    glossary = load_glossary()

    all_files = sorted(WHISPER_DIR.glob("*.json"))
    aci_files = [f for f in all_files if not f.stem.startswith(FAREEZ_PREFIXES)]
    print(f"Total whisper_out files: {len(all_files)}")
    print(f"ACI-Bench files (fareez excluded): {len(aci_files)}\n")

    df, wer_df = align_files(aci_files, glossary)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH.relative_to(ROOT)} (scratch only, word_level_labels.csv untouched)\n")

    gold_rows = df[df["gold_word"].notna()]
    new_summary = gold_rows.groupby("tier")["is_error"].agg(["mean", "count"])
    print("NEW (patched normalize, ok/okay fix) tier error rates:")
    print(new_summary)

    print(f"\nMean per-encounter WER: {wer_df['wer'].mean():.4f}")

    old = {1: 0.126249, 2: 0.673563, 3: 0.901691}
    print("\n" + "=" * 60)
    print("Comparison: old (stored) vs new (patched normalize)")
    print("=" * 60)
    print(f"{'Tier':<6}{'Old':>10}{'New':>10}{'Delta':>12}")
    for t in (1, 2, 3):
        new_val = new_summary.loc[float(t), "mean"]
        delta = new_val - old[t]
        print(f"{t:<6}{old[t]*100:>9.2f}%{new_val*100:>9.2f}%{delta*100:>+11.3f}pp")


if __name__ == "__main__":
    main()
