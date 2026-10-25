"""
R2 - production-ASR tier error rates, old vs new.

src/eval/validate_real_asr.py loads data/glossary/tiered_glossary.csv, which
has grown from 176 to 190 terms since data/processed/real_asr_word_labels.csv
was last generated (commit e30d20c, 176 terms). A plain re-run therefore
changes two things at once. This script separates them in memory, by
swapping the module's normalize/tokenize and GLOSSARY_PATH, without writing
any labels file:

  old normalize x {176, 190} glossary, new normalize x {176, 190} glossary.

It also re-reads the stored labels file (the "old" numbers as committed).
Writes r2_tier_rates.csv (aggregates only).
"""
import contextlib
import io
import sys

import pandas as pd

from common import ROOT, OUT_DIR, GLOSSARY_190, GLOSSARY_176

sys.path.insert(0, str(ROOT / "src" / "eval"))
import align_and_label  # noqa: E402
import validate_real_asr as v  # noqa: E402

OLD_NORMALIZE, OLD_TOKENIZE = v.normalize, v.tokenize


def tier_summary(df, label):
    gold = df[df["gold_word"].notna()]
    rows = []
    for t in (1, 2, 3):
        g = gold[gold["tier"] == t]
        rows.append({"condition": label, "tier": t, "gold_words": len(g),
                     "errors": int(g["is_error"].sum()),
                     "error_rate_pct": 100 * g["is_error"].mean()})
    out = pd.DataFrame(rows)
    out["encounters"] = df["encounter_id"].nunique()
    out["gold_words_total"] = int(df["gold_word"].notna().sum())
    return out


def run_in_memory(normalize_fn, tokenize_fn, glossary_path):
    v.normalize, v.tokenize, v.GLOSSARY_PATH = normalize_fn, tokenize_fn, glossary_path
    glossary = v.load_glossary()
    rows = []
    for asr_path, human_path in v.FILE_PAIRS:
        merged = pd.read_csv(ROOT / asr_path).merge(pd.read_csv(ROOT / human_path), on="id",
                                                    suffixes=("_asr", "_human"))
        source = asr_path.stem
        for _, row in merged.iterrows():
            asr_raw = v.tokenize(v.extract_text(row["dialogue_asr"]))
            human_raw = v.tokenize(v.extract_text(row["dialogue_human"]))
            v.align_pair(f"{source}_{row['id']}", human_raw, asr_raw,
                         [v.normalize(w) for w in human_raw], [v.normalize(w) for w in asr_raw],
                         glossary, rows, source)
    return pd.DataFrame(rows)


def main():
    stored_path = ROOT / "data" / "processed" / "real_asr_word_labels.csv"
    frames = [tier_summary(pd.read_csv(stored_path, keep_default_na=False, na_values=[""]),
                           "stored file (as committed)")]
    for norm_name, nfn, tfn in [("old normalize", OLD_NORMALIZE, OLD_TOKENIZE),
                                ("new normalize", align_and_label.normalize, align_and_label.tokenize)]:
        for gname, gpath in [("176", GLOSSARY_176), ("190", GLOSSARY_190)]:
            with contextlib.redirect_stdout(io.StringIO()):
                df = run_in_memory(nfn, tfn, gpath)
            frames.append(tier_summary(df, f"{norm_name}, {gname}-term glossary"))
    out = pd.concat(frames, ignore_index=True)
    out.round(4).to_csv(OUT_DIR / "r2_tier_rates.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
