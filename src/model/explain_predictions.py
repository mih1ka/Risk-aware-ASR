"""
Regenerates data/processed/risk_scores_explained.csv -- the pre-scored
held-out ACI-Bench encounters the demo's "saved sample" mode shows -- with the
paper's configuration: the tier-free classifier (confidence + word length)
and the 176-term glossary. Tier enters only via the multiplicative weight.

Input rows (encounter, gold/ASR word, confidence, error label) come from
data/processed/risk_scores_test.csv, the 42-encounter held-out split
(GroupShuffleSplit, test_size=0.2, random_state=42) that the no-tier model
was NOT trained on. Also prints the flagging threshold at the top-10% operating
point and its precision / critical-error recall on these words.

Run from the repo root:  python src/model/explain_predictions.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.pipeline.risk_pipeline import (  # noqa: E402
    MODEL_PATH, FEATURE_NAMES, TIER_WEIGHTS, load_glossary, build_phonetic_map,
    lookup_tier, normalize,
)

RISK_SCORES_PATH = Path("data/processed/risk_scores_test.csv")
OUT_PATH = Path("data/processed/risk_scores_explained.csv")
KEEP = ["encounter_id", "gold_word", "whisper_word", "confidence", "error_type", "is_error"]
TOP_N_EXAMPLES = 10


def main():
    model = joblib.load(MODEL_PATH)
    glossary = load_glossary()
    phonetic_map = build_phonetic_map(glossary)

    df = pd.read_csv(RISK_SCORES_PATH, keep_default_na=False, na_values=[""])[KEEP].copy()
    df = df[df["whisper_word"].notna()].reset_index(drop=True)
    df["confidence"] = df["confidence"].fillna(0.0)
    # Gold tier: exact glossary match on the reference word (as in align_and_label.py).
    df["tier"] = df["gold_word"].apply(
        lambda w: glossary.get(normalize(w), 1) if pd.notna(w) and normalize(w) else np.nan)
    # Inference-time tier: phonetic lookup on the ASR word (no reference used).
    df["tier_whisper"] = df["whisper_word"].apply(lambda w: lookup_tier(w, glossary, phonetic_map))
    df["word_len"] = df["whisper_word"].astype(str).str.len()

    X = df[FEATURE_NAMES]
    df["learned_uncertainty"] = model.predict_proba(X)[:, 1]
    df["tier_weight"] = df["tier_whisper"].map(TIER_WEIGHTS)
    df["risk_score"] = df["learned_uncertainty"] * df["tier_weight"]

    contribs = model.get_booster().predict(xgb.DMatrix(X, feature_names=FEATURE_NAMES), pred_contribs=True)
    contrib_df = pd.DataFrame(contribs, columns=FEATURE_NAMES + ["bias"]).add_prefix("shap_")
    df = pd.concat([df, contrib_df], axis=1)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} scored words ({df.encounter_id.nunique()} encounters) to {OUT_PATH}")

    thr = float(np.quantile(df["risk_score"], 0.90))
    flagged = df["risk_score"] >= thr
    crit_err = df["tier"].isin([2, 3]) & (df["is_error"] == 1)
    print(f"Top-10% threshold: {thr:.4f}  (flags {flagged.mean()*100:.2f}% of words)")
    print(f"  precision {df.loc[flagged, 'is_error'].mean():.3f}  "
          f"critical-error recall {(flagged & crit_err).sum()}/{crit_err.sum()} = "
          f"{(flagged & crit_err).sum() / crit_err.sum():.3f}")

    print(f"\n=== Top {TOP_N_EXAMPLES} highest-risk words ===")
    for _, r in df.sort_values("risk_score", ascending=False).head(TOP_N_EXAMPLES).iterrows():
        print(f"'{r.whisper_word}' (true: '{r.gold_word}')  risk={r.risk_score:.3f}  "
              f"u={r.learned_uncertainty:.3f}  tier={int(r.tier_whisper)}")


if __name__ == "__main__":
    main()
