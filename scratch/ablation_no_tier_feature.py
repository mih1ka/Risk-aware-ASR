"""
Throwaway diagnostic — not part of the reproducible pipeline.

Ablation: retrains the uncertainty classifier WITHOUT the glossary-tier
feature (confidence + word_len only) on the same 80/20 ACI-Bench
train/test split (GroupShuffleSplit, test_size=0.2, random_state=42) used
by src/model/risk_score.py. Compares four word-ranking strategies at
5%/10%/20% flagging budgets, side by side with the original tier-feature
model (loaded from models/uncertainty_xgb.joblib, not retrained).

Saves the no-tier model to models/uncertainty_xgb_no_tier.joblib.
Does not touch models/uncertainty_xgb.joblib, the fareez results, or
the glossary.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import jellyfish
import joblib
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "model"))
from risk_score import load_glossary, build_phonetic_map, lookup_tier, TIER_WEIGHTS  # noqa: E402
from bootstrap_significance import critical_recall_at_pct  # noqa: E402

LABELS_PATH = ROOT / "data" / "processed" / "word_level_labels.csv"
OLD_MODEL_PATH = ROOT / "models" / "uncertainty_xgb.joblib"
NEW_MODEL_PATH = ROOT / "models" / "uncertainty_xgb_no_tier.joblib"

N_BOOTSTRAP = 2000
RNG_SEED = 42
BUDGETS = [0.05, 0.10, 0.20]


def normalize(word):
    return re.sub(r"[^a-z0-9']", "", str(word).lower())


def build_features(df, glossary, phonetic_map):
    df = df[df["whisper_word"].notna()].copy()
    df["tier_whisper"] = df["whisper_word"].apply(lambda w: lookup_tier(w, glossary, phonetic_map))
    df["confidence"] = df["confidence"].fillna(0.0)
    df["word_len"] = df["whisper_word"].astype(str).str.len()
    return df


def recall_at_budget(scores, tiers, is_error, pct):
    return critical_recall_at_pct(np.asarray(scores), np.asarray(tiers), np.asarray(is_error), pct)


def main():
    glossary = load_glossary()
    phonetic_map = build_phonetic_map(glossary)
    raw = pd.read_csv(LABELS_PATH)
    df = build_features(raw, glossary, phonetic_map)

    X_full = df[["confidence", "tier_whisper", "word_len"]]
    X_no_tier = df[["confidence", "word_len"]]
    y = df["is_error"]
    groups = df["encounter_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_full, y, groups))

    test_df = df.iloc[test_idx].copy()
    y_train = y.iloc[train_idx]
    print(f"Train rows: {len(train_idx)}  Test rows: {len(test_idx)}")
    print(f"Train encounters: {groups.iloc[train_idx].nunique()}  Test encounters: {groups.iloc[test_idx].nunique()}\n")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    # --- OLD model: load existing tier-feature-included model, do not retrain/overwrite ---
    old_model = joblib.load(OLD_MODEL_PATH)
    print(f"Loaded existing tier-feature model from {OLD_MODEL_PATH.relative_to(ROOT)}")

    # --- NEW model: retrain without tier feature, save separately ---
    new_model = xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42,
                                   scale_pos_weight=scale_pos_weight)
    new_model.fit(X_no_tier.iloc[train_idx], y_train)
    joblib.dump(new_model, NEW_MODEL_PATH)
    print(f"Saved new tier-free model to {NEW_MODEL_PATH.relative_to(ROOT)}\n")

    # --- Build the four rankings on the test set ---
    test_df["raw_uncertainty"] = 1 - test_df["confidence"]
    test_df["old_learned_uncertainty"] = old_model.predict_proba(X_full.iloc[test_idx])[:, 1]
    test_df["new_learned_uncertainty"] = new_model.predict_proba(X_no_tier.iloc[test_idx])[:, 1]
    test_df["tier_weight"] = test_df["tier_whisper"].map(TIER_WEIGHTS)
    test_df["old_risk_score"] = test_df["old_learned_uncertainty"] * test_df["tier_weight"]
    test_df["new_risk_score"] = test_df["new_learned_uncertainty"] * test_df["tier_weight"]

    tiers = test_df["tier"].to_numpy()
    is_error = test_df["is_error"].to_numpy()

    rankings = {
        "(a) raw 1-confidence": "raw_uncertainty",
        "(b-old) learned uncertainty, WITH tier feature": "old_learned_uncertainty",
        "(b-new) learned uncertainty, tier feature REMOVED": "new_learned_uncertainty",
        "(c) tier weight alone": "tier_weight",
        "(d-old) risk score = (b-old) x tier weight": "old_risk_score",
        "(d-new) risk score = (b-new) x tier weight": "new_risk_score",
    }

    print("=" * 100)
    print("Critical-error recall (gold Tier 2/3) at each flagging budget")
    print("=" * 100)
    header = f"{'ranking':<50}" + "".join(f"{f'{int(p*100)}%':>12}" for p in BUDGETS)
    print(header)
    print("-" * len(header))
    for label, col in rankings.items():
        row = f"{label:<50}"
        for p in BUDGETS:
            r = recall_at_budget(test_df[col].to_numpy(), tiers, is_error, p)
            row += f"{r*100:>11.2f}%"
        print(row)

    # --- Bootstrap CI at 10%: (d-new) risk score vs (a) raw confidence ---
    print("\n" + "=" * 100)
    print("Bootstrap CI at 10% flagging: (d-new) risk_score [no tier feature] vs (a) raw 1-confidence")
    print("Method: encounter-level cluster resampling, 2000 iterations, seed 42 (same as bootstrap_significance.py)")
    print("=" * 100)

    encounter_ids = test_df["encounter_id"].to_numpy()
    score_d = test_df["new_risk_score"].to_numpy()
    score_a = test_df["raw_uncertainty"].to_numpy()

    unique_encounters = np.unique(encounter_ids)
    idx_by_encounter = {eid: np.where(encounter_ids == eid)[0] for eid in unique_encounters}

    rng = np.random.default_rng(RNG_SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        sampled_ids = rng.choice(unique_encounters, size=len(unique_encounters), replace=True)
        idx = np.concatenate([idx_by_encounter[eid] for eid in sampled_ids])

        recall_d = critical_recall_at_pct(score_d[idx], tiers[idx], is_error[idx], 0.10)
        recall_a = critical_recall_at_pct(score_a[idx], tiers[idx], is_error[idx], 0.10)
        if not np.isnan(recall_d) and not np.isnan(recall_a):
            diffs.append(recall_d - recall_a)

    diffs = np.array(diffs)
    print(f"Bootstrap iterations used: {len(diffs)} / {N_BOOTSTRAP}")
    print(f"Mean recall difference (d-new risk_score - a raw_confidence) at top 10%: {diffs.mean()*100:.2f}pp")
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    print(f"95% CI: [{ci_low*100:.2f}pp, {ci_high*100:.2f}pp]")
    pct_positive = (diffs > 0).mean()
    print(f"Fraction of bootstrap samples where risk_score >= raw_confidence: {pct_positive:.3f}")


if __name__ == "__main__":
    main()
