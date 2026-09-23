"""
Throwaway diagnostic — not part of the reproducible pipeline.

Sweeps the Tier 3 risk-weight parameter (holding Tier 1=1.0, Tier 2=3.25
fixed) and reports critical-error recall at 5/10/20% flagging budgets,
using the tier-removed classifier (models/uncertainty_xgb_no_tier.joblib,
loaded read-only, not retrained). Bootstrap CI vs. raw 1-confidence at
10% for every sweep point (same method as bootstrap_significance.py).

Read-only: does not touch tiered_glossary.csv or any stored model.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "model"))
from risk_score import load_glossary, build_phonetic_map, lookup_tier  # noqa: E402
from bootstrap_significance import critical_recall_at_pct  # noqa: E402

from sklearn.model_selection import GroupShuffleSplit

LABELS_PATH = ROOT / "data" / "processed" / "word_level_labels.csv"
NO_TIER_MODEL_PATH = ROOT / "models" / "uncertainty_xgb_no_tier.joblib"

N_BOOTSTRAP = 2000
RNG_SEED = 42
BUDGETS = [0.05, 0.10, 0.20]
TIER3_SWEEP = [2.0, 3.0, 4.25, 5.5, 7.0, 8.0]
TIER1_WEIGHT = 1.0
TIER2_WEIGHT = 3.25


def build_features(df, glossary, phonetic_map):
    df = df[df["whisper_word"].notna()].copy()
    df["tier_whisper"] = df["whisper_word"].apply(lambda w: lookup_tier(w, glossary, phonetic_map))
    df["confidence"] = df["confidence"].fillna(0.0)
    df["word_len"] = df["whisper_word"].astype(str).str.len()
    return df


def main():
    glossary = load_glossary()
    phonetic_map = build_phonetic_map(glossary)
    raw = pd.read_csv(LABELS_PATH)
    df = build_features(raw, glossary, phonetic_map)

    X_no_tier = df[["confidence", "word_len"]]
    y = df["is_error"]
    groups = df["encounter_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_no_tier, y, groups))

    test_df = df.iloc[test_idx].copy()
    print(f"Test rows: {len(test_idx)}  Test encounters: {groups.iloc[test_idx].nunique()}\n")

    model = joblib.load(NO_TIER_MODEL_PATH)
    print(f"Loaded tier-removed classifier from {NO_TIER_MODEL_PATH.relative_to(ROOT)} (read-only, not retrained)\n")

    test_df["learned_uncertainty"] = model.predict_proba(X_no_tier.iloc[test_idx])[:, 1]
    test_df["raw_uncertainty"] = 1 - test_df["confidence"]

    tiers = test_df["tier"].to_numpy()
    is_error = test_df["is_error"].to_numpy()
    encounter_ids = test_df["encounter_id"].to_numpy()
    unique_encounters = np.unique(encounter_ids)
    idx_by_encounter = {eid: np.where(encounter_ids == eid)[0] for eid in unique_encounters}
    score_a = test_df["raw_uncertainty"].to_numpy()
    learned_unc = test_df["learned_uncertainty"].to_numpy()

    tier_map_base = {1: TIER1_WEIGHT, 2: TIER2_WEIGHT}

    print("=" * 100)
    print("Critical-error recall (gold Tier 2/3) across the Tier 3 weight sweep")
    print("=" * 100)
    header = f"{'Tier3 weight':<14}" + "".join(f"{f'{int(p*100)}%':>12}" for p in BUDGETS)
    print(header)
    print("-" * len(header))

    sweep_scores = {}
    for w3 in TIER3_SWEEP:
        weight_map = {**tier_map_base, 3: w3}
        tier_weight = test_df["tier_whisper"].map(weight_map).to_numpy()
        risk_score = learned_unc * tier_weight
        sweep_scores[w3] = risk_score

        row = f"{w3:<14}"
        for p in BUDGETS:
            r = critical_recall_at_pct(risk_score, tiers, is_error, p)
            row += f"{r*100:>11.2f}%"
        print(row)

    print("\n" + "=" * 100)
    print("Bootstrap CI at 10% flagging: risk_score(tier3_weight) vs raw 1-confidence")
    print("Method: encounter-level cluster resampling, 2000 iterations, seed 42")
    print("=" * 100)
    header2 = f"{'Tier3 weight':<14}{'mean diff':>12}{'95% CI':>24}{'frac favor risk':>18}{'resamples':>12}"
    print(header2)
    print("-" * len(header2))

    for w3 in TIER3_SWEEP:
        score_d = sweep_scores[w3]
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
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        frac_favor = (diffs > 0).mean()
        print(f"{w3:<14}{diffs.mean()*100:>11.2f}pp{'[' + f'{ci_low*100:.2f}, {ci_high*100:.2f}' + ']pp':>24}"
              f"{frac_favor:>18.3f}{len(diffs):>12} / {N_BOOTSTRAP}")


if __name__ == "__main__":
    main()
