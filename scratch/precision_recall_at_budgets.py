"""
Throwaway diagnostic — not part of the reproducible pipeline.

Computes precision AND recall at 5/10/20% flagging budgets, for both
raw_confidence and risk_score rankings, on ACI-Bench (test split of
word_level_labels.csv) and Fareez (all of word_level_labels_fareez.csv,
never used for training) separately.

Uses the current no-tier-feature classifier (models/uncertainty_xgb_no_tier.joblib,
read-only) consistently across both datasets, so the risk_score ranking is
directly comparable between them.

Precision @ budget = (critical errors caught among flagged words) /
                      (total words flagged at that budget)
Recall    @ budget = (critical errors caught among flagged words) /
                      (total critical errors in the dataset)
"critical" = gold tier in {2, 3}.

Read-only: touches no stored model, no glossary, no fareez source files.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "model"))
from risk_score import load_glossary, build_phonetic_map, lookup_tier, TIER_WEIGHTS  # noqa: E402

ACI_LABELS_PATH = ROOT / "data" / "processed" / "word_level_labels.csv"
FAREEZ_LABELS_PATH = ROOT / "data" / "processed" / "word_level_labels_fareez.csv"
NO_TIER_MODEL_PATH = ROOT / "models" / "uncertainty_xgb_no_tier.joblib"

BUDGETS = [0.05, 0.10, 0.20]


def build_features(df, glossary, phonetic_map):
    df = df[df["whisper_word"].notna()].copy()
    df["tier_whisper"] = df["whisper_word"].apply(lambda w: lookup_tier(w, glossary, phonetic_map))
    df["confidence"] = df["confidence"].fillna(0.0)
    df["word_len"] = df["whisper_word"].astype(str).str.len()
    return df


def precision_recall_at_pct(scores, tiers, is_error, pct):
    n = len(scores)
    n_flag = int(n * pct)
    order = np.argsort(-scores)
    flagged_idx = order[:n_flag]

    critical_mask = np.isin(tiers, [2, 3])
    n_critical_errors = is_error[critical_mask].sum()

    flagged_critical_mask = critical_mask[flagged_idx]
    caught = is_error[flagged_idx][flagged_critical_mask].sum()

    precision = caught / n_flag if n_flag else np.nan
    recall = caught / n_critical_errors if n_critical_errors else np.nan
    return precision, recall, n_flag, caught, n_critical_errors


def report(label, test_df, model, X_cols):
    test_df = test_df.copy()
    test_df["learned_uncertainty"] = model.predict_proba(test_df[X_cols])[:, 1]
    test_df["raw_uncertainty"] = 1 - test_df["confidence"]
    test_df["tier_weight"] = test_df["tier_whisper"].map(TIER_WEIGHTS)
    test_df["risk_score"] = test_df["learned_uncertainty"] * test_df["tier_weight"]

    tiers = test_df["tier"].to_numpy()
    is_error = test_df["is_error"].to_numpy()

    print("=" * 100)
    print(f"{label}  (n={len(test_df)} rows)")
    print("=" * 100)
    header = f"{'ranking':<16}{'budget':<10}{'precision':>12}{'recall':>12}{'n_flagged':>12}{'caught':>10}{'total_crit':>12}"
    print(header)
    print("-" * len(header))

    results = {}
    for ranking_name, col in [("raw_confidence", "raw_uncertainty"), ("risk_score", "risk_score")]:
        scores = test_df[col].to_numpy()
        for p in BUDGETS:
            prec, rec, n_flag, caught, n_crit = precision_recall_at_pct(scores, tiers, is_error, p)
            print(f"{ranking_name:<16}{f'{int(p*100)}%':<10}{prec*100:>11.2f}%{rec*100:>11.2f}%{n_flag:>12}{caught:>10}{n_crit:>12}")
            results[(ranking_name, p)] = (prec, rec, n_flag, caught, n_crit)
    print()
    return results


def main():
    glossary = load_glossary()
    phonetic_map = build_phonetic_map(glossary)
    model = joblib.load(NO_TIER_MODEL_PATH)
    print(f"Loaded {NO_TIER_MODEL_PATH.relative_to(ROOT)} (read-only)\n")

    # --- ACI-Bench: same 80/20 split as prior ablations, test set only ---
    aci_raw = pd.read_csv(ACI_LABELS_PATH)
    aci_df = build_features(aci_raw, glossary, phonetic_map)
    X_no_tier = aci_df[["confidence", "word_len"]]
    y = aci_df["is_error"]
    groups = aci_df["encounter_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(splitter.split(X_no_tier, y, groups))
    aci_test_df = aci_df.iloc[test_idx]

    aci_results = report("ACI-BENCH (80/20 split test set)", aci_test_df, model, ["confidence", "word_len"])

    # --- Fareez: every encounter in the labels file, never used for training ---
    fareez_raw = pd.read_csv(FAREEZ_LABELS_PATH)
    fareez_df = build_features(fareez_raw, glossary, phonetic_map)
    n_fareez = fareez_raw["encounter_id"].nunique()
    fareez_results = report(f"FAREEZ (all n={n_fareez} encounters)", fareez_df, model, ["confidence", "word_len"])

    return aci_results, fareez_results


if __name__ == "__main__":
    main()
