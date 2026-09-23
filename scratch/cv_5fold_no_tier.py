"""
Throwaway diagnostic — not part of the reproducible pipeline.

5-fold encounter-grouped cross-validation replacing the single 80/20
GroupShuffleSplit evaluation. Each fold retrains a temporary tier-free
classifier (same architecture as uncertainty_xgb_no_tier.joblib — no
tier feature), computes critical-error recall (risk_score vs raw
1-confidence) at 5/10/20% budgets and ROC-AUC, then discards the model.

Nothing is saved to models/. Does not touch fareez results, the
original single-split model, or the glossary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "model"))
from risk_score import load_glossary, build_phonetic_map, lookup_tier, TIER_WEIGHTS  # noqa: E402
from bootstrap_significance import critical_recall_at_pct  # noqa: E402

LABELS_PATH = ROOT / "data" / "processed" / "word_level_labels.csv"
N_FOLDS = 5
RNG_SEED = 42
BUDGETS = [0.05, 0.10, 0.20]


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
    y = df["is_error"].to_numpy()
    groups = df["encounter_id"].to_numpy()
    tier_whisper = df["tier_whisper"].to_numpy()
    tier_gold = df["tier"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)

    fold_results = {"recall_risk": {p: [] for p in BUDGETS},
                     "recall_raw": {p: [] for p in BUDGETS},
                     "roc_auc": [],
                     "gap_10pct": []}

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X_no_tier, y, groups), start=1):
        print(f"--- Fold {fold_i}/{N_FOLDS} ---")
        n_train_enc = len(np.unique(groups[train_idx]))
        n_test_enc = len(np.unique(groups[test_idx]))
        print(f"  Train encounters: {n_train_enc}  Test encounters: {n_test_enc}  "
              f"Train rows: {len(train_idx)}  Test rows: {len(test_idx)}")

        X_train, X_test = X_no_tier.iloc[train_idx], X_no_tier.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42,
                                   scale_pos_weight=neg / pos)
        model.fit(X_train, y_train)

        learned_uncertainty = model.predict_proba(X_test)[:, 1]
        raw_confidence = 1 - X_test["confidence"].to_numpy()

        tw_test = tier_whisper[test_idx]
        tier_weight = np.vectorize(lambda t: TIER_WEIGHTS.get(t, 1.0))(tw_test)
        risk_score = learned_uncertainty * tier_weight

        tg_test = tier_gold[test_idx]
        ie_test = y_test

        auc = roc_auc_score(y_test, learned_uncertainty)
        fold_results["roc_auc"].append(auc)
        print(f"  ROC-AUC: {auc:.4f}")

        for p in BUDGETS:
            r_risk = critical_recall_at_pct(risk_score, tg_test, ie_test, p)
            r_raw = critical_recall_at_pct(raw_confidence, tg_test, ie_test, p)
            fold_results["recall_risk"][p].append(r_risk)
            fold_results["recall_raw"][p].append(r_raw)
            print(f"  Budget {int(p*100)}%: risk_score recall={r_risk:.4f}  raw_confidence recall={r_raw:.4f}")

        gap_10 = fold_results["recall_risk"][0.10][-1] - fold_results["recall_raw"][0.10][-1]
        fold_results["gap_10pct"].append(gap_10)
        print(f"  Gap @10% (risk_score - raw_confidence): {gap_10*100:.2f}pp\n")

        del model  # discard, nothing saved to models/

    print("=" * 90)
    print(f"5-FOLD SUMMARY (StratifiedGroupKFold, encounter-grouped, seed={RNG_SEED})")
    print("=" * 90)

    def mean_std(arr):
        arr = np.array(arr)
        return arr.mean(), arr.std(ddof=1)

    print(f"\n{'Metric':<45}{'Mean':>10}{'Std':>10}")
    print("-" * 65)
    for p in BUDGETS:
        m, s = mean_std(fold_results["recall_risk"][p])
        print(f"{'Critical recall, risk_score @' + str(int(p*100)) + '%':<45}{m*100:>9.2f}%{s*100:>9.2f}%")
    for p in BUDGETS:
        m, s = mean_std(fold_results["recall_raw"][p])
        print(f"{'Critical recall, raw_confidence @' + str(int(p*100)) + '%':<45}{m*100:>9.2f}%{s*100:>9.2f}%")
    m, s = mean_std(fold_results["roc_auc"])
    print(f"{'ROC-AUC':<45}{m:>10.4f}{s:>10.4f}")
    m, s = mean_std(fold_results["gap_10pct"])
    print(f"{'Gap @10% (risk_score - raw_confidence)':<45}{m*100:>9.2f}pp{s*100:>8.2f}pp")

    print(f"\nPer-fold gap @10%: {[f'{g*100:.2f}pp' for g in fold_results['gap_10pct']]}")


if __name__ == "__main__":
    main()
