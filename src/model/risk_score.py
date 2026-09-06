import re
from pathlib import Path

import pandas as pd
import jellyfish
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import joblib

GLOSSARY_PATH = Path("data/glossary/tiered_glossary.csv")
LABELS_PATH = Path("data/processed/word_level_labels.csv")
MODEL_PATH = Path("models/uncertainty_xgb.joblib")

TIER_WEIGHTS = {1: 1.0, 2: 3.25, 3: 4.25}


def normalize(word):
    return re.sub(r"[^a-z0-9']", "", str(word).lower())


def load_glossary():
    df = pd.read_csv(GLOSSARY_PATH)
    return dict(zip(df["term"].str.lower(), df["tier"]))


def build_phonetic_map(glossary):
    phonetic_map = {}
    for term, tier in glossary.items():
        code = jellyfish.metaphone(term)
        if code:
            phonetic_map[code] = max(phonetic_map.get(code, 0), tier)
    return phonetic_map


def lookup_tier(word, glossary, phonetic_map):
    norm = normalize(word)
    if norm in glossary:
        return glossary[norm]
    code = jellyfish.metaphone(norm)
    return phonetic_map.get(code, 1)


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

    X = df[["confidence", "tier_whisper", "word_len"]]
    y = df["is_error"]
    groups = df["encounter_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42,
                               scale_pos_weight=neg / pos)
    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    test_df = df.iloc[test_idx].copy()
    test_df["learned_uncertainty"] = model.predict_proba(X.iloc[test_idx])[:, 1]
    test_df["tier_weight"] = test_df["tier_whisper"].map(TIER_WEIGHTS)
    test_df["risk_score"] = test_df["learned_uncertainty"] * test_df["tier_weight"]

    out_path = Path("data/processed/risk_scores_test.csv")
    test_df.to_csv(out_path, index=False)
    print(f"Saved {len(test_df)} scored words to {out_path}\n")

    critical_mask = test_df["tier"].isin([2, 3])
    n_critical_errors = test_df.loc[critical_mask, "is_error"].sum()
    print(f"Total truly-critical (gold Tier 2/3) errors in test set: {n_critical_errors}\n")

    for pct in [0.05, 0.10, 0.20]:
        n_flag = int(len(test_df) * pct)
        by_risk = test_df.sort_values("risk_score", ascending=False).iloc[:n_flag]
        by_unc = test_df.sort_values("learned_uncertainty", ascending=False).iloc[:n_flag]

        precision_risk = by_risk["is_error"].mean()
        recall_risk = by_risk["is_error"].sum() / test_df["is_error"].sum()
        recall_risk_crit = by_risk.loc[by_risk["tier"].isin([2, 3]), "is_error"].sum() / n_critical_errors
        recall_unc_crit = by_unc.loc[by_unc["tier"].isin([2, 3]), "is_error"].sum() / n_critical_errors

        print(f"Top {pct*100:.0f}% flagged ({n_flag} words): "
              f"overall precision={precision_risk:.3f}  overall recall={recall_risk:.3f}")
        print(f"    critical-error recall — risk_score={recall_risk_crit:.3f}  "
              f"uncertainty_only={recall_unc_crit:.3f}\n")


if __name__ == "__main__":
    main()