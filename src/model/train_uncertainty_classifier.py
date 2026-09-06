import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

GLOSSARY_PATH = Path("data/glossary/tiered_glossary.csv")
LABELS_PATH = Path("data/processed/word_level_labels.csv")


def normalize(word):
    return re.sub(r"[^a-z0-9']", "", str(word).lower())


def load_glossary():
    df = pd.read_csv(GLOSSARY_PATH)
    return dict(zip(df["term"].str.lower(), df["tier"]))


def main():
    glossary = load_glossary()
    df = pd.read_csv(LABELS_PATH)

    # Deletions (Whisper emitted nothing) can't be scored — there's no
    # output word for the classifier to attach a risk score to.
    df = df[df["whisper_word"].notna()].copy()

    # Reference-free domain-criticality feature: looked up from WHISPER's
    # own output word, not the gold word — at inference we never see gold.
    df["tier_whisper"] = df["whisper_word"].apply(lambda w: glossary.get(normalize(w), 1))
    df["confidence"] = df["confidence"].fillna(0.0)
    df["word_len"] = df["whisper_word"].astype(str).str.len()

    X = df[["confidence", "tier_whisper", "word_len"]]
    y = df["is_error"]
    groups = df["encounter_id"]

    # Split by encounter, not by row — otherwise words from the same
    # encounter leak between train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")
    print(f"Train encounters: {groups.iloc[train_idx].nunique()}  Test encounters: {groups.iloc[test_idx].nunique()}")
    print(f"Train error rate: {y_train.mean():.4f}  Test error rate: {y_test.mean():.4f}\n")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        print(f"=== {name} ===")
        print(classification_report(y_test, preds, digits=3))
        print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}\n")


if __name__ == "__main__":
    main()