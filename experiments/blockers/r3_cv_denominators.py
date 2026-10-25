"""
R3 - ACI-Bench 5-fold CV recall with two denominators.

Same folds, model and scoring as scratch/cv_5fold_no_tier.py
(StratifiedGroupKFold, 5 splits, shuffle, seed 42, tier-free XGBoost,
190-term glossary for the inference-time tier lookup). Per fold, critical
errors (gold tier 2/3) are counted
  - scorable: rows with an ASR word (what the CV script uses), and
  - all: also deleted gold words (no ASR word, can never be flagged),
over the fold's test encounters. Hits are identical under both; only the
denominator differs.

Variants (to locate where the reported numbers come from):
  sort:  "default" = np.argsort quicksort (as in the CV script),
         "stable"  = ties broken by original word order;
  read:  "default" = pd.read_csv defaults (as in the CV script),
         "keep_none" = the word "None" kept as a word, not NaN.

Writes r3_per_fold.csv and r3_summary.csv (aggregates only).
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common import ROOT, OUT_DIR, GLOSSARY_190, BUDGETS, load_glossary, read_labels

sys.path.insert(0, str(ROOT / "src" / "model"))
from risk_score import build_phonetic_map, lookup_tier, TIER_WEIGHTS  # noqa: E402

LABELS_PATH = ROOT / "data" / "processed" / "word_level_labels.csv"
N_FOLDS = 5
RNG_SEED = 42


def build_features(df, glossary, phonetic_map):
    df = df[df["whisper_word"].notna()].copy()
    df["tier_whisper"] = df["whisper_word"].apply(lambda w: lookup_tier(w, glossary, phonetic_map))
    df["confidence"] = df["confidence"].fillna(0.0)
    df["word_len"] = df["whisper_word"].astype(str).str.len()
    return df


def caught(scores, tier_gold, is_error, pct, kind):
    n_flag = int(len(scores) * pct)
    flagged = np.argsort(-scores, kind=kind)[:n_flag]
    crit = np.isin(tier_gold, [2, 3])
    return int(is_error[flagged][crit[flagged]].sum())


def run(read, sort_kind, glossary, phonetic_map):
    raw = read_labels(LABELS_PATH) if read == "keep_none" else pd.read_csv(LABELS_PATH)
    df = build_features(raw, glossary, phonetic_map)
    X = df[["confidence", "word_len"]]
    y = df["is_error"].to_numpy()
    groups = df["encounter_id"].to_numpy()
    tier_whisper = df["tier_whisper"].to_numpy()
    tier_gold = df["tier"].to_numpy()

    crit_err_all = raw[raw["tier"].isin([2, 3]) & (raw["is_error"] == 1)]
    all_by_enc = crit_err_all.groupby("encounter_id").size()

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
        neg, pos = (y[tr] == 0).sum(), (y[tr] == 1).sum()
        model = xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42,
                                  scale_pos_weight=neg / pos)
        model.fit(X.iloc[tr], y[tr])
        learned = model.predict_proba(X.iloc[te])[:, 1]
        tw = np.vectorize(lambda t: TIER_WEIGHTS.get(t, 1.0))(tier_whisper[te])
        scores = {"riskaware": learned * tw,
                  "raw_confidence": 1 - X.iloc[te]["confidence"].to_numpy()}
        tg, ie = tier_gold[te], y[te]
        n_scorable = int(ie[np.isin(tg, [2, 3])].sum())
        n_all = int(all_by_enc.reindex(np.unique(groups[te])).fillna(0).sum())
        for name, s in scores.items():
            for p in BUDGETS:
                c = caught(s, tg, ie, p, sort_kind)
                rows.append({"read": read, "sort": sort_kind, "fold": fold,
                             "test_encounters": len(np.unique(groups[te])),
                             "roc_auc": roc_auc_score(ie, learned),
                             "ranking": name, "budget_pct": int(p * 100), "caught": c,
                             "crit_errors_scorable": n_scorable, "crit_errors_all": n_all,
                             "deleted_crit_errors": n_all - n_scorable,
                             "recall_scorable_pct": 100 * c / n_scorable,
                             "recall_all_pct": 100 * c / n_all})
    return pd.DataFrame(rows)


def main():
    glossary = load_glossary(GLOSSARY_190)
    phonetic_map = build_phonetic_map(glossary)
    per_fold = pd.concat([run(r, s, glossary, phonetic_map)
                          for r in ("default", "keep_none") for s in ("quicksort", "stable")],
                         ignore_index=True)
    per_fold["sort"] = per_fold["sort"].replace({"quicksort": "default"})
    per_fold.round(4).to_csv(OUT_DIR / "r3_per_fold.csv", index=False)

    keys = ["read", "sort", "ranking", "budget_pct"]
    summary = per_fold.groupby(keys).agg(
        recall_scorable_mean=("recall_scorable_pct", "mean"),
        recall_scorable_sd=("recall_scorable_pct", lambda x: x.std(ddof=1)),
        recall_all_mean=("recall_all_pct", "mean"),
        recall_all_sd=("recall_all_pct", lambda x: x.std(ddof=1)),
        caught_total=("caught", "sum"),
        crit_errors_scorable_total=("crit_errors_scorable", "sum"),
        crit_errors_all_total=("crit_errors_all", "sum"),
        roc_auc_mean=("roc_auc", "mean"),
    ).reset_index()
    summary.round(4).to_csv(OUT_DIR / "r3_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print(summary.round(2).to_string(index=False))
    folds = per_fold[(per_fold.read == "default") & (per_fold.sort == "default")
                     & (per_fold.ranking == "riskaware") & (per_fold.budget_pct == 10)]
    print(folds[["fold", "test_encounters", "crit_errors_scorable", "crit_errors_all",
                 "deleted_crit_errors"]].to_string(index=False))

    # Same CV on the pre-normalize()-fix labels (git 9d19693), to trace which
    # labels the reported figures came from. Pass the extracted file as argv[1]:
    #   git show 9d19693:data/processed/word_level_labels.csv > /tmp/prefix.csv
    if len(sys.argv) > 1:
        global LABELS_PATH
        LABELS_PATH = sys.argv[1]
        prefix = run("default", "quicksort", glossary, phonetic_map)
        prefix["sort"] = "default"
        prefix_summary = prefix.groupby(keys).agg(
            recall_scorable_mean=("recall_scorable_pct", "mean"),
            recall_scorable_sd=("recall_scorable_pct", lambda x: x.std(ddof=1)),
            recall_all_mean=("recall_all_pct", "mean"),
            recall_all_sd=("recall_all_pct", lambda x: x.std(ddof=1)),
            roc_auc_mean=("roc_auc", "mean"),
        ).reset_index()
        prefix_summary.insert(0, "labels", "pre-normalize-fix (git 9d19693)")
        prefix_summary.round(4).to_csv(OUT_DIR / "r3_prefix_labels_summary.csv", index=False)
        print(prefix_summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
