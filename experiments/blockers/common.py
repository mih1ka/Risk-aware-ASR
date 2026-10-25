"""
Shared helpers for the blocker re-runs (R0-R3).

Same logic as scratch/precision_recall_at_budgets.py and
scratch/bootstrap_fareez_tier_rates.py, with two deliberate differences:
  - ranking uses a stable sort, so tied scores are broken by original
    word order (the scratch scripts use numpy's default, unstable quicksort);
  - recall is reported against two denominators: critical errors among
    scorable rows (rows with an ASR word) and all critical errors
    (including deleted gold words, which can never be flagged).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
PRIVATE_DIR = OUT_DIR / "private"

sys.path.insert(0, str(ROOT / "src" / "model"))
from risk_score import build_phonetic_map, lookup_tier, TIER_WEIGHTS  # noqa: E402

GLOSSARY_190 = ROOT / "data" / "glossary" / "tiered_glossary.csv"
GLOSSARY_176 = ROOT / "data" / "glossary" / "tiered_glossary_176.csv"
FAREEZ_LABELS_190 = ROOT / "data" / "processed" / "word_level_labels_fareez.csv"
FAREEZ_LABELS_176 = PRIVATE_DIR / "word_level_labels_fareez_176.csv"
NO_TIER_MODEL_PATH = ROOT / "models" / "uncertainty_xgb_no_tier.joblib"

BUDGETS = [0.05, 0.10, 0.20]
N_BOOTSTRAP = 2000
RNG_SEED = 42


def read_labels(path):
    """Read a word-level labels CSV keeping literal words like "None"/"NA".

    pandas' default NA parsing turns the spoken word "None" into NaN, which
    drops 7 gold words (and 6 ASR words) from the 272-encounter Fareez file.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for col in ("confidence", "is_error", "tier"):
        df[col] = pd.to_numeric(df[col])
    return df


def load_glossary(path):
    df = pd.read_csv(path)
    return dict(zip(df["term"].str.lower(), df["tier"]))


def tier_rates_with_ci(labels):
    """Per-tier error rate on gold-anchored rows, with encounter-level bootstrap CI."""
    df = labels[labels["gold_word"].notna()]
    enc = df["encounter_id"].to_numpy()
    tier = df["tier"].to_numpy()
    is_error = df["is_error"].to_numpy()

    unique_enc = np.unique(enc)
    idx_by_enc = {e: np.where(enc == e)[0] for e in unique_enc}

    rng = np.random.default_rng(RNG_SEED)
    boot = {1: [], 2: [], 3: []}
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(unique_enc, size=len(unique_enc), replace=True)
        idx = np.concatenate([idx_by_enc[e] for e in sampled])
        t_idx, e_idx = tier[idx], is_error[idx]
        for t in (1, 2, 3):
            mask = t_idx == t
            if mask.sum():
                boot[t].append(e_idx[mask].sum() / mask.sum())

    rows = []
    for t in (1, 2, 3):
        mask = tier == t
        n, n_err = int(mask.sum()), int(is_error[mask].sum())
        lo, hi = np.percentile(boot[t], [2.5, 97.5])
        rows.append({"tier": t, "gold_words": n, "errors": n_err,
                     "error_rate_pct": 100 * n_err / n,
                     "ci95_low_pct": 100 * lo, "ci95_high_pct": 100 * hi,
                     "resamples_used": len(boot[t])})
    return pd.DataFrame(rows), len(unique_enc)


def score_rows(labels, glossary, model):
    """Scorable rows (ASR word present) with the three rankings attached."""
    phonetic_map = build_phonetic_map(glossary)
    df = labels[labels["whisper_word"].notna()].copy()
    df["tier_whisper"] = df["whisper_word"].apply(lambda w: lookup_tier(w, glossary, phonetic_map))
    df["confidence"] = df["confidence"].fillna(0.0)
    df["word_len"] = df["whisper_word"].astype(str).str.len()
    df["learned_uncertainty"] = model.predict_proba(df[["confidence", "word_len"]])[:, 1]
    df["raw_uncertainty"] = 1 - df["confidence"]
    df["risk_score"] = df["learned_uncertainty"] * df["tier_whisper"].map(TIER_WEIGHTS)
    return df


def caught_at_budget(scores, tier_gold, is_error, pct):
    n_flag = int(len(scores) * pct)
    order = np.argsort(-scores, kind="stable")
    flagged = order[:n_flag]
    crit = np.isin(tier_gold, [2, 3])
    return int(is_error[flagged][crit[flagged]].sum()), n_flag


def n_critical_errors(labels):
    """(all critical errors, scorable critical errors); critical = gold tier 2/3."""
    crit_err = labels["tier"].isin([2, 3]) & (labels["is_error"] == 1)
    scorable = labels["whisper_word"].notna()
    return int(crit_err.sum()), int((crit_err & scorable).sum())


RANKINGS = [("raw_confidence", "raw_uncertainty"),
            ("learned_uncertainty", "learned_uncertainty"),
            ("riskaware", "risk_score")]


def recall_table(labels, glossary, model):
    df = score_rows(labels, glossary, model)
    n_all, n_scorable = n_critical_errors(labels)
    tiers, is_error = df["tier"].to_numpy(), df["is_error"].to_numpy()
    rows = []
    for name, col in RANKINGS:
        scores = df[col].to_numpy()
        for p in BUDGETS:
            caught, n_flag = caught_at_budget(scores, tiers, is_error, p)
            rows.append({"ranking": name, "budget_pct": int(p * 100),
                         "n_scorable_rows": len(df), "n_flagged": n_flag, "caught": caught,
                         "crit_errors_scorable": n_scorable, "crit_errors_all": n_all,
                         "recall_scorable_pct": 100 * caught / n_scorable,
                         "recall_all_pct": 100 * caught / n_all})
    return pd.DataFrame(rows)
