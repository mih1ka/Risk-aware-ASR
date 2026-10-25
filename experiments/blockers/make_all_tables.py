"""
R5 - regenerate every reported table from the repo, primary glossary = 176
terms (data/glossary/tiered_glossary_176.csv), with the 190-term glossary
alongside as a sensitivity analysis.

Tables
  (a) ACI-Bench synthetic speech (TTS + Whisper-base), 207 encounters:
      tier error rates with gold-word counts.
  (b) ACI-Bench production-ASR pairs, 105 encounters: the same.
  (c) ACI-Bench 5-fold CV (StratifiedGroupKFold, seed 42, tier-free XGBoost):
      recall at 5/10/20% for RiskAware and raw confidence, mean +- sd, over
      all critical errors and over scorable ones; ROC-AUC mean +- sd;
      per-fold gain at 10%.
  (d) Fareez (272), Whisper-base: tier rates with bootstrap 95% CI; recall
      at 5/10/20% for raw confidence, learned uncertainty and RiskAware,
      both denominators.
  Checks
  (e) ACI-Bench Tier 2 under 176/190 terms x old/new normalize().
  (f) The CV script on every labels/glossary version in git history,
      against the paper's 59.31% / 47.27% / 0.8535.

Sources and what is recomputed:
  - No ACI-Bench Whisper outputs are in the repo (data/whisper_out/ holds the
    272 Fareez files only), so (a), (c) and (e) start from the stored
    alignment, data/processed/word_level_labels.csv (and its pre-fix version,
    git e30d20c). Tiers are re-derived from each gold word with the chosen
    glossary. Alignment does not depend on the glossary, so this equals
    re-aligning; the script checks it reproduces the stored tier columns.
  - (b) re-aligns the production-ASR CSVs in memory with
    src/eval/validate_real_asr.py's functions.
  - (d) re-aligns all 272 Fareez whisper_out files in memory.
  - Labels CSVs are read keeping the spoken word "None" as a word.
  - Rankings break ties by original word order (stable sort), except in (f),
    which reproduces scratch/cv_5fold_no_tier.py exactly.

Writes r5_*.csv and r5_tables.md next to this file. Writes nothing else.
Run from anywhere: python experiments/blockers/make_all_tables.py
"""
import contextlib
import io
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common import (ROOT, OUT_DIR, GLOSSARY_176, GLOSSARY_190, NO_TIER_MODEL_PATH, BUDGETS,
                    RNG_SEED, load_glossary, read_labels, tier_rates_with_ci, recall_table)

sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "model"))
import align_and_label  # noqa: E402
import validate_real_asr  # noqa: E402
from risk_score import build_phonetic_map, lookup_tier, TIER_WEIGHTS  # noqa: E402

ACI_LABELS = ROOT / "data" / "processed" / "word_level_labels.csv"
ACI_PREFIX_COMMIT = "e30d20c"  # last labels file before the ok/okay normalize() fix
GLOSSARIES = {"176": GLOSSARY_176, "190": GLOSSARY_190}
PAPER = {"riskaware_10": 59.31, "raw_10": 47.27, "roc_auc": 0.8535}

md = []


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def to_markdown(df):
    cols = [" ".join(str(p) for p in c if str(p)) if isinstance(c, tuple) else str(c)
            for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def emit(title, df, note=None):
    print(f"\n### {title}\n")
    if note:
        print(note)
    print(df.to_string(index=False))
    md.append(f"### {title}\n")
    if note:
        md.append(note + "\n")
    md.append(to_markdown(df) + "\n")


def pct(x, d=2):
    return f"{x:.{d}f}%"


def retier(labels, glossary):
    out = labels.copy()
    gold = out["gold_word"].notna()
    out["tier"] = np.nan
    out.loc[gold, "tier"] = out.loc[gold, "gold_word"].map(
        lambda w: glossary.get(align_and_label.normalize(str(w)), 1))
    return out


def tier_counts(labels):
    gold = labels[labels["gold_word"].notna()]
    rows = []
    for t in (1, 2, 3):
        g = gold[gold["tier"] == t]
        rows.append({"tier": t, "gold_words": len(g), "errors": int(g["is_error"].sum()),
                     "error_rate_pct": 100 * g["is_error"].mean()})
    return pd.DataFrame(rows)


def tier_display(df):
    """Side-by-side 176 vs 190 view, formatted as strings so counts stay integers."""
    df = df.assign(error_rate=df["error_rate_pct"].map(lambda x: f"{x:.2f}%"),
                   gold_words=df["gold_words"].map(lambda x: f"{x:,}"),
                   errors=df["errors"].map(lambda x: f"{x:,}"))
    return df.pivot(index="tier", columns="glossary",
                    values=["error_rate", "gold_words", "errors"]).reset_index()


# --- (b) production ASR -----------------------------------------------------

def align_production(glossary_path):
    v = validate_real_asr
    v.GLOSSARY_PATH = glossary_path
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


# --- (c) 5-fold CV ----------------------------------------------------------

def cv_5fold(labels, glossary, sort_kind="stable"):
    phonetic_map = build_phonetic_map(glossary)
    df = labels[labels["whisper_word"].notna()].copy()
    df["tier_whisper"] = df["whisper_word"].apply(lambda w: lookup_tier(w, glossary, phonetic_map))
    df["confidence"] = df["confidence"].fillna(0.0)
    df["word_len"] = df["whisper_word"].astype(str).str.len()
    X = df[["confidence", "word_len"]]
    y = df["is_error"].to_numpy()
    groups = df["encounter_id"].to_numpy()
    tier_whisper, tier_gold = df["tier_whisper"].to_numpy(), df["tier"].to_numpy()
    crit_all_by_enc = labels[labels["tier"].isin([2, 3]) & (labels["is_error"] == 1)] \
        .groupby("encounter_id").size()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
        neg, pos = (y[tr] == 0).sum(), (y[tr] == 1).sum()
        model = xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42,
                                  scale_pos_weight=neg / pos)
        model.fit(X.iloc[tr], y[tr])
        learned = model.predict_proba(X.iloc[te])[:, 1]
        tw = np.vectorize(lambda t: TIER_WEIGHTS.get(t, 1.0))(tier_whisper[te])
        tg, ie = tier_gold[te], y[te]
        crit = np.isin(tg, [2, 3])
        n_scorable = int(ie[crit].sum())
        n_all = int(crit_all_by_enc.reindex(np.unique(groups[te])).fillna(0).sum())
        auc = roc_auc_score(ie, learned)
        for name, s in [("riskaware", learned * tw),
                        ("raw_confidence", 1 - X.iloc[te]["confidence"].to_numpy())]:
            for p in BUDGETS:
                flagged = np.argsort(-s, kind=sort_kind)[:int(len(s) * p)]
                c = int(ie[flagged][crit[flagged]].sum())
                rows.append({"fold": fold, "ranking": name, "budget_pct": int(p * 100),
                             "caught": c, "crit_errors_scorable": n_scorable,
                             "crit_errors_all": n_all, "roc_auc": auc,
                             "recall_scorable_pct": 100 * c / n_scorable,
                             "recall_all_pct": 100 * c / n_all})
    return pd.DataFrame(rows)


def cv_summary(per_fold, glossary_name):
    rows = []
    for (rk, b), g in per_fold.groupby(["ranking", "budget_pct"], sort=False):
        rows.append({"glossary": glossary_name, "ranking": rk, "budget_pct": b,
                     "recall_all": f"{g.recall_all_pct.mean():.2f} ± {g.recall_all_pct.std(ddof=1):.2f}%",
                     "recall_scorable": f"{g.recall_scorable_pct.mean():.2f} ± {g.recall_scorable_pct.std(ddof=1):.2f}%"})
    return pd.DataFrame(rows)


def cv_extras(per_fold, glossary_name):
    at10 = per_fold[per_fold.budget_pct == 10].pivot(index="fold", columns="ranking")
    gain_all = at10["recall_all_pct"]["riskaware"] - at10["recall_all_pct"]["raw_confidence"]
    gain_sc = at10["recall_scorable_pct"]["riskaware"] - at10["recall_scorable_pct"]["raw_confidence"]
    auc = per_fold.groupby("fold")["roc_auc"].first()
    folds = pd.DataFrame({"glossary": glossary_name, "fold": auc.index,
                          "crit_errors_all": at10["crit_errors_all"]["riskaware"].values,
                          "crit_errors_scorable": at10["crit_errors_scorable"]["riskaware"].values,
                          "roc_auc": auc.round(4).values,
                          "gain_10pct_all_pp": gain_all.round(2).values,
                          "gain_10pct_scorable_pp": gain_sc.round(2).values})
    summary = {"glossary": glossary_name,
               "roc_auc": f"{auc.mean():.4f} ± {auc.std(ddof=1):.4f}",
               "gain_10pct_all": f"{gain_all.mean():+.2f} ± {gain_all.std(ddof=1):.2f} pp",
               "gain_10pct_scorable": f"{gain_sc.mean():+.2f} ± {gain_sc.std(ddof=1):.2f} pp",
               "crit_errors_all": int(folds.crit_errors_all.sum()),
               "crit_errors_scorable": int(folds.crit_errors_scorable.sum())}
    return folds, summary


def main():
    head = git("log", "-1", "--format=%H %cd", "--date=iso").strip()
    dirty = [l for l in git("status", "--porcelain").splitlines() if l]
    print(f"git commit: {head}")
    print(f"working tree: {len(dirty)} uncommitted/untracked path(s)")
    md.append(f"Generated by `experiments/blockers/make_all_tables.py`.\n\n"
              f"git commit: `{head}` (working tree: {len(dirty)} uncommitted/untracked path(s))\n")

    glossaries = {k: load_glossary(p) for k, p in GLOSSARIES.items()}
    assert len(glossaries["176"]) == 176 and len(glossaries["190"]) == 190

    # ---------------- (a) ACI-Bench synthetic ----------------
    aci_stored = read_labels(ACI_LABELS)
    assert aci_stored["encounter_id"].nunique() == 207, aci_stored["encounter_id"].nunique()
    aci = {k: retier(aci_stored, g) for k, g in glossaries.items()}
    same = aci["190"]["tier"].fillna(-1).equals(aci_stored["tier"].fillna(-1))
    print(f"\nRe-tiering stored ACI-Bench labels with 190 terms reproduces stored tiers: {same}")
    assert same
    a = pd.concat([tier_counts(aci[k]).assign(glossary=k) for k in aci])
    a.to_csv(OUT_DIR / "r5_a_aci_synthetic_tiers.csv", index=False)
    emit("(a) ACI-Bench synthetic speech, Whisper-base, 207 encounters", tier_display(a))

    # ---------------- (b) production ASR ----------------
    with contextlib.redirect_stdout(io.StringIO()):
        prod = {k: align_production(p) for k, p in GLOSSARIES.items()}
    for k, d in prod.items():
        assert d["encounter_id"].nunique() == 105 and len(d) == 131144, (k, len(d))
    b = pd.concat([tier_counts(prod[k]).assign(glossary=k) for k in prod])
    b.to_csv(OUT_DIR / "r5_b_aci_production_tiers.csv", index=False)
    emit("(b) ACI-Bench production-ASR pairs, 105 encounters (131,144 rows)", tier_display(b))

    # ---------------- (c) ACI-Bench 5-fold CV ----------------
    per_fold = {k: cv_5fold(aci[k], glossaries[k]) for k in aci}
    pd.concat([d.assign(glossary=k) for k, d in per_fold.items()]) \
        .round(4).to_csv(OUT_DIR / "r5_c_cv_per_fold.csv", index=False)
    c_summary = pd.concat([cv_summary(per_fold[k], k) for k in per_fold])
    c_summary.to_csv(OUT_DIR / "r5_c_cv_recall.csv", index=False)
    extras = [cv_extras(per_fold[k], k) for k in per_fold]
    c_folds = pd.concat([e[0] for e in extras])
    c_extra = pd.DataFrame([e[1] for e in extras])
    c_folds.to_csv(OUT_DIR / "r5_c_cv_folds.csv", index=False)
    c_extra.to_csv(OUT_DIR / "r5_c_cv_auc_gain.csv", index=False)
    emit("(c) ACI-Bench 5-fold CV recall (mean ± sd over folds)",
         c_summary.pivot(index=["ranking", "budget_pct"], columns="glossary",
                         values=["recall_all", "recall_scorable"]).reset_index())
    emit("(c) ROC-AUC and RiskAware − raw confidence gain at 10%", c_extra)
    emit("(c) per fold", c_folds)

    # ---------------- (d) Fareez ----------------
    selected = set(json.loads((ROOT / "scratch" / "fareez_272_gold_text.json").read_text()))
    files = sorted(f for f in (ROOT / "data" / "whisper_out").glob("*.json") if f.stem in selected)
    assert len(files) == 272, len(files)
    model = joblib.load(NO_TIER_MODEL_PATH)
    d_tiers, d_recall = [], []
    for k, g in glossaries.items():
        labels, _ = align_and_label.align_files(files, g)
        t, _ = tier_rates_with_ci(labels)
        d_tiers.append(t.assign(glossary=k))
        d_recall.append(recall_table(labels, g, model).assign(glossary=k))
    d_tiers, d_recall = pd.concat(d_tiers), pd.concat(d_recall)
    d_tiers.round(4).to_csv(OUT_DIR / "r5_d_fareez_tiers.csv", index=False)
    d_recall.round(4).to_csv(OUT_DIR / "r5_d_fareez_recall.csv", index=False)
    d_tiers["rate [95% CI]"] = d_tiers.apply(
        lambda r: f"{r.error_rate_pct:.2f}% [{r.ci95_low_pct:.2f}, {r.ci95_high_pct:.2f}]", axis=1)
    emit("(d) Fareez, Whisper-base, 272 encounters: tier error rates",
         d_tiers.pivot(index="tier", columns="glossary", values=["rate [95% CI]", "gold_words", "errors"])
                .reset_index())
    d_recall["caught; all / scorable"] = d_recall.apply(
        lambda r: f"{r.caught}; {r.recall_all_pct:.2f} / {r.recall_scorable_pct:.2f}%", axis=1)
    denoms = d_recall.groupby("glossary")[["crit_errors_all", "crit_errors_scorable"]].first()
    emit("(d) Fareez critical-error recall",
         d_recall.pivot(index=["ranking", "budget_pct"], columns="glossary",
                        values="caught; all / scorable").reset_index(),
         note="Critical errors, all / scorable: " + "; ".join(
             f"{k} terms {r.crit_errors_all} / {r.crit_errors_scorable}" for k, r in denoms.iterrows()))

    # ---------------- (e) Tier 2: glossary vs normalize() ----------------
    prefix = pd.read_csv(io.StringIO(git("show", f"{ACI_PREFIX_COMMIT}:data/processed/word_level_labels.csv")),
                         keep_default_na=False, na_values=[""])
    for col in ("confidence", "is_error", "tier"):
        prefix[col] = pd.to_numeric(prefix[col])
    prefix_176 = retier(prefix, glossaries["176"])
    same = prefix_176["tier"].fillna(-1).equals(prefix["tier"].fillna(-1))
    print(f"\nRe-tiering pre-fix labels ({ACI_PREFIX_COMMIT}) with 176 terms reproduces stored tiers: {same}")
    assert same
    labels_by_norm = {"old (pre-fix, git e30d20c)": prefix, "new (current)": aci_stored}
    e_rows = []
    for nname, lab in labels_by_norm.items():
        for gname, g in glossaries.items():
            t2 = tier_counts(retier(lab, g)).set_index("tier").loc[2]
            e_rows.append({"normalize": nname, "glossary": gname,
                           "tier2_rate": pct(t2.error_rate_pct), "tier2_gold_words": int(t2.gold_words),
                           "tier2_errors": int(t2.errors)})
    e = pd.DataFrame(e_rows)
    e.to_csv(OUT_DIR / "r5_e_aci_tier2_normalize_vs_glossary.csv", index=False)
    emit("(e) ACI-Bench Tier 2: normalize() fix vs glossary", e)

    # ---------------- (f) search for 59.31 / 47.27 / 0.8535 ----------------
    f_rows = []
    for nname, lab in labels_by_norm.items():
        for gname, g in glossaries.items():
            lab_g = retier(lab, g)
            for read in ("pandas default", "keep None"):
                if read == "pandas default":  # as scratch/cv_5fold_no_tier.py reads it
                    src = pd.read_csv(io.StringIO(lab_g.to_csv(index=False)))
                else:
                    src = lab_g
                pf = cv_5fold(src, g, sort_kind="quicksort")
                at10 = pf[pf.budget_pct == 10].groupby("ranking")["recall_scorable_pct"].mean()
                f_rows.append({"labels": nname, "glossary": gname, "csv_read": read,
                               "riskaware_10_scorable": round(at10["riskaware"], 2),
                               "raw_10_scorable": round(at10["raw_confidence"], 2),
                               "roc_auc": round(pf.groupby("fold")["roc_auc"].first().mean(), 4)})
    f = pd.DataFrame(f_rows)
    f["matches_paper"] = ((f.riskaware_10_scorable == PAPER["riskaware_10"])
                          & (f.raw_10_scorable == PAPER["raw_10"]) & (f.roc_auc == PAPER["roc_auc"]))
    f.to_csv(OUT_DIR / "r5_f_paper_cv_search.csv", index=False)
    emit("(f) scratch/cv_5fold_no_tier.py settings on every labels/glossary version "
         "(paper: 59.31 / 47.27 / 0.8535)", f)

    (OUT_DIR / "r5_tables.md").write_text("\n".join(md))
    print(f"\nWrote r5_*.csv and r5_tables.md to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
