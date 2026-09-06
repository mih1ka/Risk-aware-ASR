import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb

OUT_DIR = Path("report/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY_INK, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
TIER_RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # sequential blue, light->dark, for ordinal tier severity

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": GRID, "axes.labelcolor": SECONDARY_INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "grid.color": GRID, "axes.grid": True, "axes.axisbelow": True, "grid.linewidth": 0.6,
})


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", linewidth=0.6, color=GRID)
    ax.set_axisbelow(True)


def fig_wer_by_tier():
    tiers = ["Tier 1\n(everyday)", "Tier 2\n(clinical term)", "Tier 3\n(drug/dosage)"]
    rates = [12.6, 67.4, 90.2]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(tiers, rates, color=TIER_RAMP, width=0.6)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2, f"{rate:.1f}%",
                 ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel("Word error rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Transcription error rate by domain-criticality tier", fontsize=12, color=INK, pad=12)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_wer_by_tier.png", dpi=200)
    plt.close(fig)


def fig_roc_curves():
    df = pd.read_csv("data/processed/word_level_labels.csv")
    df = df[df["whisper_word"].notna()].copy()

    def normalize(word):
        return re.sub(r"[^a-z0-9']", "", str(word).lower())

    glossary_df = pd.read_csv("data/glossary/tiered_glossary.csv")
    glossary = dict(zip(glossary_df["term"].str.lower(), glossary_df["tier"]))

    df["tier_whisper"] = df["whisper_word"].apply(lambda w: glossary.get(normalize(w), 1))
    df["confidence"] = df["confidence"].fillna(0.0)
    df["word_len"] = df["whisper_word"].astype(str).str.len()

    X = df[["confidence", "tier_whisper", "word_len"]]
    y = df["is_error"]
    groups = df["encounter_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced"), BLUE),
        "Random Forest": (RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42), ORANGE),
        "XGBoost": (xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, scale_pos_weight=neg/pos), AQUA),
    }

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, (model, color) in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — reference-free error classifiers", fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_roc_curves.png", dpi=200)
    plt.close(fig)


def fig_flagging_curve():
    df = pd.read_csv("data/processed/risk_scores_test.csv")
    df_sorted = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    pcts = np.arange(1, 51) / 100
    precisions, recalls = [], []
    total_errors = df_sorted["is_error"].sum()
    for pct in pcts:
        n_flag = int(n * pct)
        flagged = df_sorted.iloc[:n_flag]
        precisions.append(flagged["is_error"].mean())
        recalls.append(flagged["is_error"].sum() / total_errors)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(pcts * 100, np.array(precisions) * 100, color=BLUE, linewidth=2, label="Precision")
    ax.plot(pcts * 100, np.array(recalls) * 100, color=ORANGE, linewidth=2, label="Recall")
    ax.set_xlabel("% of words flagged for review")
    ax.set_ylabel("%")
    ax.set_title("Review-flagging tradeoff: precision & recall vs. flagging budget", fontsize=11, color=INK, pad=12)
    ax.legend(frameon=False, loc="center right", fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_flagging_curve.png", dpi=200)
    plt.close(fig)


def fig_ablation():
    df = pd.read_csv("data/processed/risk_scores_test.csv")
    n = len(df)
    n_critical_errors = df.loc[df["tier"].isin([2, 3]), "is_error"].sum()

    cutoffs = [0.05, 0.10, 0.20]
    risk_recalls, unc_recalls = [], []
    for pct in cutoffs:
        n_flag = int(n * pct)
        by_risk = df.sort_values("risk_score", ascending=False).iloc[:n_flag]
        by_unc = df.sort_values("learned_uncertainty", ascending=False).iloc[:n_flag]
        risk_recalls.append(by_risk.loc[by_risk["tier"].isin([2, 3]), "is_error"].sum() / n_critical_errors)
        unc_recalls.append(by_unc.loc[by_unc["tier"].isin([2, 3]), "is_error"].sum() / n_critical_errors)

    x = np.arange(len(cutoffs))
    width = 0.32
    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars1 = ax.bar(x - width/2, np.array(risk_recalls) * 100, width, color=BLUE, label="Risk Score (uncertainty × criticality)")
    bars2 = ax.bar(x + width/2, np.array(unc_recalls) * 100, width, color=ORANGE, label="Uncertainty only")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top {int(c*100)}% flagged" for c in cutoffs])
    ax.set_ylabel("Critical-error recall (%)")
    ax.set_title("Does domain-criticality weighting help?", fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_ablation.png", dpi=200)
    plt.close(fig)


def fig_synthetic_vs_real():
    tiers = ["Tier 1", "Tier 2", "Tier 3"]
    synthetic = [12.6, 67.4, 90.2]
    real = [2.2, 1.8, 1.3]
    x = np.arange(len(tiers))
    width = 0.32
    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars1 = ax.bar(x - width/2, synthetic, width, color=BLUE, label="Synthetic (Whisper-base + TTS)")
    bars2 = ax.bar(x + width/2, real, width, color=ORANGE, label="Real ASR (production system)")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Word error rate (%)")
    ax.set_title("General-purpose vs. production ASR: error rate by tier", fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_synthetic_vs_real.png", dpi=200)
    plt.close(fig)


def main():
    fig_wer_by_tier()
    fig_roc_curves()
    fig_flagging_curve()
    fig_ablation()
    fig_synthetic_vs_real()
    print(f"Saved 5 figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()