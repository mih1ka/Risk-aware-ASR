"""
RiskAware-ASR — live demo

Two modes:
  1. Upload audio -> Whisper transcribes it live -> the real risk-scoring
     pipeline flags words. True reference-free operation (no gold text used).
  2. Load a saved sample -> an already-processed ACI-Bench encounter,
     scored by the same model, no audio or internet needed. Guaranteed to
     work -- use this as the rehearsal/fallback path.

Run from the repo root:
    streamlit run demo/app.py
"""
import html
import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.risk_pipeline import (  # noqa: E402
    RiskAwareASRPipeline, suggest_correction, lookup_tier, explain_reason,
    FLAG_RISK_THRESHOLD, TIER_WEIGHTS,
)

st.set_page_config(page_title="RiskAware-ASR Demo", layout="wide")

# Light-mode ink roles (keep in sync with .streamlit/config.toml's `base`).
# Flip LIGHT_MODE back to False only if config.toml's theme.base is set back
# to "dark" -- the two must always match.
LIGHT_MODE = True
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
CARD_BG = "rgba(11,11,11,0.035)"
SUBTLE_BORDER = "rgba(11,11,11,0.12)"
SUBTLE_FILL = "rgba(11,11,11,0.035)"
BRAND_ACCENT = "#1baf7a"

# Custom styling only -- every selector below is either a stable, documented
# Streamlit data-testid hook or a plain HTML tag. A selector that doesn't
# match anything is just a silent no-op in CSS; it can't break the app.
st.markdown(
    f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1100px; }}

h2, h3 {{ border-left: 3px solid {BRAND_ACCENT}; padding-left: 12px; }}

/* Mode selector -> segmented-pill look */
div[data-testid="stRadio"] > div[role="radiogroup"] {{ gap: 8px; flex-wrap: wrap; }}
div[data-testid="stRadio"] label {{
    background: {SUBTLE_FILL};
    border: 1px solid {SUBTLE_BORDER};
    border-radius: 999px;
    padding: 6px 16px;
    margin-right: 0 !important;
    transition: border-color 0.15s ease, background 0.15s ease;
}}
div[data-testid="stRadio"] label:hover {{ border-color: {BRAND_ACCENT}; }}
div[data-testid="stRadio"] label:has(input:checked) {{
    background: rgba(27,175,122,0.14);
    border-color: {BRAND_ACCENT};
}}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {{
    border-radius: 12px;
    border: 1.5px dashed rgba(27,175,122,0.5);
}}

/* Expander, dataframe, audio player */
[data-testid="stExpander"] {{
    border-radius: 10px;
    border: 1px solid {SUBTLE_BORDER};
    overflow: hidden;
}}
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
[data-testid="stAudio"] {{ margin: 10px 0 20px 0; }}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {{
    border-radius: 8px;
}}

hr {{ margin: 1.6rem 0; opacity: 0.18; }}
</style>
""",
    unsafe_allow_html=True,
)

SAMPLE_CSV = REPO_ROOT / "data" / "processed" / "risk_scores_explained.csv"

# -- SOAP note generation (optional feature) --------------------------------
# Uses Groq's free API (console.groq.com) rather than a paid key.
# llama-3.3-70b-versatile moved to Groq's Enterprise tier; gpt-oss-120b is
# the current free-tier model (confirmed against console.groq.com/docs/
# rate-limits -- free plan: 30 req/min, 1,000 req/day, 8,000 tokens/min).
SOAP_MODEL = "openai/gpt-oss-120b"
SOAP_SYSTEM_PROMPT = (
    "You are a clinical scribe. You are given a word-by-word transcript of "
    "a doctor-patient conversation (no punctuation or speaker labels). "
    "Write a concise, structured SOAP note. Infer speaker turns from "
    "context. If a section has no supporting content in the transcript, "
    "write 'Not documented' for that section rather than inventing detail. "
    "Preserve every medication name and dosage exactly as given in the "
    "transcript, including any '(heard as: ...)' annotations -- keep those "
    "annotations in the note wherever that term is used, so a reviewer can "
    "see what was corrected.\n\n"
    "Format with exactly these four markdown headings, in this order, each "
    "as '### Subjective', '### Objective', '### Assessment', '### Plan' -- "
    "no other heading, title, or preamble before or after them."
)
FALLBACK_SOAP_NOTE_PATH = REPO_ROOT / "demo" / "fallback_soap_note.md"

# The paper's primary evaluation: Fareez n=272, 176-term glossary, ten
# methods x three budgets, with paired whole-encounter bootstrap CIs against
# RiskAware-ASR (PR #7, experiments/blockers_yashvi/RESULTS.md).
VALIDATION_CSV = (REPO_ROOT / "experiments" / "blockers_yashvi"
                   / "PRIMARY_176_term_comparison.csv")
RAW_VS_RISKAWARE_CSV = (REPO_ROOT / "experiments" / "blockers_yashvi"
                         / "y176_riskaware_vs_raw_5_10_20.csv")
N_CRITICAL_ALL = 180      # 171 scorable + 9 deleted gold words
N_TOTAL_WORDS = 352473
N_ENCOUNTERS = 272

# Fixed order = fixed categorical color assignment (never re-cycled).
# Light-mode steps from the validated default palette; passes the adjacent-pair
# CVD check for line charts. RiskAware-ASR gets a bolder stroke, not a
# different color, to stay on-model as "our method" without breaking the
# color-by-entity rule. "Disagreement + confidence" is dashed because it's a
# genuinely different class of method (two ASR passes compared against each
# other), not a single-pass competitor.
METHOD_ORDER = ["Raw confidence", "Learned uncertainty", "RiskAware-ASR", "Disagreement + confidence"]
# Short labels for the full ten-method table.
ALL_METHOD_LABELS = {
    "A1 Glossary-only strict": "A1: glossary only",
    "A2 Glossary-restricted confidence strict": "A2: glossary, by confidence",
    "A3 Glossary first then confidence filled": "A3: glossary first, confidence fills",
    "B (1-confidence) x tier": "B: (1 - confidence) x tier",
    "D Tier gate then learned uncertainty": "D: tier gate + learned uncertainty",
    "C1 Disagreement then RiskAware score": "C1: disagreement + risk score (2 models)",
}
METHOD_COLORS = {
    "Raw confidence": "#2a78d6",
    "Learned uncertainty": "#eb6834",
    "RiskAware-ASR": BRAND_ACCENT,
    "Disagreement + confidence": "#eda100",
}
METHOD_LABELS = {
    "Raw confidence": "Raw confidence",
    "Learned uncertainty": "Learned uncertainty",
    "RiskAware-ASR": "RiskAware-ASR (ours)",
    "Disagreement + confidence": "Disagreement + confidence",
}
METHOD_DASH = {
    "Raw confidence": [1, 0],
    "Learned uncertainty": [1, 0],
    "RiskAware-ASR": [1, 0],
    "Disagreement + confidence": [6, 3],
}
METHOD_WIDTH = {
    "Raw confidence": 2,
    "Learned uncertainty": 2,
    "RiskAware-ASR": 3.5,
    "Disagreement + confidence": 2,
}

# Severity palette (status roles, not categorical -- these mean "state", so
# they're reserved and never reused as a series color elsewhere in the app).
RISK_TIERS = {
    "critical": {"color": "#d03b3b", "label": "High risk"},
    "serious":  {"color": "#ec835a", "label": "Medium risk"},
    "warning":  {"color": "#fab219", "label": "Flagged, lower risk"},
}
BADGE_GOOD = {"bg": "#0ca30c", "fg": "#ffffff"}     # a suggestion exists
BADGE_CRITICAL = {"bg": "#d03b3b", "fg": "#ffffff"}  # no suggestion -- escalate


def risk_tier(risk_score: float) -> str:
    if risk_score >= 2.5:
        return "critical"
    if risk_score >= 1.0:
        return "serious"
    return "warning"


# Glossary-tier identity colors for the confidence-vs-risk scatter -- a
# separate categorical set from METHOD_COLORS (different chart, different
# meaning) and deliberately not reusing the RISK_TIERS status reds, since
# tier (word identity) and severity (outcome) are different variables that
# happen to correlate, not the same thing.
GLOSSARY_TIER_LABELS = {1: "Everyday", 2: "Clinical term", 3: "Drug/dosage"}
GLOSSARY_TIER_ORDER = ["Everyday", "Clinical term", "Drug/dosage"]
GLOSSARY_TIER_COLORS = {"Everyday": "#2a78d6", "Clinical term": "#6250d6", "Drug/dosage": "#e34948"}
# Rare tiers (2/3) get full opacity + bigger points so they aren't washed out
# by the dense cloud of ordinary words -- the point is to make the signal
# visible, not to plot every point identically.
GLOSSARY_TIER_OPACITY = {"Everyday": 0.35, "Clinical term": 0.95, "Drug/dosage": 0.95}
GLOSSARY_TIER_SIZE = {"Everyday": 30, "Clinical term": 110, "Drug/dosage": 110}


def render_confidence_scatter(df: pd.DataFrame):
    """df needs columns: word, confidence, tier_whisper, risk_score.
    Shows why the model isn't just thresholding on confidence: tier-3
    (drug/dosage) words get flagged even at high confidence."""
    try:
        import altair as alt
    except ImportError:
        st.caption("(Install `altair` to see this chart: `pip install altair`.)")
        return

    plot_df = df.copy()
    plot_df["tier_label"] = plot_df["tier_whisper"].map(GLOSSARY_TIER_LABELS)
    plot_df["_opacity"] = plot_df["tier_label"].map(GLOSSARY_TIER_OPACITY)
    plot_df["_size"] = plot_df["tier_label"].map(GLOSSARY_TIER_SIZE)

    color_scale = alt.Scale(
        domain=GLOSSARY_TIER_ORDER,
        range=[GLOSSARY_TIER_COLORS[t] for t in GLOSSARY_TIER_ORDER],
    )

    points = alt.Chart(plot_df).mark_circle().encode(
        x=alt.X("confidence:Q", title="ASR confidence", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("risk_score:Q", title="Risk score",
                scale=alt.Scale(domain=[0, max(float(plot_df["risk_score"].max()) * 1.05, 1.0)])),
        color=alt.Color("tier_label:N", scale=color_scale, title="Glossary tier", sort=GLOSSARY_TIER_ORDER),
        opacity=alt.Opacity("_opacity:Q", legend=None, scale=None),
        size=alt.Size("_size:Q", legend=None, scale=None),
        order=alt.Order("tier_whisper:Q"),
        tooltip=[
            alt.Tooltip("word:N", title="Word"),
            alt.Tooltip("confidence:Q", title="Confidence", format=".2f"),
            alt.Tooltip("tier_label:N", title="Tier"),
            alt.Tooltip("risk_score:Q", title="Risk score", format=".2f"),
        ],
    )
    threshold_df = pd.DataFrame({"y": [FLAG_RISK_THRESHOLD]})
    rule = alt.Chart(threshold_df).mark_rule(strokeDash=[4, 4], color=INK_MUTED).encode(y="y:Q")

    chart = (
        (points + rule).properties(height=320, background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(gridColor=GRID_COLOR, domainColor=AXIS_COLOR,
                         labelColor=INK_SECONDARY, titleColor=INK_PRIMARY)
        .configure_legend(labelColor=INK_PRIMARY, titleColor=INK_PRIMARY)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Dashed line = flagging threshold. Drug/dosage words (red) get flagged "
        "even at high ASR confidence -- that's the clinical-criticality weight "
        "doing its job, not just a confidence cutoff."
    )


@st.cache_resource
def load_whisper_model():
    import whisper
    return whisper.load_model("base")


@st.cache_resource
def load_risk_pipeline():
    return RiskAwareASRPipeline()


@st.cache_data
def load_sample_data():
    df = pd.read_csv(SAMPLE_CSV)
    df["flagged"] = df["risk_score"] >= FLAG_RISK_THRESHOLD
    tier3_flags = df[df["flagged"] & (df["tier_whisper"] == 3)]
    counts = tier3_flags.groupby("encounter_id").size().sort_values(ascending=False)
    return df, counts


@st.cache_data
def load_validation_data():
    full = pd.read_csv(VALIDATION_CSV)
    full = full[full["glossary_terms"] == 176].copy()
    full["method_label"] = full["method"].map({**METHOD_LABELS, **ALL_METHOD_LABELS})
    df = full[full["method"].isin(METHOD_ORDER)].copy()
    boot = pd.read_csv(RAW_VS_RISKAWARE_CSV) if RAW_VS_RISKAWARE_CSV.exists() else None
    return df, full, boot


def render_confusion_matrix(df: pd.DataFrame):
    """2x2 confusion matrix for RiskAware-ASR at a chosen budget, derived
    from the same committed CSV the line chart uses. TP/FP/FN/TN sum to
    N_TOTAL_WORDS exactly and reproduce the paper's reported recall/precision
    -- verified by hand before this was wired in."""
    st.subheader("Confusion matrix at your chosen review budget")
    budget = st.select_slider(
        "Review budget", options=[5, 10, 20], value=10, format_func=lambda b: f"{b}%"
    )
    row = df[(df["method"] == "RiskAware-ASR") & (df["budget_pct"] == budget)]
    if row.empty:
        st.info("No data for this budget.")
        return
    row = row.iloc[0]
    tp = int(row["critical_caught"])
    flagged = int(row["flagged_words"])
    fp = flagged - tp
    fn = N_CRITICAL_ALL - tp
    tn = N_TOTAL_WORDS - flagged - fn

    def cell(label, value, color):
        st.markdown(
            f"""
<div style="background:{color}1a; border:1px solid {color}; border-radius:10px;
            padding:16px; text-align:center;">
  <div style="font-size:1.6rem; font-weight:700; color:{INK_PRIMARY};">{value:,}</div>
  <div style="color:{color}; font-size:0.78rem; font-weight:600; text-transform:uppercase;
              letter-spacing:0.03em; margin-top:4px;">{label}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    h1, h2 = st.columns(2)
    h1.caption("**Flagged for review**")
    h2.caption("**Not flagged**")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        cell("Caught (TP)", tp, "#0ca30c")
    with r1c2:
        cell("Missed (FN)", fn, "#d03b3b")
    st.caption("↑ Actually a critical error")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        cell("Flagged, not critical (FP)", fp, "#fab219")
    with r2c2:
        cell("Correctly left alone (TN)", tn, "#0ca30c")
    st.caption("↑ Not a critical error")

    st.caption(
        f"At a {budget}% review budget: catches {tp} of {tp + fn} real critical errors "
        f"({tp / (tp + fn) * 100:.1f}% recall), at the cost of reviewing {flagged:,} words "
        f"total -- {fp:,} of which turn out not to be critical errors."
    )


def render_validation_section():
    st.subheader("Validation — real recordings (Fareez, n=272)")
    st.caption(
        "Critical-error recall at each review budget on the Fareez real-speech "
        "set (272 recorded encounters, 352,473 scored words, 180 critical errors "
        "under the 176-term glossary: 171 scorable, 9 deleted). The classifier "
        "never saw this data. This is the evaluation reported in the paper -- "
        "not a live run."
    )

    if not VALIDATION_CSV.exists():
        st.error(f"Validation results not found at {VALIDATION_CSV}")
        return

    df, full, boot = load_validation_data()

    try:
        import altair as alt

        color_scale = alt.Scale(
            domain=[METHOD_LABELS[m] for m in METHOD_ORDER],
            range=[METHOD_COLORS[m] for m in METHOD_ORDER],
        )
        dash_scale = alt.Scale(
            domain=METHOD_ORDER, range=[METHOD_DASH[m] for m in METHOD_ORDER]
        )
        width_scale = alt.Scale(
            domain=METHOD_ORDER, range=[METHOD_WIDTH[m] for m in METHOD_ORDER]
        )

        base = alt.Chart(df).encode(
            x=alt.X("budget_pct:O", title="Review budget (% of words flagged)"),
            y=alt.Y("recall_all_pct:Q", title="Critical-error recall (%)",
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("method_label:N", scale=color_scale, title="Method",
                             sort=[METHOD_LABELS[m] for m in METHOD_ORDER]),
        )

        lines = base.mark_line(point=True).encode(
            strokeDash=alt.StrokeDash("method:N", scale=dash_scale, legend=None),
            strokeWidth=alt.StrokeWidth("method:N", scale=width_scale, legend=None),
            tooltip=[
                alt.Tooltip("method_label:N", title="Method"),
                alt.Tooltip("budget_pct:O", title="Budget (%)"),
                alt.Tooltip("recall_all_pct:Q", title="Recall (%)", format=".1f"),
                alt.Tooltip("precision_pct:Q", title="Precision (%)", format=".1f"),
            ],
        )

        chart = (
            lines.properties(height=380, background="transparent")
            .configure_view(strokeWidth=0, fill="transparent")
            .configure_axis(gridColor=GRID_COLOR, domainColor=AXIS_COLOR,
                             labelColor=INK_SECONDARY, titleColor=INK_PRIMARY)
            .configure_legend(labelColor=INK_PRIMARY, titleColor=INK_PRIMARY)
        )
        st.altair_chart(chart, width="stretch")
    except ImportError:
        pivot = df.pivot(index="budget_pct", columns="method_label", values="recall_all_pct")
        st.line_chart(pivot)

    st.caption(
        "RiskAware-ASR (bold line, ours) is above both single-model baselines "
        "at every budget, but on real speech the lead over raw confidence is "
        "statistically established only at the 5% budget (see below). "
        "\"Disagreement + confidence\" (dashed) needs two ASR models run and "
        "compared -- a different, costlier class of method -- and catches far "
        "more critical errors."
    )

    if boot is not None:
        lines_txt = []
        for _, g in boot.sort_values("budget_pct").iterrows():
            lo, hi = g["ci_lower_pp"], g["ci_upper_pp"]
            verdict = "excludes zero" if lo > 0 else ("touches zero" if abs(lo) < 1e-9 else "includes zero")
            lines_txt.append(f"{int(g['budget_pct'])}%: {g['bootstrap_mean_gap_pp']:+.1f} pp "
                             f"(95% CI [{lo:.1f}, {hi:.1f}], {verdict})")
        st.caption("RiskAware-ASR minus raw confidence, paired whole-encounter bootstrap "
                   "(2,000 resamples): " + "; ".join(lines_txt) + ".")

    row10 = df[(df["method"] == "RiskAware-ASR") & (df["budget_pct"] == 10)]
    if not row10.empty:
        flagged10 = int(row10.iloc[0]["flagged_words"])
        st.caption(
            f"In real terms: at a 10% budget across the {N_ENCOUNTERS}-clip validation "
            f"set, that's reviewing about {flagged10 / N_ENCOUNTERS:.0f} words per "
            f"encounter, out of roughly {N_TOTAL_WORDS / N_ENCOUNTERS:.0f} words total -- "
            f"not the whole note, just the risky part of it."
        )

    st.divider()
    render_confusion_matrix(df)

    st.divider()
    with st.expander("Full ten-method comparison table (incl. simpler glossary heuristics)"):
        st.caption("A1/A2 flag only the 1,175 glossary matches at every budget, so their "
                   "review effort is not comparable with the full-budget methods.")
        display_cols = ["method_label", "budget_pct", "critical_caught", "recall_all_pct", "precision_pct"]
        st.dataframe(
            full[display_cols].rename(columns={
                "method_label": "Method", "budget_pct": "Budget %",
                "critical_caught": "Critical errors caught",
                "recall_all_pct": "Recall %", "precision_pct": "Precision %",
            }).sort_values("Budget %"),
            hide_index=True, width="stretch",
        )


def transcribe(audio_path: str):
    model = load_whisper_model()
    result = model.transcribe(
        audio_path,
        task="transcribe",
        language="en",
        word_timestamps=True,
        fp16=False,
        verbose=False,
    )
    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "confidence": float(w.get("probability", 0.0)),
            })
    return result["text"].strip(), words


def rows_to_scored(rows_df, pipeline):
    """Turn rows of the pre-scored sample CSV into the same shape
    RiskAwareASRPipeline.score_words() returns, so rendering is identical
    for both modes."""
    tier_labels = pipeline.tier_labels
    out = []
    for _, r in rows_df.iterrows():
        word = str(r["whisper_word"]) if pd.notna(r["whisper_word"]) else ""
        flagged = bool(r["flagged"])
        reason = None
        suggestion = None
        if flagged:
            reason = explain_reason(r["confidence"], r["shap_confidence"], r["word_len"],
                                    r["shap_word_len"], int(r["tier_whisper"]), tier_labels)
            suggestion = suggest_correction(word, pipeline.candidates, pipeline.dictionary)
        action = "OK"
        if flagged:
            action = f"SUGGEST: {suggestion[0]}" if suggestion else "ESCALATE TO HUMAN REVIEW"
        out.append({
            "word": word,
            "gold_word": r.get("gold_word", ""),
            "is_error": bool(r.get("is_error", 0)),
            "risk_score": round(float(r["risk_score"]), 3),
            "flagged": flagged,
            "action": action,
            "alternate_suggestions": suggestion[1:] if suggestion else [],
            "reason": reason,
            "shap_confidence": float(r["shap_confidence"]) if flagged else None,
            "tier_whisper": int(r["tier_whisper"]),
            "shap_word_len": float(r["shap_word_len"]) if flagged else None,
            "shap_bias": float(r["shap_bias"]) if flagged else None,
        })
    return out


def risk_color(risk_score: float, flagged: bool) -> str:
    if not flagged:
        return "inherit"
    return RISK_TIERS[risk_tier(risk_score)]["color"]


def risk_legend_html(extra: str = "") -> str:
    """Builds the color legend using the exact hex values the cards and
    transcript spans actually use -- not Streamlit's built-in :color[] markdown,
    which only offers a handful of fixed colors and can't represent the real
    (distinct) coral vs. amber hexes."""
    parts = [
        f'<span style="color:{INK_SECONDARY};">Hover a flagged word to see why. '
        f'<span style="color:{RISK_TIERS["critical"]["color"]};font-weight:600;">Red</span> = high risk &middot; '
        f'<span style="color:{RISK_TIERS["serious"]["color"]};font-weight:600;">Coral</span> = medium &middot; '
        f'<span style="color:{RISK_TIERS["warning"]["color"]};font-weight:600;">Amber</span> = '
        f'flagged, lower risk.{extra}</span>'
    ]
    return "".join(parts)


def render_transcript(scored_words):
    spans = []
    for r in scored_words:
        color = risk_color(r["risk_score"], r["flagged"])
        title = (r["reason"] or "").replace('"', "&quot;")
        weight = "font-weight:700;" if r["flagged"] else ""
        spans.append(
            f'<span title="{title}" '
            f'style="color:{color};{weight}padding:1px 2px;">{r["word"]}</span>'
        )
    return " ".join(spans)


def render_shap_breakdown(r):
    """Small diverging bar readout of how each feature pushed this one
    word's learned uncertainty up or down, then the tier weight that multiplies
    it. Needs shap_confidence/shap_word_len/shap_bias/tier_whisper on r -- callers that don't have them (nothing
    currently omits them for flagged words) just get nothing rendered.

    Hand-built HTML bars, not an Altair chart: with only 4 rows and two
    colors, a full charting library buys nothing here, and this specific
    chart (categorical y-axis, 4 rows, mixed-sign values) reproducibly hit
    a Vega-Lite rendering bug in testing -- rows merging into each other,
    sometimes as overlapping bars, sometimes as overlapping axis labels.
    Confirmed via a standalone repro outside the app before switching."""
    if r.get("shap_confidence") is None:
        return

    features = [
        ("Confidence", r["shap_confidence"]),
        ("Word length", r["shap_word_len"]),
        ("Base rate", r["shap_bias"]),
    ]
    max_abs = max(abs(v) for _, v in features) or 1.0

    rows_html = []
    for name, val in features:
        pct = min(100.0, abs(val) / max_abs * 100.0)
        color = "#e34948" if val >= 0 else "#2a78d6"
        rows_html.append(f"""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
  <div style="width:112px; font-size:0.82em; color:{INK_SECONDARY}; text-align:right; flex-shrink:0;">{html.escape(name)}</div>
  <div style="flex:1; background:{SUBTLE_FILL}; border-radius:4px; height:16px;">
    <div style="width:{pct:.1f}%; background:{color}; height:100%; border-radius:4px;"></div>
  </div>
  <div style="width:52px; font-size:0.8em; color:{INK_PRIMARY}; font-weight:600; text-align:right;">{val:+.2f}</div>
</div>""")

    legend = (
        f'<div style="font-size:0.78em; color:{INK_MUTED}; margin-top:2px;">'
        f'<span style="color:#e34948;">&#9632;</span> pushes toward flagging &nbsp;&nbsp; '
        f'<span style="color:#2a78d6;">&#9632;</span> pushes away from flagging</div>'
    )
    tier = int(r.get("tier_whisper", 1))
    weight_note = (
        f'<div style="font-size:0.84em; color:{INK_PRIMARY}; margin-top:10px;">'
        f'Learned uncertainty (from the two inputs above) &times; '
        f'<b>{GLOSSARY_TIER_LABELS.get(tier, "Everyday").lower()} weight {TIER_WEIGHTS[tier]:.2f}</b> '
        f'= risk score {r["risk_score"]}. The glossary tier is not a model input.</div>'
    )
    st.markdown("".join(rows_html) + legend + weight_note, unsafe_allow_html=True)


def render_flagged_section(scored, show_gold: bool):
    st.divider()
    flagged = [r for r in scored if r["flagged"]]
    total = len(scored)
    pct = 100 * len(flagged) / total if total else 0
    max_risk = max((r["risk_score"] for r in scored), default=0.0)

    st.subheader(f"Flagged for review ({len(flagged)} of {total} words)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words processed", total)
    m2.metric("Flagged", len(flagged))
    m3.metric("Flag rate", f"{pct:.1f}%",
              delta=f"{pct - 10:+.1f}pp vs. paper's 10% budget", delta_color="off")
    m4.metric("Highest risk score", f"{max_risk:.2f}")

    st.progress(
        min(pct / 100, 1.0),
        text=f"Reviewing {len(flagged)} of {total} words ({pct:.1f}%) instead of the full transcript",
    )

    st.caption(
        f"Fixed threshold {FLAG_RISK_THRESHOLD} = the top-10% operating point on the 42 "
        "held-out ACI-Bench encounters (precision 0.535, critical-error recall 0.556)."
    )
    if not flagged:
        st.info("No words crossed the flagging threshold in this clip.")
        return
    flagged_sorted = sorted(flagged, key=lambda r: r["risk_score"], reverse=True)
    for r in flagged_sorted:
        tier = risk_tier(r["risk_score"])
        border_color = RISK_TIERS[tier]["color"]
        tier_label = RISK_TIERS[tier]["label"]
        badge = BADGE_GOOD if r["action"].startswith("SUGGEST") else BADGE_CRITICAL
        word = html.escape(r["word"])
        action = html.escape(r["action"])
        reason = html.escape(r["reason"] or "")

        card = f"""
<div style="border-left:4px solid {border_color}; background:{CARD_BG};
            border-radius:8px; padding:14px 18px; margin-bottom:10px;
            display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
  <div style="min-width:160px;">
    <span style="font-size:1.08em; font-weight:700; color:{INK_PRIMARY};">{word}</span><br>
    <span style="color:{border_color}; font-size:0.78em; font-weight:600;
                text-transform:uppercase; letter-spacing:0.04em;">{tier_label}</span>
    <span style="color:{INK_MUTED}; font-size:0.82em;"> &middot; risk={r['risk_score']}</span>
  </div>
  <div>
    <span style="background:{badge['bg']}; color:{badge['fg']}; padding:4px 12px;
                border-radius:999px; font-family:monospace; font-size:0.78em;
                font-weight:600; white-space:nowrap;">{action}</span>
  </div>
  <div style="flex:1; min-width:220px; color:{INK_SECONDARY}; font-size:0.88em;">{reason}</div>
</div>
"""
        st.markdown(card, unsafe_allow_html=True)
        if r["alternate_suggestions"]:
            st.caption(f"Other options: {', '.join(r['alternate_suggestions'])}")
        if show_gold and r["is_error"]:
            st.caption(f"Reference check (not used by the system): "
                       f"ASR said \"{r['word']}\", reference was \"{r['gold_word']}\".")
        if r.get("shap_confidence") is not None:
            with st.expander(f"Why '{r['word']}' scored {r['risk_score']}"):
                render_shap_breakdown(r)


def build_annotated_corrected_text(scored_words):
    """Reconstruct the transcript with the pipeline's own corrections
    applied -- but never silently: every change is shown inline as
    'spiriva (heard as: spareva)', using exactly the same action field the
    flagged-word cards above already display, so this can't disagree with
    what the reviewer sees there. Words the pipeline couldn't confidently
    correct (ESCALATE TO HUMAN REVIEW) are left as the raw ASR word."""
    tokens = []
    for r in scored_words:
        word = r["word"]
        if r["flagged"] and r["action"].startswith("SUGGEST: "):
            corrected = r["action"].split("SUGGEST: ", 1)[1]
            tokens.append(f"{corrected} (heard as: {word})")
        else:
            tokens.append(word)
    text = " ".join(tokens)
    return (text[0].upper() + text[1:]) if text else text


def get_groq_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            key = None
    if not key:
        key = st.session_state.get("_groq_api_key_input")
    return key


def generate_soap_note(transcript_text, api_key):
    from groq import Groq
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        model=SOAP_MODEL,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SOAP_SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
    )
    return chat_completion.choices[0].message.content


HEARD_AS_PATTERN = re.compile(r"\(heard as:\s*([^)]+)\)")


def style_heard_as_annotations(note_text):
    """Turns each '(heard as: spareva)' the model kept into a small pill,
    so the correction-transparency promise in the caption above is visible
    at a glance in the note itself, not just readable on close inspection."""
    def repl(m):
        heard = html.escape(m.group(1).strip())
        return (
            f'<span style="background:{SUBTLE_FILL}; color:{INK_SECONDARY}; '
            f'font-size:0.8em; padding:2px 8px; border-radius:999px; '
            f'border:1px solid {SUBTLE_BORDER}; white-space:nowrap;">'
            f'heard as: {heard}</span>'
        )
    return HEARD_AS_PATTERN.sub(repl, note_text)


def render_note_card(note_text):
    styled = style_heard_as_annotations(note_text)
    st.markdown(
        f'<div style="background:{CARD_BG}; border:1px solid {SUBTLE_BORDER}; '
        f'border-left:4px solid {BRAND_ACCENT}; border-radius:10px; '
        f'padding:6px 28px 18px 24px; margin-top:6px;">\n\n{styled}\n\n</div>',
        unsafe_allow_html=True,
    )


def render_soap_section(scored_words, key_prefix):
    """Optional final step: draft a SOAP note from the corrected transcript.
    Deliberately isolated from the rest of the demo -- nothing above this
    depends on it, and if it fails (no key, no internet, API error) it
    falls back to a static saved example rather than breaking the page."""
    st.divider()
    st.subheader("Draft clinical note (optional)")
    st.markdown(
        '<span style="background:#d03b3b; color:#ffffff; padding:3px 10px; '
        'border-radius:6px; font-size:0.78rem; font-weight:700; '
        'text-transform:uppercase; letter-spacing:0.03em;">AI-generated draft '
        "&mdash; not clinical advice</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Generated from the transcript with the pipeline's own corrections "
        "applied. Every correction is shown inline as 'drug (heard as: "
        "what ASR heard)' so nothing changes silently."
    )

    corrected_text = build_annotated_corrected_text(scored_words)

    env_api_key = get_groq_api_key()
    note_state_key = f"_soap_note_{key_prefix}"
    error_state_key = f"{note_state_key}_error"

    # A form, not a bare text_input + button: Streamlit only guarantees a
    # text_input's typed value is committed to session_state on blur/Enter,
    # so a paste immediately followed by a button click can race and submit
    # before the key value lands -- this produced a real "no key" error
    # during testing even though a key had just been typed. A form commits
    # every field in it atomically with the submit click, so that race is
    # impossible here.
    with st.form(key=f"_soap_form_{key_prefix}"):
        manual_key = ""
        if not env_api_key:
            manual_key = st.text_input(
                "Groq API key (free at console.groq.com -- used only for this session, never stored)",
                type="password",
            )
        submitted = st.form_submit_button("Generate SOAP note")

    if st.button("Show static example instead", key=f"_soap_fallback_btn_{key_prefix}"):
        st.session_state[error_state_key] = None
        if FALLBACK_SOAP_NOTE_PATH.exists():
            st.session_state[note_state_key] = FALLBACK_SOAP_NOTE_PATH.read_text()
        else:
            st.session_state[note_state_key] = None

    if submitted:
        api_key = (env_api_key or manual_key or "").strip()
        st.session_state[error_state_key] = None
        if not api_key:
            st.warning("Enter an API key above first, or use the static example instead.")
        else:
            with st.spinner("Drafting note..."):
                try:
                    st.session_state[note_state_key] = generate_soap_note(corrected_text, api_key)
                except Exception as e:
                    st.session_state[error_state_key] = str(e)
                    if FALLBACK_SOAP_NOTE_PATH.exists():
                        st.session_state[note_state_key] = FALLBACK_SOAP_NOTE_PATH.read_text()

    if st.session_state.get(error_state_key):
        st.warning(f"Live generation failed ({st.session_state[error_state_key]}) -- "
                   f"showing a static fallback instead.")

    note = st.session_state.get(note_state_key)
    if note:
        render_note_card(note)

    with st.expander("Transcript sent to the note generator"):
        st.write(corrected_text)


header_col, glossary_col = st.columns([7, 1])
with header_col:
    st.markdown(
        f"""
<div style="display:flex; align-items:center; gap:14px; border-bottom:2px solid {BRAND_ACCENT};
            padding-bottom:14px; margin-bottom:6px;">
  <div style="font-size:2.1rem; line-height:1;">🩺</div>
  <div>
    <div style="font-size:1.7rem; font-weight:700; color:{INK_PRIMARY}; line-height:1.2;">
      RiskAware-ASR — Live Demo
    </div>
    <div style="color:{INK_SECONDARY}; font-size:0.95rem;">
      Reference-free risk scoring for clinical transcription safety
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with glossary_col:
    with st.popover("📖 Glossary"):
        st.markdown(
            "**Reference-free** — the system never sees the correct transcript; "
            "it decides what to flag using only the ASR output itself.\n\n"
            "**Glossary tier** — how clinically important a term is: *everyday* "
            "(tier 1), *clinical term* (tier 2), *drug/dosage* (tier 3, "
            "weighted highest).\n\n"
            "**Risk score** — learned uncertainty × tier weight. Higher means "
            "more likely to be a dangerous, undetected error.\n\n"
            "**Critical-error recall** — of all the clinically dangerous "
            "errors that actually exist, what fraction the system caught.\n\n"
            "**Precision** — of everything flagged, what fraction was an "
            "actual error (vs. a false alarm).\n\n"
            "**Contribution (SHAP-style)** — how much one input feature "
            "pushed a specific word's learned uncertainty up or down."
        )

st.caption(
    "Every word is scored as learned uncertainty × clinical-criticality weight "
    "(ASR confidence and word length are the only model inputs; the 176-term "
    "glossary tier enters only as the weight) and flagged "
    "for human review above a fixed, data-calibrated threshold."
)

with st.expander("How to use this demo"):
    st.markdown(
        "1. **Upload audio (live)** — upload a short clip; Whisper transcribes "
        "it and the pipeline flags risky words in real time, with no reference "
        "transcript involved anywhere.\n"
        "2. **Load a saved sample** — no audio needed; see the system run on a "
        "real, already-scored clinical encounter.\n"
        "3. **Validation results** — the paper's evaluation on 272 real "
        "recordings: RiskAware-ASR against raw confidence, learned uncertainty, "
        "cross-model disagreement and five simpler glossary heuristics."
    )

mode = st.radio(
    "Input source",
    ["Upload audio (live)", "Load a saved sample (no audio needed)", "Validation results (paper)"],
    horizontal=True,
)
st.divider()

if mode == "Upload audio (live)":
    pipeline = load_risk_pipeline()
    st.caption("True reference-free run: no gold transcript is used anywhere below.")

    source = st.radio(
        "Audio source", ["Upload a file", "Record from microphone"],
        horizontal=True, label_visibility="collapsed",
    )

    if source == "Upload a file":
        uploaded = st.file_uploader(
            "Upload audio (mp3, wav, m4a)", type=["mp3", "wav", "m4a", "mp4"]
        )
    else:
        st.caption("Read a prepared sentence aloud — e.g. one with a drug name or "
                    "dosage in it — then click stop.")
        uploaded = st.audio_input("Record a clip")

    if uploaded is not None:
        st.audio(uploaded)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded.name).suffix
        ) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        with st.spinner("Transcribing with Whisper..."):
            transcript_text, words = transcribe(tmp_path)

        if not words:
            st.warning("No words detected in this clip. Try a longer or clearer recording.")
        else:
            with st.spinner("Scoring words..."):
                scored = pipeline.score_words(words)
                # Tiers + SHAP-style contributions, computed straight from the
                # pipeline's own public glossary/phonetic_map/booster -- not a
                # reimplementation, the exact same xgboost call score_words()
                # makes internally. Verified byte-identical against a known
                # live result before shipping.
                tiers = [lookup_tier(w["word"], pipeline.glossary, pipeline.phonetic_map) for w in words]
                feat_df = pd.DataFrame({
                    "confidence": [w["confidence"] for w in words],
                    "word_len": [len(w["word"]) for w in words],
                })
                dmatrix = xgb.DMatrix(feat_df[pipeline.feature_names], feature_names=pipeline.feature_names)
                contribs = pipeline.booster.predict(dmatrix, pred_contribs=True)
                for i, s in enumerate(scored):
                    if s["flagged"]:
                        s["shap_confidence"] = float(contribs[i][0])
                        s["shap_word_len"] = float(contribs[i][1])
                        s["shap_bias"] = float(contribs[i][2])
                        s["tier_whisper"] = int(tiers[i])
                    else:
                        s["shap_confidence"] = None

            st.subheader("Transcript (color = flagged risk)")
            st.markdown(render_transcript(scored), unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.875rem;">{risk_legend_html()}</div>',
                        unsafe_allow_html=True)
            render_flagged_section(scored, show_gold=False)
            render_soap_section(scored, key_prefix="live")

            with st.expander("Why these words were flagged (confidence vs. risk, this clip)"):
                scatter_df = pd.DataFrame({
                    "word": [w["word"] for w in words],
                    "confidence": [w["confidence"] for w in words],
                    "tier_whisper": tiers,
                    "risk_score": [s["risk_score"] for s in scored],
                })
                render_confidence_scatter(scatter_df)

            with st.expander("Full transcript text"):
                st.write(transcript_text)

elif mode == "Load a saved sample (no audio needed)":
    pipeline = load_risk_pipeline()
    df, counts = load_sample_data()
    if counts.empty:
        st.error("No flagged Tier-3 encounters found in the sample data.")
    else:
        options = list(counts.index[:5])
        chosen = st.selectbox(
            "Encounter (ranked by number of drug/dosage-term flags)",
            options,
            format_func=lambda eid: f"{eid}  —  {counts[eid]} Tier-3 flags",
        )
        rows = df[df["encounter_id"] == chosen]
        scored = rows_to_scored(rows, pipeline)

        st.subheader(f"Transcript — {chosen} (color = flagged risk)")
        st.markdown(render_transcript(scored), unsafe_allow_html=True)
        extra = (" This is ACI-Bench + synthetic speech, already processed -- scored by "
                 "the same model, no live transcription step here.")
        st.markdown(f'<div style="font-size:0.875rem;">{risk_legend_html(extra)}</div>',
                    unsafe_allow_html=True)
        render_flagged_section(scored, show_gold=True)
        render_soap_section(scored, key_prefix=f"sample_{chosen}")

        with st.expander("Why these words were flagged (confidence vs. risk, this encounter)"):
            scatter_df = rows.rename(columns={"whisper_word": "word"})[
                ["word", "confidence", "tier_whisper", "risk_score"]
            ]
            render_confidence_scatter(scatter_df)

else:
    render_validation_section()
