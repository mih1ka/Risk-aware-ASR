"""
R1 steps 3-4 - Whisper-base on Fareez (272), 190-term vs 176-term glossary.

For each glossary, the same file is used for the gold tier labels (alignment)
and for the inference-time tier lookup inside RiskAware (tier_whisper).
Model: models/uncertainty_xgb_no_tier.joblib (read-only).

Writes r1_tier_rates.csv and r1_recall.csv (aggregates only).
"""
import joblib
import pandas as pd

from common import (OUT_DIR, GLOSSARY_190, GLOSSARY_176, FAREEZ_LABELS_190, FAREEZ_LABELS_176,
                    NO_TIER_MODEL_PATH, load_glossary, read_labels, tier_rates_with_ci,
                    recall_table)


def main():
    model = joblib.load(NO_TIER_MODEL_PATH)
    tier_frames, recall_frames = [], []
    for name, gpath, lpath in [("190", GLOSSARY_190, FAREEZ_LABELS_190),
                               ("176", GLOSSARY_176, FAREEZ_LABELS_176)]:
        glossary = load_glossary(gpath)
        labels = read_labels(lpath)
        tiers, n_enc = tier_rates_with_ci(labels)
        tiers.insert(0, "glossary_terms", name)
        tier_frames.append(tiers)
        rec = recall_table(labels, glossary, model)
        rec.insert(0, "glossary_terms", name)
        recall_frames.append(rec)
        print(f"glossary={name} encounters={n_enc}")

    tiers = pd.concat(tier_frames, ignore_index=True)
    recall = pd.concat(recall_frames, ignore_index=True)
    tiers.round(4).to_csv(OUT_DIR / "r1_tier_rates.csv", index=False)
    recall.round(4).to_csv(OUT_DIR / "r1_recall.csv", index=False)

    pd.set_option("display.width", 200)
    print(tiers.round(2).to_string(index=False))
    wide = recall.pivot_table(index=["ranking", "budget_pct"], columns="glossary_terms",
                              values=["caught", "recall_scorable_pct", "recall_all_pct"])
    print(wide.round(2).to_string())
    print(recall.groupby("glossary_terms")[["crit_errors_scorable", "crit_errors_all"]].first())


if __name__ == "__main__":
    main()
