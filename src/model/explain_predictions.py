from pathlib import Path

import pandas as pd
import joblib
import xgboost as xgb

MODEL_PATH = Path("models/uncertainty_xgb.joblib")
RISK_SCORES_PATH = Path("data/processed/risk_scores_test.csv")
FEATURE_NAMES = ["confidence", "tier_whisper", "word_len"]
TOP_N_EXAMPLES = 10


def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(RISK_SCORES_PATH)

    X = df[FEATURE_NAMES]
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_NAMES)

    # Built-in tree SHAP: one contribution column per feature, plus a final
    # "bias" column — they sum to the model's raw output for that row.
    contribs = booster.predict(dmatrix, pred_contribs=True)
    contrib_df = pd.DataFrame(contribs, columns=FEATURE_NAMES + ["bias"])

    df = pd.concat([df.reset_index(drop=True), contrib_df.add_prefix("shap_")], axis=1)

    out_path = Path("data/processed/risk_scores_explained.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved SHAP-explained scores to {out_path}\n")

    top_flagged = df.sort_values("risk_score", ascending=False).head(TOP_N_EXAMPLES)
    tier_labels = {1: "everyday", 2: "clinical term", 3: "drug/dosage"}

    print(f"=== Explanations for top {TOP_N_EXAMPLES} highest-risk flagged words ===\n")
    for _, row in top_flagged.iterrows():
        print(f"Word: '{row['whisper_word']}'  (true word: '{row['gold_word']}')")
        print(f"  Risk score: {row['risk_score']:.3f}  |  Model uncertainty: {row['learned_uncertainty']:.3f}")
        print(f"  Confidence: {row['confidence']:.3f}  (SHAP contribution: {row['shap_confidence']:+.3f})")
        print(f"  Tier: {int(row['tier_whisper'])} - {tier_labels.get(int(row['tier_whisper']), 'unknown')}  "
              f"(SHAP contribution: {row['shap_tier_whisper']:+.3f})")
        print(f"  Word length: {int(row['word_len'])}  (SHAP contribution: {row['shap_word_len']:+.3f})")
        print()


if __name__ == "__main__":
    main()