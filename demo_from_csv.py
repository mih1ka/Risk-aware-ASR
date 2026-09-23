import pandas as pd

FLAG_RISK_THRESHOLD = 0.839
TIER_LABELS = {1: "everyday", 2: "clinical term", 3: "drug/dosage"}

df = pd.read_csv("data/processed/risk_scores_explained.csv")
df["flagged"] = df["risk_score"] >= FLAG_RISK_THRESHOLD

tier3_flags = df[(df["flagged"]) & (df["tier_whisper"] == 3)]
best_encounter = tier3_flags.groupby("encounter_id").size().idxmax()
enc = df[df["encounter_id"] == best_encounter].copy()
flagged = enc[enc["flagged"]].sort_values("risk_score", ascending=False)

print(f"Encounter: {best_encounter}")
print(f"Processed {len(enc)} words, flagged {len(flagged)} for review\n")

for _, r in flagged.head(15).iterrows():
    reason = (f"confidence={r['confidence']:.2f} (contributed {r['shap_confidence']:+.2f}), "
              f"tier={TIER_LABELS.get(int(r['tier_whisper']), 'unknown')} "
              f"(contributed {r['shap_tier_whisper']:+.2f})")
    print(f"'{r['whisper_word']}'  (gold: '{r['gold_word']}')  risk={r['risk_score']:.3f}  error_type={r['error_type']}")
    print(f"    reason: {reason}")
    print()
