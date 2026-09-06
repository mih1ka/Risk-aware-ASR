import re
import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
import jellyfish

MODEL_PATH = Path("models/uncertainty_xgb.joblib")
GLOSSARY_PATH = Path("data/glossary/tiered_glossary.csv")
DICT_PATH = Path("/usr/share/dict/words")
TIER_WEIGHTS = {1: 1.0, 2: 3.25, 3: 4.25}
CORRECTION_THRESHOLD = 0.5

# Fixed, data-calibrated flagging threshold (the top-10% operating point
# measured on the held-out test set: precision 0.544, critical-error recall
# 0.577). Deliberately NOT recomputed per-encounter — a per-input percentile
# would always flag exactly the same fraction of words regardless of how
# risky that encounter actually is.
FLAG_RISK_THRESHOLD = 0.839


def normalize(word):
    return re.sub(r"[^a-z0-9']", "", str(word).lower())


def load_glossary():
    df = pd.read_csv(GLOSSARY_PATH)
    return dict(zip(df["term"].str.lower(), df["tier"]))


def load_dictionary():
    with open(DICT_PATH) as f:
        return set(w.strip().lower() for w in f)


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


def candidate_similarity(word, term):
    word_n, term_n = normalize(word), normalize(term)
    if not word_n or not term_n:
        return 0.0
    max_len = max(len(word_n), len(term_n))
    edit_sim = 1 - (jellyfish.levenshtein_distance(word_n, term_n) / max_len)
    phonetic_bonus = 0.5 if jellyfish.metaphone(word_n) == jellyfish.metaphone(term_n) else 0.0
    return edit_sim + phonetic_bonus


def suggest_correction(word, candidates, dictionary, threshold=CORRECTION_THRESHOLD, top_k=3):
    # Never "correct" a word that's already a recognized, ordinary English
    # word — it almost certainly wasn't a garbled critical term to begin
    # with, and suggesting a drug name in its place would be actively harmful.
    if normalize(word) in dictionary:
        return None
    scored = [(term, candidate_similarity(word, term)) for term in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored or scored[0][1] < threshold:
        return None
    return [term for term, _ in scored[:top_k]]


class RiskAwareASRPipeline:
    def __init__(self):
        self.glossary = load_glossary()
        self.dictionary = load_dictionary()
        self.phonetic_map = build_phonetic_map(self.glossary)
        self.candidates = list(self.glossary.keys())
        self.model = joblib.load(MODEL_PATH)
        self.booster = self.model.get_booster()
        self.feature_names = ["confidence", "tier_whisper", "word_len"]
        self.tier_labels = {1: "everyday", 2: "clinical term", 3: "drug/dosage"}

    def score_words(self, words):
        rows = []
        for w in words:
            word = w["word"].strip()
            confidence = w.get("confidence", 0.0) or 0.0
            tier = lookup_tier(word, self.glossary, self.phonetic_map)
            rows.append({"word": word, "confidence": confidence,
                         "tier_whisper": tier, "word_len": len(word)})

        df = pd.DataFrame(rows)
        X = df[self.feature_names]
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)

        df["learned_uncertainty"] = self.model.predict_proba(X)[:, 1]
        df["tier_weight"] = df["tier_whisper"].map(TIER_WEIGHTS)
        df["risk_score"] = df["learned_uncertainty"] * df["tier_weight"]

        contribs = self.booster.predict(dmatrix, pred_contribs=True)
        contrib_df = pd.DataFrame(contribs, columns=self.feature_names + ["bias"])
        df = pd.concat([df, contrib_df.add_prefix("shap_")], axis=1)

        df["flagged"] = df["risk_score"] >= FLAG_RISK_THRESHOLD

        results = []
        for _, row in df.iterrows():
            suggestion = None
            if row["flagged"]:
                suggestion = suggest_correction(row["word"], self.candidates, self.dictionary)

            reason = None
            if row["flagged"]:
                reason = (f"confidence={row['confidence']:.2f} (contributed {row['shap_confidence']:+.2f}), "
                          f"tier={self.tier_labels.get(int(row['tier_whisper']), 'unknown')} "
                          f"(contributed {row['shap_tier_whisper']:+.2f})")

            action = "OK"
            if row["flagged"]:
                action = f"SUGGEST: {suggestion[0]}" if suggestion else "ESCALATE TO HUMAN REVIEW"

            results.append({
                "word": row["word"],
                "risk_score": round(float(row["risk_score"]), 3),
                "flagged": bool(row["flagged"]),
                "action": action,
                "alternate_suggestions": suggestion[1:] if suggestion else [],
                "reason": reason,
            })
        return results


if __name__ == "__main__":
    demo_path = Path("data/whisper_out/D2N001.json")
    with open(demo_path) as f:
        data = json.load(f)

    pipeline = RiskAwareASRPipeline()
    results = pipeline.score_words(data["words"])

    flagged = [r for r in results if r["flagged"]]
    
    flagged_sorted = sorted(flagged, key=lambda r: r["risk_score"], reverse=True)
    print(f"Processed {len(results)} words, flagged {len(flagged)} for review\n")
    for r in flagged_sorted[:15]:
        print(f"'{r['word']}'  risk={r['risk_score']}  action={r['action']}")
        print(f"    reason: {r['reason']}")
        if r["alternate_suggestions"]:
            print(f"    other options: {r['alternate_suggestions']}")
        print()