"""
Fact-level evaluation of SOAP-note generation from three transcript conditions:
  raw       -- unmodified Whisper ASR transcript (what the model actually heard)
  corrected -- the risk pipeline's own corrections applied, with every change
               shown inline as "spiriva (heard as: spareva)" -- never a silent edit
  gold      -- the true dialogue transcript (ceiling / upper bound)

For each of ~20 sampled encounters (from the 42 in data/processed/
risk_scores_explained.csv, all of which are also in ACI-Bench's
challenge_data and so have a real reference note), a SOAP note is generated
from each condition with Groq's free API (openai/gpt-oss-120b), then checked
against a per-encounter checklist of clinical facts: every glossary tier-2
("clinical term") and tier-3 ("drug/dosage") word that was actually said
(gold_word), deduplicated. A fact "passes" if the checklist term (or its
pipeline-suggested correction, for the corrected/raw conditions) appears in
the generated note, case-insensitively. The same checklist is also run
against ACI-Bench's own human-written reference note, as a sanity baseline.

This is a small, honest check, not a reproduction of ACI-Bench's own
(UMLS-based) evaluation -- it reports raw counts, not a polished metric.

Groq's free plan caps throughput at 8,000 tokens/minute (30 req/min), and
each call here runs close to 3,000 tokens, so calls are paced 25s apart to
stay under that -- a full 20-encounter x 3-condition run (60 calls) takes
roughly 25 minutes, not a "couple minutes." Get a free key at
console.groq.com (no card required).

Usage:
    python run_soap_eval.py --dry-run        # no API calls; verifies the
                                              # transcript/checklist wiring
    python run_soap_eval.py                  # real run, needs
                                              # GROQ_API_KEY in the env
    python run_soap_eval.py --n 20 --seed 42 --out results.csv
"""
import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.risk_pipeline import (  # noqa: E402
    suggest_correction, load_glossary, load_dictionary, FLAG_RISK_THRESHOLD,
)

SCORES_CSV = REPO_ROOT / "data/processed/risk_scores_explained.csv"
CHALLENGE_DIR = REPO_ROOT / "data/raw_aci_bench/data/challenge_data"
CHALLENGE_FILES = [
    "train.csv", "valid.csv", "clinicalnlp_taskB_test1.csv",
    "clinicalnlp_taskC_test2.csv", "clef_taskC_test3.csv",
]

SOAP_SYSTEM_PROMPT = (
    "You are a clinical scribe. You are given a word-by-word transcript of "
    "a doctor-patient conversation (no punctuation or speaker labels). "
    "Write a concise, structured SOAP note (Subjective, Objective, "
    "Assessment, Plan). Infer speaker turns from context. If a section has "
    "no supporting content in the transcript, write 'Not documented' for "
    "that section rather than inventing detail. Preserve every medication "
    "name and dosage exactly as given in the transcript, including any "
    "'(heard as: ...)' annotations -- keep those annotations in the note "
    "wherever that term is used, so a reviewer can see what was corrected."
)


def load_reference_notes():
    frames = []
    for fname in CHALLENGE_FILES:
        frames.append(pd.read_csv(CHALLENGE_DIR / fname))
    df = pd.concat(frames, ignore_index=True)
    return dict(zip(df["encounter_id"], df["note"]))


def build_transcripts(enc_df, candidates, dictionary):
    """enc_df: rows of risk_scores_explained.csv for one encounter, in
    original word order. Returns (raw_text, corrected_text, gold_text,
    checklist_terms)."""
    raw_tokens, corrected_tokens, gold_tokens = [], [], []
    checklist = set()

    for _, r in enc_df.iterrows():
        whisper_word = r["whisper_word"]
        gold_word = r["gold_word"]
        flagged = bool(r["risk_score"] >= FLAG_RISK_THRESHOLD) if pd.notna(r["risk_score"]) else False

        if pd.notna(whisper_word):
            raw_tokens.append(str(whisper_word))
            corrected_word = str(whisper_word)
            if flagged:
                suggestion = suggest_correction(str(whisper_word), candidates, dictionary)
                if suggestion:
                    corrected_word = f"{suggestion[0]} (heard as: {whisper_word})"
            corrected_tokens.append(corrected_word)

        if pd.notna(gold_word):
            gold_tokens.append(str(gold_word))
            if pd.notna(r.get("tier")) and r["tier"] in (2, 3):
                checklist.add(str(gold_word).strip().lower())

    def join(tokens):
        text = " ".join(tokens)
        return (text[0].upper() + text[1:]) if text else text

    return join(raw_tokens), join(corrected_tokens), join(gold_tokens), sorted(checklist)


def check_terms(note_text, terms):
    note_norm = note_text.lower()
    present = []
    for t in terms:
        if t in note_norm:
            present.append(t)
    return present


def generate_note(client, model, transcript_text):
    chat_completion = client.chat.completions.create(
        model=model,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SOAP_SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
    )
    return chat_completion.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--dry-run", action="store_true",
                     help="Build transcripts/checklists and print them; no API calls.")
    ap.add_argument("--out", default=str(Path(__file__).parent / "soap_eval_results.csv"))
    args = ap.parse_args()

    scores = pd.read_csv(SCORES_CSV)
    ref_notes = load_reference_notes()
    candidates = list(load_glossary().keys())
    dictionary = load_dictionary()

    all_encounters = sorted(scores["encounter_id"].unique())
    missing_refs = [e for e in all_encounters if e not in ref_notes]
    if missing_refs:
        print(f"WARNING: {len(missing_refs)} encounters have no ACI-Bench reference "
              f"note and will be skipped: {missing_refs}")
    all_encounters = [e for e in all_encounters if e in ref_notes]

    random.seed(args.seed)
    sample = sorted(random.sample(all_encounters, min(args.n, len(all_encounters))))
    print(f"Sampled {len(sample)} of {len(all_encounters)} eligible encounters "
          f"(seed={args.seed}): {sample}")

    client = None
    if not args.dry_run:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("No GROQ_API_KEY in environment. Get a free key at console.groq.com, "
                  "export it, or pass --dry-run to just verify the transcript/checklist wiring.")
            sys.exit(1)
        from groq import Groq
        client = Groq(api_key=api_key)

    rows = []
    for eid in sample:
        enc_df = scores[scores["encounter_id"] == eid]
        raw_text, corrected_text, gold_text, checklist = build_transcripts(
            enc_df, candidates, dictionary
        )
        ref_note = ref_notes[eid]
        ref_present = check_terms(ref_note, checklist)

        print(f"\n=== {eid} -- {len(checklist)} checklist terms (tier 2/3, actually said) ===")
        print(f"checklist: {checklist}")
        print(f"reference note covers: {len(ref_present)}/{len(checklist)}")

        if args.dry_run:
            print(f"--- raw ({len(raw_text.split())} words) ---\n{raw_text[:300]}")
            print(f"--- corrected ({len(corrected_text.split())} words) ---\n{corrected_text[:300]}")
            print(f"--- gold ({len(gold_text.split())} words) ---\n{gold_text[:300]}")
            rows.append({
                "encounter_id": eid, "condition": "dry_run", "n_checklist": len(checklist),
                "n_present": None, "missing": None,
                "n_present_reference_note": len(ref_present),
            })
            continue

        for condition, text in [("raw", raw_text), ("corrected", corrected_text), ("gold", gold_text)]:
            try:
                note = generate_note(client, args.model, text)
            except Exception as e:
                print(f"  [{condition}] generation failed: {e}")
                rows.append({
                    "encounter_id": eid, "condition": condition, "n_checklist": len(checklist),
                    "n_present": None, "missing": "GENERATION_FAILED",
                    "n_present_reference_note": len(ref_present),
                })
                continue
            present = check_terms(note, checklist)
            missing = [t for t in checklist if t not in present]
            print(f"  [{condition}] {len(present)}/{len(checklist)} checklist terms present"
                  + (f" -- missing: {missing}" if missing else ""))
            rows.append({
                "encounter_id": eid, "condition": condition, "n_checklist": len(checklist),
                "n_present": len(present), "missing": "; ".join(missing),
                "n_present_reference_note": len(ref_present),
            })
            # Groq's free plan caps at 8,000 tokens/minute; the longer
            # encounters here run close to 3,000 tokens per call (input +
            # completion), so pace calls 25s apart with margin, rather than
            # hitting 429s partway through a 20-minute unattended run.
            time.sleep(25)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df)} rows to {args.out}")

    if not args.dry_run:
        summary = out_df.groupby("condition")[["n_checklist", "n_present"]].sum()
        summary["coverage"] = summary["n_present"] / summary["n_checklist"]
        print("\n=== Summary (counts across all sampled encounters) ===")
        print(summary)


if __name__ == "__main__":
    main()
