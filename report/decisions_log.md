# RiskAware-ASR — Decisions Log

Running record of methodology decisions and why they were made, kept
for the final report's methodology/limitations sections.

## Environment
- Conda env `riskaware-asr`, Python 3.11 — switched from the system default
  (3.14) after several ML packages (spacy/blis) failed to build against it.
  Locking to a well-supported Python version early avoided repeated
  compatibility issues.

## Dataset access
- ACI-Bench (Yim et al., 2023) is openly licensed (CC BY 4.0) on GitHub —
  no data use agreement required. It is text-only; no source audio is
  provided, which is why a TTS-synthesis step is needed to generate audio
  for the ASR pipeline.
- Repo trimmed of the original authors' own baseline-prediction files
  (BART/LED/GPT-4 note-generation outputs) — unrelated to this project's
  scope. `challenge_data` and `src_experiment_data` were kept.

## Key finding: real ASR-error pairs exist in the dataset
- ACI-Bench has three dialogue creation modes: `aci` (raw ASR transcript
  only), `virtassist` (human transcription only), `virtscribe` (both raw
  ASR and human-corrected versions of the same encounters).
- `src_experiment_data/` contains ~150 real, human-verified ASR-vs-corrected
  pairs (e.g. `virtscribe_asr.csv` / `virtscribe_humantrans.csv`).
- Decision: use these as validation for the error taxonomy and as a
  discussion point (do synthetic Whisper errors resemble real clinical ASR
  errors?), not as a substitute for the synthetic pipeline — no per-word
  confidence scores exist for the original ASR system behind these pairs,
  so they can't supply the classifier's input features.

## Phase 0 — proof of concept
- gTTS + Whisper (`base`) on real dataset sentences.
- Trial 1 (clinical condition names: "congestive heart failure",
  "hypertension") showed no confidence drop — these are common enough
  words that Whisper transcribed them cleanly.
- Trial 2 (drug names + dosages) worked: "lisinopril" → mistranscribed as
  "lysiniprol", confidence 0.43 vs ~0.95+ elsewhere in the sentence.
  Confirms the core mechanism. Also observed: correctly-transcribed-but-
  still-uncertain words (e.g. "LASIX" at 0.61), and noisy confidence on
  non-clinical filler speech — evidence for why the risk score fuses
  uncertainty with clinical criticality rather than using either alone.

## Phase 1 — dataset parsing and glossary
- `turns.jsonl`: 11,303 doctor/patient turns parsed from all 5 splits.
- Glossary built from RxNorm + dataset vocabulary rather than RxNorm alone:
  RxNorm's full name list (28,084 entries) is mostly obscure raw chemical
  names not relevant to this dataset. Instead, extracted actual
  out-of-dictionary words from the 11,303 turns, cross-referenced against
  RxNorm for Tier 3 (136 auto-confirmed drug names + 2 added by hand that
  RxNorm's exact-match missed: flexeril, zofran), and manually curated
  Tier 2 (40 clinical/anatomical/procedural terms) from the remainder.
  176 total entries in `tiered_glossary.csv`.
- Scoping decision: dropped the separate FDA NDC glossary source from the
  original plan — RxNorm + the dataset-driven approach already covers the
  terms that actually appear in this dataset.

## Open / upcoming decisions
- Phase 2 TTS/Whisper batching: per-encounter, not per-turn (see below).