# RiskAware-ASR

A reference-free risk-scoring pipeline for clinical transcription safety. Instead of
requiring a gold transcript at inference time, this system estimates per-word
transcription uncertainty from ASR confidence alone, combines it with a tiered
clinical/drug-term criticality score, flags high-risk words for mandatory review,
and suggests corrections from a constrained clinical glossary where confident enough
to do so.

## Core idea

    Risk Score(word) = Learned Uncertainty(word) × Domain Criticality(word)

Trained and evaluated on ACI-Bench (Yim et al., 2023) doctor-patient dialogues.
Full methodology, decisions, and dated findings are in `report/decisions_log.md`.

## Headline results

- Tier 3 (drug/dosage) terms are mistranscribed **90.2%** of the time by a
  general-purpose Whisper-base pipeline, vs. **12.6%** for everyday words
  (see `report/figures/fig1_wer_by_tier.png`)
- Reference-free classifiers (confidence + criticality + word length) predict
  transcription errors with ROC-AUC ~0.86 with no access to the gold transcript
- Risk-weighted flagging beats confidence-alone flagging on catching critical
  errors specifically (bootstrap 95% CI [0.9, 6.0]pp at a 10% review budget)
- The constrained correction module recovers ~93% of genuine phonetic
  mistranscriptions of critical terms, and correctly declines to guess on
  unrecoverable ones rather than presenting a bad guess as reliable
- This pattern is specific to general-purpose ASR (e.g. Whisper) without medical
  fine-tuning — validated against real ASR-vs-human-transcript pairs from
  ACI-Bench, where a production-grade system shows no such tier gap

## Setup

    conda create -n riskaware-asr python=3.11 -y
    conda activate riskaware-asr
    pip install -r requirements.txt
    conda install -c conda-forge llvm-openmp -y   # needed for XGBoost on macOS

Requires the ACI-Bench dataset cloned into `data/raw_aci_bench/` (see Yim et al.,
2023 for access).

## Reproducing the full pipeline (in order)

**1. Ingest data**

    python src/ingest/load_aci_bench.py            # -> data/processed/turns.jsonl
    python src/ingest/build_encounter_texts.py      # -> data/processed/encounter_texts.jsonl

**2. Build the tiered clinical glossary**

    python src/glossary/extract_medical_candidates.py   # candidate terms (manual curation step)
    python src/glossary/build_glossary.py                # -> data/glossary/tiered_glossary.csv

**3. Synthesize audio + transcribe (Google Colab, GPU required)**

Upload `data/processed/encounter_texts.jsonl` to a Colab session and run
`notebooks/colab_tts_whisper.py` (see the header comment in that file for the
exact cell order). This produces one JSON file per encounter (confidence-scored
Whisper transcript). Download the resulting `whisper_out/` folder from Google
Drive into `data/whisper_out/` locally.

**4. Align, label, and evaluate**

    python src/eval/align_and_label.py                  # -> data/processed/word_level_labels.csv
    python src/model/train_uncertainty_classifier.py    # baseline classifier comparison
    python src/model/risk_score.py                      # -> models/uncertainty_xgb.joblib, risk_scores_test.csv
    python src/model/bootstrap_significance.py          # statistical significance of the risk-score ablation
    python src/model/correct_flagged_words.py           # correction module evaluation
    python src/eval/error_taxonomy.py                   # -> data/processed/word_level_labels_taxonomy.csv
    python src/model/explain_predictions.py             # -> data/processed/risk_scores_explained.csv
    python src/eval/validate_real_asr.py                # validation against real ASR-vs-human pairs
    python src/eval/make_figures.py                     # -> report/figures/*.png

**5. Run the live system**

    python src/pipeline/risk_pipeline.py

Runs the full integrated pipeline (risk scoring + flagging + correction +
explanation) end-to-end on one saved encounter as a demo.

## Repo structure

    src/
      ingest/     dataset parsing and text preparation
      glossary/   tiered clinical/drug glossary construction
      eval/       alignment, error taxonomy, real-data validation, figures
      model/      uncertainty classifier, risk scoring, correction, explainability
      pipeline/   the integrated end-to-end system (risk_pipeline.py)
    notebooks/    Colab-only scripts (GPU-dependent TTS + Whisper batch step)
    data/         raw and processed data (large/derived files gitignored)
    models/       trained model artifacts
    report/       decisions_log.md (full methodology + dated findings) and figures/

`src/ingest/find_fused_words.py` is an early exploratory diagnostic (used once
to check the dataset for a specific text artifact) — not part of the
reproducible pipeline above.