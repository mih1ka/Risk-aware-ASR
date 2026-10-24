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

- On 272 real recorded encounters (Fareez), Whisper-base mistranscribes
  everyday / clinical / drug terms at **10.24% / 24.33% / 55.04%**;
  Whisper-small at **9.21% / 13.69% / 35.69%**
- Reference-free classifiers (confidence + criticality + word length) predict
  transcription errors with ROC-AUC **0.8535** (ACI-Bench, 5-fold CV) with no
  access to the gold transcript
- At a 10% review budget on ACI-Bench (5-fold CV), RiskAware catches
  **59.31%** of critical errors vs. **47.27%** for raw confidence (+12.04 pp)
- At a 10% review budget on Fareez, critical-error recall is **48.50%** for
  raw confidence, **58.27%** for RiskAware, and **78.57%** for
  disagreement + confidence
- The constrained correction module recovers ~93% of genuine phonetic
  mistranscriptions of critical terms, and correctly declines to guess on
  unrecoverable ones rather than presenting a bad guess as reliable
- This pattern is specific to general-purpose ASR (e.g. Whisper) without medical
  fine-tuning — validated against real ASR-vs-human-transcript pairs from
  ACI-Bench, where a production-grade system shows no such tier gap

## Reproducing the Fareez results

- `experiments/fareez_followup/fareez_followup.ipynb` (added in PR #4) runs
  the Fareez Whisper-small and cross-model experiments; run it in Google Colab
  on a T4 GPU.
- `scratch/run_align_fareez272.py` aligns and labels all 272 Fareez encounters
  into `data/processed/word_level_labels_fareez.csv`.

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