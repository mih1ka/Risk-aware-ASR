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
## Phase 3 — Word-level alignment & error labeling
- Aligned Whisper output to gold transcripts via edit-distance alignment (jiwer.process_words) across all 207 encounters → 242,505 word-level rows (data/processed/word_level_labels.csv)
- Overall WER: 15.92%
- Error rate by glossary tier: Tier 1 (everyday) 12.9% (n=235,155) · Tier 2 (clinical) 41.9% (n=356) · Tier 3 (drug/dosage) 54.7% (n=763)
- Confirms core hypothesis: domain-critical terms are mistranscribed far more often than everyday words (Tier 3 ≈ 4.2x Tier 1)

## Deliverable 1 — Reference-free uncertainty estimation
- Features (all reference-free, available at inference): Whisper word confidence, criticality tier of Whisper's own output word, word length
- Encounter-level train/test split (80/20, GroupShuffleSplit) to prevent leakage
- Baselines: Logistic Regression (ROC-AUC 0.858), Random Forest (ROC-AUC 0.780, underperforms — not tuned further), XGBoost (ROC-AUC 0.856)
- Selected XGBoost as primary model: prioritizes recall (0.807) over raw accuracy, appropriate for a safety application where missing an error is costlier than a false alarm

## Deliverable 2 — Risk-scoring mechanism
- Risk Score = Learned Uncertainty (XGBoost predicted probability) × Domain Criticality weight
- Tier weights (1.0 / 3.25 / 4.25) set proportional to Phase 3's measured tier error rates
- Initial criticality lookup used exact string match against glossary on Whisper's own output word — ablation showed this added NO improvement over uncertainty alone (identical critical-error recall), because a badly garbled critical term no longer matches the glossary string, failing exactly where the weighting matters most
- Fixed via phonetic matching (jellyfish metaphone) instead of exact string match, so garbled-but-similar-sounding words still trigger criticality weighting
- Flagging top 20% of words by risk score catches 64.1% of all transcription errors (vs. 0% flagging nothing, 100% flagging everything) — precision 46.7%
- Ablation after phonetic fix: risk_score beats uncertainty-only on critical-error recall at all cutoffs (+5.0pp at 10% threshold); bootstrap significance test (2000 resamples over 207 encounters): mean difference 4.25pp, 95% CI [0.0, 8.5]pp, one-sided p≈0.037 — directionally validated, borderline on strict two-sided significance, honestly reported as such given n=100 critical errors in the test set
## Real-ASR validation (virtscribe + aci modes, 105 real encounters, 131k words)
- Fixed a tokenization artifact: raw dialogue text represents punctuation (periods, commas, %) as space-separated tokens, which were being counted as trivial "correct" word matches, deflating measured WER — fixed by dropping tokens with no alphanumeric characters before alignment
- After the fix, Tier 1/2/3 error rates: 2.23% / 1.75% / 1.33% — Tier 2/3 are NOT elevated relative to Tier 1, contradicting the synthetic-pipeline finding
- Conclusion: the "domain-critical terms are riskier" effect appears specific to general-purpose, non-domain-adapted ASR (Whisper-base) rather than a universal ASR property — likely because whatever system produced this real reference data has vocabulary/tuning suited to clinical speech
- Scope of the paper's central claim narrowed accordingly: this work addresses risk mitigation for general-purpose ASR (e.g. Whisper) deployed in clinical settings without domain fine-tuning — a real, common, low-cost deployment pattern — not a universal ASR vulnerability
- Correction module's near-miss/dropout mechanism replicated on real data: 4 near-miss errors (100% top-1 recovery), 6 catastrophic dropouts (0% recovery) — same bimodal pattern as the synthetic pipeline, different ASR system and error source, evidence the mechanism itself generalizes even though the underlying error rate does not

## Error taxonomy (formal breakdown of critical-term errors)
- Categories: phonetic_substitution, plausible_but_wrong, unrelated_substitution, omission, insertion
- Among 566 critical (Tier 2/3) error instances: unrelated substitution 55.7%, phonetic substitution 41.0%, omission 3.4%, plausible-but-wrong 0%
- Confirms the 41% phonetic-substitution rate is the addressable fraction recoverable by the correction module (matches Deliverable 3's near-miss rate)
- Plausible-but-wrong (real-but-different drug name substituted) essentially absent — likely specific to this TTS+Whisper-base error source, not claimed as a general guarantee

## Fareez dataset ingestion (`load_fareez.py`)
- 272 audio/transcript pairs (matched 1:1 by filename stem), parsed 26,620
  doctor/patient turns into `turns.jsonl` as `dataset="fareez"`, `split="test"`.
- Two source files (`RES0002.txt`, `RES0054.txt`) are UTF-16LE, unlike the
  rest of the corpus (UTF-8) — loader now tries UTF-8 first and falls back
  to UTF-16 on decode failure rather than silently replacing bad bytes.
- Accepted, unfixed parsing artifact: ~2-3 lines have a garbled speaker
  marker (`DL OK, ...`, `D" And do you have...` instead of `D: ...`) and get
  merged into the previous turn's text rather than parsed as their own
  doctor turn. Rare enough (2-3 of ~26,600 turns) not to warrant a bespoke
  regex exception.
- Accepted, unfixed spelling-variant mismatch found during real-ASR pilot
  alignment (3 fareez encounters, MSK0006/49/07): gold `achey` vs. Whisper
  `achy` count as a substitution error under `normalize()`, same issue
  class as the `ok`/`okay` case that was fixed — left as-is for now.

## Correction (post-hoc): tokenization bug in Phase 3 alignment
- Found via explainability inspection: align_and_label.py's gold-side tokenizer treated standalone punctuation (periods, commas) as words, causing local misalignment near punctuation and corrupting some gold-word attributions
- Fixed identically to the real-ASR script (drop tokens with no alphanumeric characters) and re-ran the full downstream chain
- Corrected Tier 1/2/3 error rates: 12.6% / 67.4% / 90.2% (up from 12.9% / 41.9% / 54.7%) — effect size is now substantially larger
- Corrected bootstrap CI for risk-score-vs-uncertainty-only ablation at 10% flagging: [0.9, 6.0]pp — now excludes zero, confirming standard two-sided significance (previously bordered on zero)
- Deliverable 1 (classifier) numbers unaffected — those features derive only from Whisper's own output tokens, which were never affected by this bug

## 2026-09-23 — Three-way tier comparison: synthetic vs. real audio vs. production ASR

| Source | Tier 1 | Tier 2 | Tier 3 | Tier3/Tier1 |
|---|---|---|---|---|
| ACI-Bench + TTS (synthetic, Whisper-base) | 12.6% | 62.05% (updated, see below) | 90.2% | 7.2x |
| Fareez real audio (Whisper-base, n=60/60) | 9.55% | 16.03% | 54.88% | 5.7x |
| ACI-Bench real production ASR | 2.23% | 1.75% | 1.33% | 0.6x (inverted) |

- The domain-critical-term error effect reproduces on real audio at a comparable
  magnitude to the synthetic pipeline (5.7x vs. 7.2x), and does NOT reproduce
  on production ASR (0.6x, inverted) — suggesting the effect is a property of
  Whisper-base itself, not an artifact of TTS synthesis.
- Fareez figures are n=60/60 encounters, complete. Overall pooled WER 12.84%.
  Tier 1: 9.55% (90,334 tokens, 8,626 errors). Tier 2: 16.03% (131 tokens, 21
  errors). Tier 3: 54.88% (164 tokens, 90 errors). Tier 2/3 token counts are
  still small (131, 164) — no confidence interval computed yet; treat as
  directional pending a bootstrap.
- The 60 fareez encounters were NOT randomly sampled: selected by ranking
  eligible Tier-2/3-containing encounters by distinct glossary term count,
  top 60 taken (see `scratch/select_fareez_60.py`).
- UPDATE (2026-09-23, later same day): the ACI-Bench+TTS Tier 2 cell above was
  originally 67.4%, computed before the `ok`/`okay` `normalize()` fix.
  `data/processed/word_level_labels.csv` has now been regenerated (207
  encounters, fareez excluded) with the patched normalize() and the table
  updated in place. Result: Tier 1: 12.54% (−0.09pp, unaffected within
  rounding), Tier 2: **62.05% (−5.30pp)**, Tier 3: 90.17% (byte-identical,
  946/946 tokens) — only the Tier 2 cell and its row are affected; Tier 1,
  Tier 3, and the 7.2x Tier3/Tier1 ratio (90.2/12.6, unchanged by a Tier-2
  move) are confirmed unaffected.
- Why Tier 2's token count moved by 42 even though `ok`/`okay` aren't
  glossary terms: this is a sequence-alignment boundary-shift effect, not a
  reclassification of specific words. `normalize()` feeds into the strings
  jiwer aligns (`ref_text`/`hyp_text`), so fixing the `ok`/`okay` mismatch
  changes where jiwer places match/substitution/deletion boundaries nearby;
  that shifts which gold-word *indices* land in which alignment chunk,
  which changes which gold words get counted as Tier 1 vs. Tier 2 — the
  glossary lookup itself (on `gold_norm`) is unchanged, only the alignment
  feeding it moved.

## 2026-09-23 — Ablation: uncertainty classifier without the glossary-tier feature

Same 80/20 ACI-Bench split (`GroupShuffleSplit`, `test_size=0.2`, `random_state=42`,
42 test encounters, 48,921 test rows). Critical-error recall (fraction of gold
Tier 2/3 errors caught) at each flagging budget, four rankings, with vs. without
the tier feature inside the classifier:

| Ranking | 5% | 10% | 20% |
|---|---|---|---|
| (a) raw 1−confidence (no model) | 14.52% | 47.30% | 72.61% |
| (b) learned uncertainty — with tier feature (original) | 35.68% | 54.36% | 73.86% |
| (b) learned uncertainty — tier feature removed | 32.78% | 51.04% | 71.78% |
| (c) tier weight alone (no learned uncertainty) | 17.84% | 21.16% | 28.22% |
| (d) risk score — with tier feature (original) | 39.83% | 57.68% | 75.93% |
| (d) risk score — tier feature removed | 36.93% | 55.60% | 74.27% |

- Bootstrap at 10% flagging, risk_score (tier feature removed) vs. raw
  1-confidence, same cluster-resampling method as `bootstrap_significance.py`
  (2000 iterations, encounter-level resampling, seed 42): mean difference
  +8.02pp, 95% CI [2.23pp, 14.11pp] (excludes zero), 99.9% of bootstrap
  samples favor risk_score.
- This is a different comparison from the existing +3.33pp/4.25pp figures
  elsewhere in this log, which compare risk_score against uncertainty-only
  (not against raw confidence) — both comparisons are valid and kept as
  separate baselines, not reconciled into one number.
- Removing the tier feature from the classifier costs ~2-3pp of recall at
  every budget (the learned model partially reconstructs the tier signal
  from confidence + word length alone, but not fully) — the explicit tier
  feature still helps, though most of the risk score's advantage over raw
  confidence survives even without it, since tier weighting is still
  applied downstream in step (d) regardless of what the classifier sees.
- New artifact: `models/uncertainty_xgb_no_tier.joblib` (tier feature
  removed). The original `models/uncertainty_xgb.joblib` and its stored
  results are unchanged — this ablation loaded it read-only for comparison
  and did not retrain or overwrite it.
- UPDATE (2026-09-23, later same day): this ablation was computed against
  the stale pre-`normalize()`-fix `word_level_labels.csv`. Rerun against
  the regenerated file: (a)/(b-old)/(c)/(d-old) are byte-identical (they
  depend only on `confidence`/`tier_whisper`/`word_len`, unaffected by
  realignment, and the pre-existing model). (b-new) moved to 33.61% /
  51.45% / 71.78% (5/10/20%, was 32.78% / 51.04% / 71.78%); (d-new) moved
  to 39.83% / 55.19% / 73.86% (was 36.93% / 55.60% / 74.27%) — the
  retrained no-tier model shifted slightly because its training labels
  came from the corrected alignment. Bootstrap (d-new vs. a) at 10%:
  +8.17pp, 95% CI [2.39pp, 13.94pp] (was +8.02pp, [2.23pp, 14.11pp]) —
  still excludes zero, CI tightened slightly, no change in conclusion.

## 2026-09-23 — Bootstrap 95% CI on Fareez per-tier error rates (n=60)

Encounter-level cluster resampling, 2000 resamples, seed 42 (same method as
`bootstrap_significance.py`), applied to today's n=60 fareez tier error rates:

| Tier | Point estimate | 95% CI | Resamples |
|---|---|---|---|
| 1 | 9.55% | [8.87%, 10.26%] | 2000 |
| 2 | 16.03% | [8.26%, 25.18%] | 2000 |
| 3 | 54.88% | [47.53%, 63.13%] | 2000 |

- Tier 3's CI lower bound (47.53%) exceeds Tier 1's CI upper bound (10.26%),
  so the tier effect itself is robust to sampling uncertainty even though the
  exact Tier 3 point estimate is loosely pinned by n=164 tokens.

## 2026-09-23 — Tier 3 weight sensitivity sweep (2.0-8.0)

Swept the Tier 3 risk-weight parameter (2.0, 3.0, 4.25, 5.5, 7.0, 8.0), holding
Tier 1=1.0 and Tier 2=3.25 fixed, using the tier-removed classifier
(`uncertainty_xgb_no_tier.joblib`, read-only) on the same ACI-Bench 80/20 split.

- Recall @10% budget is flat at 55.60% from weight 3.0 through 7.0, with only
  minor movement at the extremes (2.0: 54.77%; 7.0-8.0 move only at 5%/20%,
  not at 10%).
- All 6 bootstrap CIs vs. raw confidence (2000 resamples, encounter-level
  clustering, seed 42) exclude zero: mean differences range +7.32pp to
  +8.37pp, tightest 95% CI [1.78, 13.31]pp.
- The current weight (4.25) sits in the flat plateau, not tuned to an edge
  case — the result is robust to this hyperparameter choice.
- UPDATE (2026-09-23, later same day): rerun against the regenerated
  `word_level_labels.csv` (patched `normalize()`) and the freshly-retrained
  `uncertainty_xgb_no_tier.joblib`. Recall @10% across the sweep: 2.0→54.36%,
  3.0→54.77%, 4.25→55.19%, 5.5→55.19%, 7.0→55.60%, 8.0→55.60% (was 54.77% /
  55.60% / 55.60% / 55.60% / 55.60% / 55.60%) — the plateau shape is
  preserved, just shifted down by ~0.4pp at the middle of the range. All 6
  bootstrap CIs still exclude zero: mean differences now range +7.46pp to
  +8.54pp, tightest 95% CI [1.90, 13.08]pp (was +7.32pp to +8.37pp,
  [1.78, 13.31]pp) — no change in conclusion.

## 2026-09-23 — 5-fold cross-validation replaces single 80/20 split (tier-free classifier)

`StratifiedGroupKFold`, 5 splits, seed 42, encounter-grouped (no encounter spans
train/test within a fold). Each fold retrains a temporary tier-free classifier
(same architecture as `uncertainty_xgb_no_tier.joblib`); nothing saved to
`models/` — all 5 fold models discarded after use.

| Metric | Mean | Std |
|---|---|---|
| Critical recall, risk_score @5% | 42.92% | 1.11% |
| Critical recall, risk_score @10% | 58.29% | 2.70% |
| Critical recall, risk_score @20% | 77.25% | 2.90% |
| Critical recall, raw_confidence @5% | 20.81% | 2.15% |
| Critical recall, raw_confidence @10% | 47.27% | 2.41% |
| Critical recall, raw_confidence @20% | 75.99% | 3.20% |
| ROC-AUC | 0.8534 | 0.0024 |
| Gap @10% (risk_score − raw_confidence) | +11.02pp | ±2.84pp |

- Per-fold gap @10%: 13.93pp, 13.99pp, 8.04pp, 8.64pp, 10.50pp — all positive,
  range 8.04-13.99pp.
- 5-fold mean gap (+11.02pp) is higher than the single 80/20 split's result
  (+8.02pp from the earlier tier-feature ablation) — the single split wasn't
  an outlier, it just landed toward the low end of the fold distribution.
- ROC-AUC is very stable across folds (std 0.0024) — the extra variance in
  the recall gap comes from the small Tier 2/3 critical-error denominators,
  not from classifier instability.
- UPDATE (2026-09-23, later same day): rerun against the regenerated
  `word_level_labels.csv` (patched `normalize()`), same seed 42. New means:
  risk_score recall 43.24%/59.09%/77.52% (5/10/20%, was 42.92%/58.29%/77.25%),
  raw_confidence recall 20.74%/46.93%/76.07% (was 20.81%/47.27%/75.99%),
  ROC-AUC 0.8525 ± 0.0040 (was 0.8534 ± 0.0024). **Gap @10%: +12.16pp ±
  1.45pp (was +11.02pp ± 2.84pp)** — per-fold gaps now 10.81-14.35pp (was
  8.04-13.99pp), all still positive. The gap got larger and the std got
  tighter, not weaker — no change in conclusion, if anything a stronger
  result. Note: `StratifiedGroupKFold` fold membership shifted slightly
  between runs even at the same seed, because it stratifies on `is_error`,
  which changed under the corrected alignment — expected, not a bug.

## 2026-09-23 — Precision at flagging budgets (alongside existing recall)

Precision @ budget = critical errors caught among flagged words / total words
flagged at that budget. Recall as before (critical errors caught / total
critical errors). "Critical" = gold Tier 2/3. Same `uncertainty_xgb_no_tier.joblib`
(read-only) used for both datasets so risk_score is directly comparable.
ACI-Bench = 80/20 split test set (48,921 rows); Fareez = all n=60 encounters
(88,989 scorable rows — deletions excluded, no output word to attach a score to,
same caveat as the original classifier).

**ACI-Bench:**

| Ranking | Budget | Precision | Recall |
|---|---|---|---|
| raw_confidence | 5% | 1.43% | 14.52% |
| raw_confidence | 10% | 2.33% | 47.30% |
| raw_confidence | 20% | 1.79% | 72.61% |
| risk_score | 5% | 3.92% | 39.83% |
| risk_score | 10% | 2.72% | 55.19% |
| risk_score | 20% | 1.82% | 73.86% |
| random (chance) | 5% | 0.49% | 5.00% |
| random (chance) | 10% | 0.49% | 10.00% |
| random (chance) | 20% | 0.49% | 20.00% |

**Fareez:**

| Ranking | Budget | Precision | Recall |
|---|---|---|---|
| raw_confidence | 5% | 0.67% | 29.41% |
| raw_confidence | 10% | 0.66% | 57.84% |
| raw_confidence | 20% | 0.46% | 80.39% |
| risk_score | 5% | 1.33% | 57.84% |
| risk_score | 10% | 0.78% | 67.65% |
| risk_score | 20% | 0.48% | 84.31% |
| random (chance) | 5% | 0.11% | 5.00% |
| random (chance) | 10% | 0.11% | 10.00% |
| random (chance) | 20% | 0.11% | 20.00% |

Random baseline: precision under uniform random flagging equals the dataset's
overall critical-error rate (`n_critical_errors / n_total_rows`) and is
constant across budgets by construction — a random sample's expected hit rate
doesn't depend on sample size. Recall under random flagging equals the budget
fraction itself, also by construction (expected fraction of critical errors
caught = fraction of words flagged). ACI-Bench: 241/48,921 = 0.49%. Fareez:
102/88,989 = 0.11%.

- risk_score beats raw_confidence on precision at every budget on both
  datasets — most dramatically at 5% (ACI-Bench 3.92% vs. 1.43%, ~2.7x;
  Fareez 1.33% vs. 0.67%, ~2x).
- Precision is low in absolute terms everywhere (under 4%) because critical
  (Tier 2/3) errors are a small fraction of all flagged words at these
  budgets — expected given Tier 2/3 tokens are themselves rare (241 critical
  errors in 48,921 ACI-Bench test rows; 102 in 88,989 Fareez rows). Recall
  is the more informative metric for this application, precision is reported
  for completeness.
- Against the random floor, both rankings clear it by a wide margin on
  precision: raw_confidence is ~3-5x chance, risk_score is ~4-8x chance
  (ACI-Bench 5%: 3.92% vs. 0.49% floor, ~8x; Fareez 5%: 1.33% vs. 0.11%
  floor, ~12x). The absolute precision numbers look small in isolation, but
  relative to how rare critical errors actually are in each dataset, both
  rankings — and risk_score in particular — are doing substantial work above
  chance.
- Fareez precision is consistently lower than ACI-Bench precision at the same
  budget (e.g. 10%: 0.78% vs. 2.72%) — consistent with Fareez's much lower
  overall critical-error density relative to its total token count.