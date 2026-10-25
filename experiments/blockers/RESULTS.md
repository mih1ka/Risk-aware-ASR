# Blocker re-runs R1–R5 (2026-10-25)

## Which glossary each table uses

The 176-term glossary is `data/glossary/tiered_glossary_176.csv`, which
leaves out the 14 terms taken from Fareez transcripts. The 190-term glossary
is `data/glossary/tiered_glossary.csv`.

| Table | Glossary |
|---|---|
| **R5 (a)–(d): primary tables** (`make_all_tables.py`, `r5_tables.md`) | **176 terms**, with 190 terms in the next column as a sensitivity check |
| R5 (e): ACI-Bench Tier 2, normalize() vs glossary | Both, 2×2 design |
| R5 (f): search for the paper's CV figures | Both |
| R0: baseline check | 190 (the published baseline) |
| R1: Fareez, 190 vs 176 | Both, side by side (same numbers as R5 (d)) |
| R2: production ASR | "Old" row 176 (as originally generated); "new" row 190 (what the script loads); decomposition rows labelled |
| R3: ACI-Bench CV denominators | 190 (what `cv_5fold_no_tier.py` loads) |

In every condition, one glossary sets both the gold tiers and RiskAware's
inference-time tier lookup.

All runs are on CPU with seed 42. Bootstrap CIs use 2000 resamples of whole
encounters. Ties in rankings are broken by original word order (stable sort)
unless a row says otherwise. Model: `models/uncertainty_xgb_no_tier.joblib`
(read-only) for Fareez. ACI-Bench uses the 5-fold CV models, retrained per
fold as in `scratch/cv_5fold_no_tier.py`.

"Critical" means gold tier 2 or 3. "Scorable" critical errors are those with
an ASR word. "All" critical errors also include deleted gold words, which no
ranking can flag. The hits are the same under both; only the denominator
changes.

## R0: baseline check (passed)

`r0_reproduce_baseline.py`, output in `r0_baseline_check.csv`. I re-aligned
all 272 Fareez files with the 190-term glossary, and the result matches
`data/processed/word_level_labels_fareez.csv` row for row.

| Check | Expected | Observed |
|---|---|---|
| T1 / T2 / T3 error rate | 10.24 / 24.33 / 55.04% | 10.24 / 24.33 / 55.04% |
| T1 / T2 / T3 gold words | 356,873 / 263 / 367 | 356,873 / 263 / 367 |
| Scorable critical errors | 245 | 245 (266 including deletions) |
| Raw confidence caught at 10% | 129 | 129 |
| RiskAware caught at 10% | 154 or 155 | 155 with the stable sort, 154 with numpy's default sort |

Two things matter for reproducing these:

- **Reading the CSV.** By default `pd.read_csv` turns the spoken word "None"
  into NaN. That drops 7 Tier 1 gold words (356,866 instead of 356,873) and
  6 ASR words. `common.read_labels` keeps "None" as a word. The budget hits
  are the same either way.
- **The 154 vs 155 hits.** This comes only from how ties are broken. Our log
  reports 154/245 = 62.86% (default sort). Yashvi's 58.27% is 155/266.

## R1: 176-term glossary (glossary without the 14 Fareez-derived terms)

`r1_make_glossary_176.py` writes `data/glossary/tiered_glossary_176.csv`:
176 rows, 40 Tier 2 and 136 Tier 3. It has the same rows as the glossary at
commit e30d20c, before Fareez terms were added. `run_align_fareez272_176.py`
is a copy of `scratch/run_align_fareez272.py` pointed at that file. It writes
`private/word_level_labels_fareez_176.csv`, which is git-ignored because it
holds transcript words. The original labels file is untouched. In each
condition, the same glossary sets the gold tiers and RiskAware's
inference-time tier lookup.

**Whisper-base tier error rates, Fareez n=272** (`r1_tier_rates.csv`)

| Tier | 190 terms: rate [95% CI] | words (errors) | 176 terms: rate [95% CI] | words (errors) |
|---|---|---|---|---|
| 1 | 10.24% [9.88, 10.59] | 356,873 (36,538) | 10.26% [9.91, 10.61] | 357,022 (36,624) |
| 2 | 24.33% [16.93, 31.96] | 263 (64) | 20.79% [13.66, 29.02] | 178 (37) |
| 3 | 55.04% [49.73, 60.33] | 367 (202) | 47.19% [41.18, 52.82] | 303 (143) |

**Critical-error recall** (`r1_recall.csv`). Each cell gives hits, then
recall over scorable errors / recall over all errors. Critical errors:
190 terms, 245 scorable / 266 all; 176 terms, 171 scorable / 180 all.

| Ranking | Budget | 190 terms | 176 terms |
|---|---|---|---|
| Raw confidence | 5% | 71: 28.98 / 26.69% | 49: 28.65 / 27.22% |
| Raw confidence | 10% | 129: 52.65 / 48.50% | 88: 51.46 / 48.89% |
| Raw confidence | 20% | 186: 75.92 / 69.92% | 131: 76.61 / 72.78% |
| Learned uncertainty | 5% | 100: 40.82 / 37.59% | 61: 35.67 / 33.89% |
| Learned uncertainty | 10% | 134: 54.69 / 50.38% | 91: 53.22 / 50.56% |
| Learned uncertainty | 20% | 183: 74.69 / 68.80% | 131: 76.61 / 72.78% |
| RiskAware | 5% | 129: 52.65 / 48.50% | 78: 45.61 / 43.33% |
| RiskAware | 10% | 155: 63.27 / 58.27% | 100: 58.48 / 55.56% |
| RiskAware | 20% | 196: 80.00 / 73.68% | 134: 78.36 / 74.44% |

- The tier ordering holds without the Fareez-derived terms. Tier 3 drops
  from 55.04% to 47.19%, and the Tier 2 and Tier 3 CIs still do not overlap
  Tier 1's.
- RiskAware still leads raw confidence at every budget, but by less. At 10%
  with all errors counted, the lead is +6.67 pp with 176 terms against
  +9.77 pp with 190. Over scorable errors it is +7.02 pp against +10.62 pp.
  Part of RiskAware's Fareez advantage therefore depends on terms taken from
  Fareez itself.
- I did not compute CIs for these recall differences; you asked for CIs
  only on the tier rates.

## R2: production-ASR re-run with the shared normalize()

I edited `src/eval/validate_real_asr.py` so that it imports
`normalize`/`tokenize` from `align_and_label.py`. I made no other change to
it. The script was then re-run, which overwrote
`data/processed/real_asr_word_labels.csv`. Its console output is in
`r2_validate_real_asr_stdout.txt`. `r2_decompose.py` separates the two
changes in memory (`r2_tier_rates.csv`). The script reads
`tiered_glossary.csv`, which grew from 176 to 190 terms after the stored file
was made, so a plain re-run also picks up the glossary change.

Each cell shows T1 / T2 / T3 error rate, then gold words. All conditions
cover 105 encounters, 131,144 rows and 130,639 gold words.

| Condition | T1 / T2 / T3 | Gold words T1 / T2 / T3 |
|---|---|---|
| Old: stored file (old normalize, 176 terms) | 2.23 / 1.75 / 1.33% | 129,883 / 228 / 528 |
| New normalize, 176 terms | 2.23 / 1.75 / 1.33% | 129,883 / 228 / 528 |
| Old normalize, 190 terms | 2.23 / 1.95 / 1.33% | 129,854 / 257 / 528 |
| **New: edited script as run (new normalize, 190 terms)** | **2.23 / 1.95 / 1.33%** | 129,854 / 257 / 528 |

- **The normalize fix changes nothing here.** "ok" never appears in the ASR
  or the human text; only "okay" does (4,386 tokens). Rows, errors and
  alignment are identical.
- **The only change is Tier 2, from 1.75% to 1.95% (4/228 to 5/257).** It
  comes from the 14 glossary terms. Tier 3 / Tier 1 is still 0.6x
  (1.33 / 2.23).
- **The correction-module output differs from the log.** The re-run gives
  11 critical substitutions: 4 near-miss (top-1 100%) and 7 dropout (top-1
  1/7 = 14.3%). The log recorded 4 near-miss and 6 dropout (0%). The
  normalize-only comparison above is identical, so the difference is due to
  the glossary.
- **Figures were not regenerated.** `make_figures.py` does not read
  `real_asr_word_labels.csv`; fig5 hardcodes 2.2 / 1.8 / 1.3. Under the
  190-term glossary, Tier 2 would be 1.9 rather than 1.8.

## R3: ACI-Bench 5-fold recall denominator

`r3_cv_denominators.py` uses the same folds, model and glossary as
`scratch/cv_5fold_no_tier.py`. Per-fold rows are in `r3_per_fold.csv` and
means in `r3_summary.csv`.

Critical errors per fold, scorable / all (deleted):
230/238 (8), 208/209 (1), 248/256 (8), 220/220 (0), 222/226 (4).
In total: 1,128 scorable and 1,149 all, so 21 are deletions.

Mean ± sd over 5 folds on the current labels. The script's own sort and the
stable sort give identical results. ROC-AUC is 0.8525.

| Ranking | Budget | Scorable denominator | All denominator |
|---|---|---|---|
| RiskAware | 5% | 43.24 ± 1.76% | 42.48 ± 1.76% |
| RiskAware | 10% | 59.09 ± 2.74% | 58.06 ± 2.98% |
| RiskAware | 20% | 77.52 ± 4.11% | 76.14 ± 3.52% |
| Raw confidence | 5% | 20.74 ± 2.99% | 20.37 ± 2.94% |
| Raw confidence | 10% | 46.93 ± 2.93% | 46.11 ± 3.11% |
| Raw confidence | 20% | 76.07 ± 4.02% | 74.73 ± 3.85% |
| Gap at 10% | | +12.16 pp | +11.95 pp |

**Which denominator the reported numbers used.** `cv_5fold_no_tier.py` drops
rows without an ASR word before scoring. Every ACI-Bench recall in the
decisions log therefore uses the **scorable** denominator.

**Where the paper's 59.31% / 47.27% / 0.8535 / +12.04 pp come from.** I could
not trace them.

- 47.27% is the raw-confidence recall on the **pre-normalize-fix** labels
  (git 9d19693; `r3_prefix_labels_summary.csv`, scorable denominator). That
  same run gives RiskAware 58.29%, not 59.31%, and ROC-AUC 0.8534.
- No run here produces 59.31% or 0.8535. That includes current or pre-fix
  labels, either sort, either CSV reading, and either denominator.
- I am not guessing their source. The figures this repo supports are
  59.09% vs 46.93% (+12.16 pp) on the scorable denominator, or 58.06% vs
  46.11% (+11.95 pp) counting all critical errors.

## Files

| File | Contents |
|---|---|
| `common.py` | Shared helpers: CSV reader, bootstrap, stable-sort budget recall |
| `r0_reproduce_baseline.py`, `r0_baseline_check.csv` | Baseline check |
| `r1_make_glossary_176.py`, `run_align_fareez272_176.py`, `r1_compare_176_vs_190.py`, `r1_tier_rates.csv`, `r1_recall.csv` | R1 |
| `r2_decompose.py`, `r2_tier_rates.csv`, `r2_validate_real_asr_stdout.txt` | R2 |
| `r3_cv_denominators.py`, `r3_per_fold.csv`, `r3_summary.csv`, `r3_prefix_labels_summary.csv` | R3 |
| `make_all_tables.py`, `r5_tables.md`, `r5_*.csv` | R5: all tables, 176-term primary, 190-term sensitivity |
| `private/` (git-ignored) | Fareez word-level labels for the 176-term glossary |

The R3 pre-fix check needs the old labels extracted first:
`git show 9d19693:data/processed/word_level_labels.csv > /tmp/prefix.csv`,
then `python r3_cv_denominators.py /tmp/prefix.csv`.

## R5: all tables regenerated, 176-term glossary primary

`make_all_tables.py` rebuilds everything below from the repo in a single run
(36 s on CPU). It prints the git commit and date at the top: `2940b8d`,
2026-10-24 23:32 +0530, plus this branch's uncommitted files. It writes
`r5_*.csv` and `r5_tables.md`. I ran it from scratch twice (clean
`__pycache__`, outputs deleted first) and got identical numbers both times.

How the inputs are handled:

- **ACI-Bench synthetic speech.** The repo has no ACI-Bench Whisper outputs;
  `data/whisper_out/` holds only the 272 Fareez files. So (a), (c) and (e)
  start from the stored alignment, `data/processed/word_level_labels.csv`,
  and re-derive each gold word's tier from the glossary. The glossary only
  sets tiers, not the alignment, so this is the same as re-aligning. The
  script checks it: re-tiering with 190 terms reproduces the stored tier
  column exactly, and re-tiering the pre-fix file (e30d20c) with 176 terms
  reproduces that file's tier column exactly.
- **Production ASR and Fareez.** (b) and (d) are re-aligned from source in
  memory.
- **Reading and sorting.** "None" is kept as a word, and ties are broken by
  word order. That is why the 190-term CV numbers here (59.03% / 47.07%)
  differ slightly from R3's 59.09% / 46.93%, which reproduce the CV
  script's own pandas reading.

**Item 5: the ACI-Bench Tier 2 change from 67.4% to 62.05% came from the
glossary, not normalize().** See table (e). Under 176 terms, Tier 2 is
67.36% (293/435) with both old and new normalize(). Under 190 terms it is
62.05% (296/477) with both. normalize() does change the ACI-Bench alignment
(37,032 errors become 36,827), but no Tier 2 word is affected. The
2026-09-23 log entry that credits this change to normalize() and alignment
boundary shifts is wrong. The labels were regenerated with the fixed
normalize() and the new 190-term glossary at the same time, and the whole
change comes from the glossary.

**Item 6: I could not find a script version that produces 59.31% / 0.8535.**
- `scratch/cv_5fold_no_tier.py` has one version in git (9d19693).
  `word_level_labels.csv` has two (e30d20c, before the normalize fix;
  a234584, after it).
- Searching every commit on every branch, including upstream PRs #3–#5, for
  "59.31", "0.5931", "0.8535" and "12.04 pp" (CSVs excluded) finds them only
  in the README lines added on 2026-10-24. Yashvi's notebook and its results
  CSVs contain no ACI-Bench CV.
- Table (f) runs the CV script's exact settings on both label versions ×
  both glossaries × both CSV readings. None matches.
- The closest is pre-fix labels, 176 terms, pandas reading: 58.47% / 47.27%
  / 0.8534. So 47.27% and roughly 0.853 trace to the pre-fix labels, but
  59.31% does not. Different library versions might explain it, but I
  could not test that.

### (a) ACI-Bench synthetic speech, Whisper-base, 207 encounters

| tier | error_rate 176 | error_rate 190 | gold_words 176 | gold_words 190 | errors 176 | errors 190 |
|---|---|---|---|---|---|---|
| 1 | 12.54% | 12.54% | 234,893 | 234,851 | 29,449 | 29,446 |
| 2 | 67.36% | 62.05% | 435 | 477 | 293 | 296 |
| 3 | 90.17% | 90.17% | 946 | 946 | 853 | 853 |

### (b) ACI-Bench production-ASR pairs, 105 encounters (131,144 rows)

| tier | error_rate 176 | error_rate 190 | gold_words 176 | gold_words 190 | errors 176 | errors 190 |
|---|---|---|---|---|---|---|
| 1 | 2.23% | 2.23% | 129,883 | 129,854 | 2,892 | 2,891 |
| 2 | 1.75% | 1.95% | 228 | 257 | 4 | 5 |
| 3 | 1.33% | 1.33% | 528 | 528 | 7 | 7 |

### (c) ACI-Bench 5-fold CV recall (mean ± sd over folds)

| ranking | budget_pct | recall_all 176 | recall_all 190 | recall_scorable 176 | recall_scorable 190 |
|---|---|---|---|---|---|
| raw_confidence | 5 | 20.24 ± 3.01% | 20.26 ± 2.93% | 20.60 ± 3.04% | 20.63 ± 2.97% |
| raw_confidence | 10 | 46.13 ± 2.61% | 46.26 ± 2.74% | 46.93 ± 2.17% | 47.07 ± 2.31% |
| raw_confidence | 20 | 74.65 ± 3.81% | 74.71 ± 3.88% | 75.99 ± 4.00% | 76.04 ± 4.06% |
| riskaware | 5 | 44.17 ± 2.70% | 44.15 ± 2.84% | 44.96 ± 2.70% | 44.93 ± 2.78% |
| riskaware | 10 | 57.91 ± 2.78% | 58.01 ± 2.90% | 58.92 ± 2.28% | 59.03 ± 2.41% |
| riskaware | 20 | 75.98 ± 3.40% | 76.04 ± 3.47% | 77.35 ± 3.89% | 77.40 ± 3.93% |

### (c) ROC-AUC and RiskAware − raw confidence gain at 10%

| glossary | roc_auc | gain_10pct_all | gain_10pct_scorable | crit_errors_all | crit_errors_scorable |
|---|---|---|---|---|---|
| 176 | 0.8525 ± 0.0040 | +11.78 ± 0.80 pp | +12.00 ± 0.84 pp | 1146 | 1125 |
| 190 | 0.8525 ± 0.0040 | +11.75 ± 0.77 pp | +11.96 ± 0.81 pp | 1149 | 1128 |

### (c) per fold

| glossary | fold | crit_errors_all | crit_errors_scorable | roc_auc | gain_10pct_all_pp | gain_10pct_scorable_pp |
|---|---|---|---|---|---|---|
| 176 | 1 | 237 | 229 | 0.8531 | 12.66 | 13.1 |
| 176 | 2 | 204 | 203 | 0.8548 | 11.76 | 11.82 |
| 176 | 3 | 261 | 253 | 0.8478 | 11.49 | 11.86 |
| 176 | 4 | 218 | 218 | 0.8576 | 12.39 | 12.39 |
| 176 | 5 | 226 | 222 | 0.8491 | 10.62 | 10.81 |
| 190 | 1 | 238 | 230 | 0.8531 | 12.61 | 13.04 |
| 190 | 2 | 204 | 203 | 0.8548 | 11.76 | 11.82 |
| 190 | 3 | 261 | 253 | 0.8478 | 11.49 | 11.86 |
| 190 | 4 | 220 | 220 | 0.8576 | 12.27 | 12.27 |
| 190 | 5 | 226 | 222 | 0.8491 | 10.62 | 10.81 |

### (d) Fareez, Whisper-base, 272 encounters: tier error rates

| tier | rate [95% CI] 176 | rate [95% CI] 190 | gold_words 176 | gold_words 190 | errors 176 | errors 190 |
|---|---|---|---|---|---|---|
| 1 | 10.26% [9.91, 10.61] | 10.24% [9.88, 10.59] | 357022 | 356873 | 36624 | 36538 |
| 2 | 20.79% [13.66, 29.02] | 24.33% [16.93, 31.96] | 178 | 263 | 37 | 64 |
| 3 | 47.19% [41.18, 52.82] | 55.04% [49.73, 60.33] | 303 | 367 | 143 | 202 |

### (d) Fareez critical-error recall

Critical errors, all / scorable: 176 terms 180 / 171; 190 terms 266 / 245

| ranking | budget_pct | 176 | 190 |
|---|---|---|---|
| learned_uncertainty | 5 | 61; 33.89 / 35.67% | 100; 37.59 / 40.82% |
| learned_uncertainty | 10 | 91; 50.56 / 53.22% | 134; 50.38 / 54.69% |
| learned_uncertainty | 20 | 131; 72.78 / 76.61% | 183; 68.80 / 74.69% |
| raw_confidence | 5 | 49; 27.22 / 28.65% | 71; 26.69 / 28.98% |
| raw_confidence | 10 | 88; 48.89 / 51.46% | 129; 48.50 / 52.65% |
| raw_confidence | 20 | 131; 72.78 / 76.61% | 186; 69.92 / 75.92% |
| riskaware | 5 | 78; 43.33 / 45.61% | 129; 48.50 / 52.65% |
| riskaware | 10 | 100; 55.56 / 58.48% | 155; 58.27 / 63.27% |
| riskaware | 20 | 134; 74.44 / 78.36% | 196; 73.68 / 80.00% |

### (e) ACI-Bench Tier 2: normalize() fix vs glossary

| normalize | glossary | tier2_rate | tier2_gold_words | tier2_errors |
|---|---|---|---|---|
| old (pre-fix, git e30d20c) | 176 | 67.36% | 435 | 293 |
| old (pre-fix, git e30d20c) | 190 | 62.05% | 477 | 296 |
| new (current) | 176 | 67.36% | 435 | 293 |
| new (current) | 190 | 62.05% | 477 | 296 |

### (f) scratch/cv_5fold_no_tier.py settings on every labels/glossary version (paper: 59.31 / 47.27 / 0.8535)

| labels | glossary | csv_read | riskaware_10_scorable | raw_10_scorable | roc_auc | matches_paper |
|---|---|---|---|---|---|---|
| old (pre-fix, git e30d20c) | 176 | pandas default | 58.47 | 47.27 | 0.8534 | False |
| old (pre-fix, git e30d20c) | 176 | keep None | 58.76 | 47.78 | 0.8534 | False |
| old (pre-fix, git e30d20c) | 190 | pandas default | 58.4 | 47.41 | 0.8534 | False |
| old (pre-fix, git e30d20c) | 190 | keep None | 58.59 | 47.91 | 0.8534 | False |
| new (current) | 176 | pandas default | 59.07 | 46.79 | 0.8525 | False |
| new (current) | 176 | keep None | 59.01 | 46.93 | 0.8525 | False |
| new (current) | 190 | pandas default | 59.09 | 46.93 | 0.8525 | False |
| new (current) | 190 | keep None | 59.03 | 47.07 | 0.8525 | False |
