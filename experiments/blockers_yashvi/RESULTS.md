# Yashvi blockers — 176-term primary evaluation

**Public commit scope:** This folder contains only `.py` analysis code, aggregate `.csv` files and `RESULTS.md`; all private per-word files and both rating sheets stay outside the repository. This is a handoff artifact, not a completed GitHub push or pull request. There are no paper edits.

**PRIMARY TABLE:** `PRIMARY_176_term_comparison.csv` holds all ten methods, each at 5%, 10% and 20%, with recall over 180 total / 171 scorable critical errors, precision, flags per critical, and requested paired bootstrap intervals for A3/D/B vs RiskAware and C1 vs disagreement + confidence. Intervals are percentage points; each is obtained with seed 42, 2,000 paired whole-encounter bootstrap resamples. A1/A2 stop early at 1,175 words.

**SENSITIVITY — SEPARATE FILE:** `SENSITIVITY_ONLY_190_term.csv` holds the 190-term historical evaluation (266 critical errors, 245 scorable), reconstructed from the historical scored-word CSV. It must **not** be pooled, mixed, or directly substituted into the 176-term primary table.

**Clinical-tier qualification:** For Whisper-small at 176 terms, Tier 2 is **5.62%** versus Tier 1 **9.24%**; only Tier 3 (drug/dosage, **22.44%**) remains elevated over Tier 1. The reductions from Whisper-base do not establish that every clinical tier remains elevated after upgrading the model.

**Sharing:** Share only blinded rating sheets privately with Rashi. Do not share the answer key with raters until submissions are received. Do not commit anything from the private sheet bundle.

---

# Fareez 176-term full re-run — Yashvi handoff

**Status:** Rashi R5 cross-check matched exactly (180 critical errors: 171 scorable, 9 deleted; at 10%, raw confidence 88 and RiskAware 100). The tier-free uncertainty scores were reused; glossary tiers, critical labels, tier weights, RiskAware ranking, and all requested alternative rankings were rebuilt. Total scored words: 352,473; 272 encounters. Seed 42; paired bootstrap 2,000 whole-encounter resamples.

**Disagreement:** 28,896 / 352,473 scored Whisper-base positions = 8.1981%. This excludes small-only inserted tokens.

**Completed paired model analysis:** Whisper-small word-level labels were subsequently provided privately. The 176-term Whisper-base and Whisper-small error rates and 95% paired, encounter-bootstrap confidence intervals are now in `y176_whisper_base_small_tier_paired.csv`. All 272 encounter IDs and all per-tier reference-word counts agree. The private per-word files are not included in this handoff.

## Full table: counts, both recall denominators, precision, flags/caught, paired 95% CI vs RiskAware

All recalls and precision are percentages. The CI represents **method minus RiskAware-ASR** in percentage points. A1/A2 each flag only 1,175 output words across all budgets and do not fill the nominal budget.

| method                                   |   budget_pct |   flagged_words |   critical_caught |   recall_all_pct |   recall_scorable_pct |   precision_pct |   flags_per_critical |   observed_gap_vs_riskaware_pp |   ci_lower_pp |   ci_upper_pp |
|:-----------------------------------------|-------------:|----------------:|------------------:|-----------------:|----------------------:|----------------:|---------------------:|-------------------------------:|--------------:|--------------:|
| Raw confidence                           |            5 |           17623 |                49 |           27.222 |                28.655 |           0.278 |              359.653 |                        -16.111 |       -23.916 |        -8.791 |
| Raw confidence                           |           10 |           35247 |                88 |           48.889 |                51.462 |           0.25  |              400.534 |                         -6.667 |       -12.638 |         0     |
| Raw confidence                           |           20 |           70494 |               131 |           72.778 |                76.608 |           0.186 |              538.122 |                         -1.667 |        -4.918 |         1.07  |
| Learned uncertainty                      |            5 |           17623 |                61 |           33.889 |                35.673 |           0.346 |              288.902 |                         -9.444 |       -14.975 |        -4.854 |
| Learned uncertainty                      |           10 |           35247 |                91 |           50.556 |                53.216 |           0.258 |              387.33  |                         -5     |        -8.889 |        -0.546 |
| Learned uncertainty                      |           20 |           70494 |               131 |           72.778 |                76.608 |           0.186 |              538.122 |                         -1.667 |        -3.637 |         0     |
| RiskAware-ASR                            |            5 |           17623 |                78 |           43.333 |                45.614 |           0.443 |              225.936 |                          0     |         0     |         0     |
| RiskAware-ASR                            |           10 |           35247 |               100 |           55.556 |                58.48  |           0.284 |              352.47  |                          0     |         0     |         0     |
| RiskAware-ASR                            |           20 |           70494 |               134 |           74.444 |                78.363 |           0.19  |              526.075 |                          0     |         0     |         0     |
| Disagreement + confidence                |            5 |           17623 |               119 |           66.111 |                69.591 |           0.675 |              148.092 |                         22.778 |        13.559 |        30.953 |
| Disagreement + confidence                |           10 |           35247 |               161 |           89.444 |                94.152 |           0.457 |              218.925 |                         33.889 |        25.465 |        42.308 |
| Disagreement + confidence                |           20 |           70494 |               167 |           92.778 |                97.661 |           0.237 |              422.12  |                         18.333 |        12.941 |        24.023 |
| A1 Glossary-only strict                  |            5 |            1175 |                36 |           20     |                21.053 |           3.064 |               32.639 |                        -23.333 |       -30.435 |       -16.667 |
| A1 Glossary-only strict                  |           10 |            1175 |                36 |           20     |                21.053 |           3.064 |               32.639 |                        -35.556 |       -43.478 |       -27.976 |
| A1 Glossary-only strict                  |           20 |            1175 |                36 |           20     |                21.053 |           3.064 |               32.639 |                        -54.444 |       -62.094 |       -46.427 |
| A2 Glossary-restricted confidence strict |            5 |            1175 |                36 |           20     |                21.053 |           3.064 |               32.639 |                        -23.333 |       -30.435 |       -16.667 |
| A2 Glossary-restricted confidence strict |           10 |            1175 |                36 |           20     |                21.053 |           3.064 |               32.639 |                        -35.556 |       -43.478 |       -27.976 |
| A2 Glossary-restricted confidence strict |           20 |            1175 |                36 |           20     |                21.053 |           3.064 |               32.639 |                        -54.444 |       -62.094 |       -46.427 |
| A3 Glossary first then confidence filled |            5 |           17623 |                76 |           42.222 |                44.444 |           0.431 |              231.882 |                         -1.111 |        -6.945 |         4.488 |
| A3 Glossary first then confidence filled |           10 |           35247 |               104 |           57.778 |                60.819 |           0.295 |              338.913 |                          2.222 |        -2.891 |         6.989 |
| A3 Glossary first then confidence filled |           20 |           70494 |               136 |           75.556 |                79.532 |           0.193 |              518.338 |                          1.111 |        -1.841 |         4.186 |
| B (1-confidence) x tier                  |            5 |           17623 |                71 |           39.444 |                41.52  |           0.403 |              248.211 |                         -3.889 |        -9.548 |         1.765 |
| B (1-confidence) x tier                  |           10 |           35247 |                99 |           55     |                57.895 |           0.281 |              356.03  |                         -0.556 |        -5.715 |         4.652 |
| B (1-confidence) x tier                  |           20 |           70494 |               134 |           74.444 |                78.363 |           0.19  |              526.075 |                          0     |        -2.762 |         2.14  |
| D Tier gate then learned uncertainty     |            5 |           17623 |                81 |           45     |                47.368 |           0.46  |              217.568 |                          1.667 |         0     |         3.571 |
| D Tier gate then learned uncertainty     |           10 |           35247 |               102 |           56.667 |                59.649 |           0.289 |              345.559 |                          1.111 |        -1.571 |         3.297 |
| D Tier gate then learned uncertainty     |           20 |           70494 |               137 |           76.111 |                80.117 |           0.194 |              514.555 |                          1.667 |         0     |         3.615 |
| C1 Disagreement then RiskAware score     |            5 |           17623 |               122 |           67.778 |                71.345 |           0.692 |              144.451 |                         24.444 |        16.949 |        32.37  |
| C1 Disagreement then RiskAware score     |           10 |           35247 |               166 |           92.222 |                97.076 |           0.471 |              212.331 |                         36.667 |        29.104 |        44.199 |
| C1 Disagreement then RiskAware score     |           20 |           70494 |               167 |           92.778 |                97.661 |           0.237 |              422.12  |                         18.333 |        12.941 |        24.023 |

## Whisper-base vs Whisper-small 176-term tier analysis

| model        |   tier |   reference_words |   errors |   error_rate_pct |   ci_lower_pct |   ci_upper_pct |   bootstrap_samples |   seed |
|:-------------|-------:|------------------:|---------:|-----------------:|---------------:|---------------:|--------------------:|-------:|
| Whisper-base |      1 |            357022 |    36624 |           10.258 |          9.908 |         10.613 |                2000 |     42 |
| Whisper-base |      2 |               178 |       37 |           20.787 |         13.661 |         29.016 |                2000 |     42 |
| Whisper-base |      3 |               303 |      143 |           47.195 |         41.176 |         52.817 |                2000 |     42 |

## Review time sensitivity — 10 seconds per flagged word plus 30 seconds per encounter containing a flag

**Hypothetical review-time assumptions, not observed clinician timings.** For 5- and 20-second assumptions, see `y176_review_minutes_scenarios.csv`.

| method                                   |   budget_pct |   total_flags |   critical_caught |   mean_minutes_per_encounter |   median_minutes_per_encounter |   p90_minutes_per_encounter |   total_review_hours |
|:-----------------------------------------|-------------:|--------------:|------------------:|-----------------------------:|-------------------------------:|----------------------------:|---------------------:|
| Raw confidence                           |            5 |         17623 |                49 |                       11.298 |                         10.167 |                      17.15  |               51.219 |
| Raw confidence                           |           10 |         35247 |                88 |                       22.097 |                         20.833 |                      30.983 |              100.175 |
| Raw confidence                           |           20 |         70494 |               131 |                       43.695 |                         42.333 |                      58     |              198.083 |
| Learned uncertainty                      |            5 |         17623 |                61 |                       11.298 |                         10.833 |                      15.667 |               51.219 |
| Learned uncertainty                      |           10 |         35247 |                91 |                       22.097 |                         21     |                      30.5   |              100.175 |
| Learned uncertainty                      |           20 |         70494 |               131 |                       43.695 |                         42.167 |                      57.65  |              198.083 |
| RiskAware-ASR                            |            5 |         17623 |                78 |                       11.298 |                         10.667 |                      15.667 |               51.219 |
| RiskAware-ASR                            |           10 |         35247 |               100 |                       22.097 |                         21     |                      30.15  |              100.175 |
| RiskAware-ASR                            |           20 |         70494 |               134 |                       43.695 |                         42.333 |                      57.483 |              198.083 |
| Disagreement + confidence                |            5 |         17623 |               119 |                       11.298 |                         10.167 |                      18.467 |               51.219 |
| Disagreement + confidence                |           10 |         35247 |               161 |                       22.097 |                         19.667 |                      35.933 |              100.175 |
| Disagreement + confidence                |           20 |         70494 |               167 |                       43.695 |                         41.583 |                      60.117 |              198.083 |
| A1 Glossary-only strict                  |            5 |          1175 |                36 |                        1.209 |                          1.167 |                       1.833 |                5.481 |
| A1 Glossary-only strict                  |           10 |          1175 |                36 |                        1.209 |                          1.167 |                       1.833 |                5.481 |
| A1 Glossary-only strict                  |           20 |          1175 |                36 |                        1.209 |                          1.167 |                       1.833 |                5.481 |
| A2 Glossary-restricted confidence strict |            5 |          1175 |                36 |                        1.209 |                          1.167 |                       1.833 |                5.481 |
| A2 Glossary-restricted confidence strict |           10 |          1175 |                36 |                        1.209 |                          1.167 |                       1.833 |                5.481 |
| A2 Glossary-restricted confidence strict |           20 |          1175 |                36 |                        1.209 |                          1.167 |                       1.833 |                5.481 |
| A3 Glossary first then confidence filled |            5 |         17623 |                76 |                       11.298 |                         10.167 |                      17.467 |               51.219 |
| A3 Glossary first then confidence filled |           10 |         35247 |               104 |                       22.097 |                         20.917 |                      31     |              100.175 |
| A3 Glossary first then confidence filled |           20 |         70494 |               136 |                       43.695 |                         42.25  |                      58.317 |              198.083 |
| B (1-confidence) x tier                  |            5 |         17623 |                71 |                       11.298 |                         10.083 |                      17.15  |               51.219 |
| B (1-confidence) x tier                  |           10 |         35247 |                99 |                       22.097 |                         20.833 |                      31.3   |              100.175 |
| B (1-confidence) x tier                  |           20 |         70494 |               134 |                       43.695 |                         42.417 |                      58     |              198.083 |
| D Tier gate then learned uncertainty     |            5 |         17623 |                81 |                       11.298 |                         10.833 |                      15.833 |               51.219 |
| D Tier gate then learned uncertainty     |           10 |         35247 |               102 |                       22.097 |                         21     |                      30.317 |              100.175 |
| D Tier gate then learned uncertainty     |           20 |         70494 |               137 |                       43.695 |                         42.5   |                      57.817 |              198.083 |
| C1 Disagreement then RiskAware score     |            5 |         17623 |               122 |                       11.298 |                         10     |                      18.15  |               51.219 |
| C1 Disagreement then RiskAware score     |           10 |         35247 |               166 |                       22.097 |                         19.583 |                      35.65  |              100.175 |
| C1 Disagreement then RiskAware score     |           20 |         70494 |               167 |                       43.695 |                         41     |                      59.817 |              198.083 |

## Upstream Git requirements

Only public **aggregate** tables and reproducibility code belong in `experiments/blockers_yashvi/`. Keep private word-level scored data, audio, transcripts, and rating sheets off GitHub. No paper change is made. The connected GitHub integration is not authorized to create the branch or push.

---

## Detailed original independent-run methodology and results

# Full 176-term Metaphone Fareez re-run

Rebuilt ASR-word and reference tiers from upstream 190-term source glossary with 14 predefined terms removed using an exact verified CPU Metaphone port; tier-free learned uncertainty unchanged.

Original 190-term mapping independently matched 352,479/352,479 archived ASR tiers. The six literal "None" tokens were excluded to match the requested 352,473 legacy scope.

Rashi R5 10% gate **PASSED**: raw confidence 88, RiskAware 100; total=180, scorable=171, deleted=9; budget=35,247.

A1 flags all glossary output matches at every budget (non-comparable review cost); A2 stops early if glossary matches are fewer than the budget.

All scores, weights, ranked selections, and critical labels were recomputed for the 176-term glossary; the ASR model and learned uncertainty classifier were not retrained.

All uncertainty intervals below are paired whole-encounter bootstraps with seed 42, 2,000 resamples, fixed proportional review budget within each resample.

## Full method table

| method                                   |   budget_pct |   flagged_words |   critical_caught |   recall_all_pct |   recall_scorable_pct |   precision_pct |   flags_per_critical |
|:-----------------------------------------|-------------:|----------------:|------------------:|-----------------:|----------------------:|----------------:|---------------------:|
| Raw confidence                           |            5 |           17623 |                49 |           27.222 |                28.655 |           0.278 |              359.653 |
| Raw confidence                           |           10 |           35247 |                88 |           48.889 |                51.462 |           0.250 |              400.534 |
| Raw confidence                           |           20 |           70494 |               131 |           72.778 |                76.608 |           0.186 |              538.122 |
| Learned uncertainty                      |            5 |           17623 |                61 |           33.889 |                35.673 |           0.346 |              288.902 |
| Learned uncertainty                      |           10 |           35247 |                91 |           50.556 |                53.216 |           0.258 |              387.330 |
| Learned uncertainty                      |           20 |           70494 |               131 |           72.778 |                76.608 |           0.186 |              538.122 |
| RiskAware-ASR                            |            5 |           17623 |                78 |           43.333 |                45.614 |           0.443 |              225.936 |
| RiskAware-ASR                            |           10 |           35247 |               100 |           55.556 |                58.480 |           0.284 |              352.470 |
| RiskAware-ASR                            |           20 |           70494 |               134 |           74.444 |                78.363 |           0.190 |              526.075 |
| Disagreement + confidence                |            5 |           17623 |               119 |           66.111 |                69.591 |           0.675 |              148.092 |
| Disagreement + confidence                |           10 |           35247 |               161 |           89.444 |                94.152 |           0.457 |              218.925 |
| Disagreement + confidence                |           20 |           70494 |               167 |           92.778 |                97.661 |           0.237 |              422.120 |
| A1 Glossary-only strict                  |            5 |            1175 |                36 |           20.000 |                21.053 |           3.064 |               32.639 |
| A1 Glossary-only strict                  |           10 |            1175 |                36 |           20.000 |                21.053 |           3.064 |               32.639 |
| A1 Glossary-only strict                  |           20 |            1175 |                36 |           20.000 |                21.053 |           3.064 |               32.639 |
| A2 Glossary-restricted confidence strict |            5 |            1175 |                36 |           20.000 |                21.053 |           3.064 |               32.639 |
| A2 Glossary-restricted confidence strict |           10 |            1175 |                36 |           20.000 |                21.053 |           3.064 |               32.639 |
| A2 Glossary-restricted confidence strict |           20 |            1175 |                36 |           20.000 |                21.053 |           3.064 |               32.639 |
| A3 Glossary first then confidence filled |            5 |           17623 |                76 |           42.222 |                44.444 |           0.431 |              231.882 |
| A3 Glossary first then confidence filled |           10 |           35247 |               104 |           57.778 |                60.819 |           0.295 |              338.913 |
| A3 Glossary first then confidence filled |           20 |           70494 |               136 |           75.556 |                79.532 |           0.193 |              518.338 |
| B (1-confidence) x tier                  |            5 |           17623 |                71 |           39.444 |                41.520 |           0.403 |              248.211 |
| B (1-confidence) x tier                  |           10 |           35247 |                99 |           55.000 |                57.895 |           0.281 |              356.030 |
| B (1-confidence) x tier                  |           20 |           70494 |               134 |           74.444 |                78.363 |           0.190 |              526.075 |
| D Tier gate then learned uncertainty     |            5 |           17623 |                81 |           45.000 |                47.368 |           0.460 |              217.568 |
| D Tier gate then learned uncertainty     |           10 |           35247 |               102 |           56.667 |                59.649 |           0.289 |              345.559 |
| D Tier gate then learned uncertainty     |           20 |           70494 |               137 |           76.111 |                80.117 |           0.194 |              514.555 |
| C1 Disagreement then RiskAware score     |            5 |           17623 |               122 |           67.778 |                71.345 |           0.692 |              144.451 |
| C1 Disagreement then RiskAware score     |           10 |           35247 |               166 |           92.222 |                97.076 |           0.471 |              212.331 |
| C1 Disagreement then RiskAware score     |           20 |           70494 |               167 |           92.778 |                97.661 |           0.237 |              422.120 |

## Paired bootstrap difference vs RiskAware (method minus RiskAware, percentage points)

| method                                   | reference     |   budget_pct |   bootstrap_mean_gap_pp |   ci_lower_pp |   ci_upper_pp |   resamples |   seed |
|:-----------------------------------------|:--------------|-------------:|------------------------:|--------------:|--------------:|------------:|-------:|
| Raw confidence                           | RiskAware-ASR |            5 |                 -16.161 |       -23.916 |        -8.791 |        2000 |     42 |
| Raw confidence                           | RiskAware-ASR |           10 |                  -6.366 |       -12.638 |         0.000 |        2000 |     42 |
| Raw confidence                           | RiskAware-ASR |           20 |                  -1.810 |        -4.918 |         1.070 |        2000 |     42 |
| Learned uncertainty                      | RiskAware-ASR |            5 |                  -9.486 |       -14.975 |        -4.854 |        2000 |     42 |
| Learned uncertainty                      | RiskAware-ASR |           10 |                  -4.542 |        -8.889 |        -0.546 |        2000 |     42 |
| Learned uncertainty                      | RiskAware-ASR |           20 |                  -1.674 |        -3.637 |         0.000 |        2000 |     42 |
| Disagreement + confidence                | RiskAware-ASR |            5 |                  22.196 |        13.559 |        30.953 |        2000 |     42 |
| Disagreement + confidence                | RiskAware-ASR |           10 |                  34.117 |        25.465 |        42.308 |        2000 |     42 |
| Disagreement + confidence                | RiskAware-ASR |           20 |                  18.281 |        12.941 |        24.023 |        2000 |     42 |
| A1 Glossary-only strict                  | RiskAware-ASR |            5 |                 -23.485 |       -30.435 |       -16.667 |        2000 |     42 |
| A1 Glossary-only strict                  | RiskAware-ASR |           10 |                 -35.614 |       -43.478 |       -27.976 |        2000 |     42 |
| A1 Glossary-only strict                  | RiskAware-ASR |           20 |                 -54.495 |       -62.094 |       -46.427 |        2000 |     42 |
| A2 Glossary-restricted confidence strict | RiskAware-ASR |            5 |                 -23.485 |       -30.435 |       -16.667 |        2000 |     42 |
| A2 Glossary-restricted confidence strict | RiskAware-ASR |           10 |                 -35.614 |       -43.478 |       -27.976 |        2000 |     42 |
| A2 Glossary-restricted confidence strict | RiskAware-ASR |           20 |                 -54.495 |       -62.094 |       -46.427 |        2000 |     42 |
| A3 Glossary first then confidence filled | RiskAware-ASR |            5 |                  -1.440 |        -6.945 |         4.488 |        2000 |     42 |
| A3 Glossary first then confidence filled | RiskAware-ASR |           10 |                   2.158 |        -2.891 |         6.989 |        2000 |     42 |
| A3 Glossary first then confidence filled | RiskAware-ASR |           20 |                   1.224 |        -1.841 |         4.186 |        2000 |     42 |
| B (1-confidence) x tier                  | RiskAware-ASR |            5 |                  -4.018 |        -9.548 |         1.765 |        2000 |     42 |
| B (1-confidence) x tier                  | RiskAware-ASR |           10 |                  -0.507 |        -5.715 |         4.652 |        2000 |     42 |
| B (1-confidence) x tier                  | RiskAware-ASR |           20 |                  -0.227 |        -2.762 |         2.140 |        2000 |     42 |
| D Tier gate then learned uncertainty     | RiskAware-ASR |            5 |                   1.465 |         0.000 |         3.571 |        2000 |     42 |
| D Tier gate then learned uncertainty     | RiskAware-ASR |           10 |                   0.988 |        -1.571 |         3.297 |        2000 |     42 |
| D Tier gate then learned uncertainty     | RiskAware-ASR |           20 |                   1.631 |         0.000 |         3.615 |        2000 |     42 |
| C1 Disagreement then RiskAware score     | RiskAware-ASR |            5 |                  24.453 |        16.949 |        32.370 |        2000 |     42 |
| C1 Disagreement then RiskAware score     | RiskAware-ASR |           10 |                  36.664 |        29.104 |        44.199 |        2000 |     42 |
| C1 Disagreement then RiskAware score     | RiskAware-ASR |           20 |                  18.281 |        12.941 |        24.023 |        2000 |     42 |

## RiskAware vs raw confidence

| method        | reference      |   budget_pct |   bootstrap_mean_gap_pp |   ci_lower_pp |   ci_upper_pp |   resamples |   seed |
|:--------------|:---------------|-------------:|------------------------:|--------------:|--------------:|------------:|-------:|
| RiskAware-ASR | Raw confidence |            5 |                  16.161 |         8.791 |        23.916 |        2000 |     42 |
| RiskAware-ASR | Raw confidence |           10 |                   6.366 |        -0.000 |        12.638 |        2000 |     42 |
| RiskAware-ASR | Raw confidence |           20 |                   1.810 |        -1.070 |         4.918 |        2000 |     42 |

## C1 vs disagreement + confidence

| method                               | reference                 |   budget_pct |   bootstrap_mean_gap_pp |   ci_lower_pp |   ci_upper_pp |   resamples |   seed |
|:-------------------------------------|:--------------------------|-------------:|------------------------:|--------------:|--------------:|------------:|-------:|
| C1 Disagreement then RiskAware score | Disagreement + confidence |            5 |                   2.257 |        -1.250 |         6.181 |        2000 |     42 |
| C1 Disagreement then RiskAware score | Disagreement + confidence |           10 |                   2.547 |         0.000 |         5.128 |        2000 |     42 |
| C1 Disagreement then RiskAware score | Disagreement + confidence |           20 |                   0.000 |         0.000 |         0.000 |        2000 |     42 |


## Whisper-base vs Whisper-small — 176-term glossary (paired encounter bootstrap)

Reference-word errors count substitutions and deletions; insertions are not part of the tier-specific denominator. This is **not** overall WER. 2,000 encounter-level paired bootstrap resamples, seed 42, common denominator within each resample. 176-term glossary categories are rebuilt from reference tokens for **both models**.

| Tier | Reference words | Base errors | Small errors | Base error rate (%) | Small error rate (%) | Reduction (pp) | Paired 95% CI (pp) |
|:---|---:|---:|---:|---:|---:|---:|:---|
| 1: Everyday | 357,022 | 36,624 | 32,973 | 10.26 | 9.24 | 1.02 | [0.75, 1.29] |
| 2: Clinical | 178 | 37 | 10 | 20.79 | 5.62 | 15.17 | [8.65, 22.35] |
| 3: Drug/dosage | 303 | 143 | 68 | 47.19 | 22.44 | 24.75 | [19.05, 30.74] |

All three paired confidence intervals exclude zero. The source CSV contains unrounded rates, model-specific 95% CIs and methodological metadata. These figures describe *simulated patient interviews recorded by humans*; no real-world clinical safety benefit was evaluated.

To regenerate this table without a GPU, from private source data:

```bash
python experiments/blockers_yashvi/compute_176_base_small_tier_paired.py \
  --base-labels /private/fareez_base_word_labels.csv \
  --small-labels /private/fareez_small_word_labels.csv \
  --out experiments/blockers_yashvi/y176_whisper_base_small_tier_paired.csv
```

No model inference, classifier retraining, or glossary editing is required.
