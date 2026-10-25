"""
R0 - reproduce the published Fareez baseline before running anything else.

Expected (190-term glossary, Whisper-base, 272 encounters):
  tier error rates 10.24% / 24.33% / 55.04% on 356,873 / 263 / 367 gold words;
  at a 10% budget raw confidence catches 129 of 245 scorable critical errors,
  RiskAware catches 154 or 155.

Re-aligns all 272 whisper_out files in memory with the 190-term glossary and
checks the result matches the stored labels file. Exits non-zero on any
mismatch. Writes r0_baseline_check.csv (aggregates only).
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

from common import (ROOT, OUT_DIR, GLOSSARY_190, FAREEZ_LABELS_190, NO_TIER_MODEL_PATH,
                    load_glossary, read_labels, tier_rates_with_ci, score_rows, caught_at_budget,
                    n_critical_errors)

sys.path.insert(0, str(ROOT / "src" / "eval"))
import align_and_label  # noqa: E402

EXPECTED_RATES = {1: 10.24, 2: 24.33, 3: 55.04}
EXPECTED_GOLD = {1: 356873, 2: 263, 3: 367}


def main():
    glossary = load_glossary(GLOSSARY_190)
    assert len(glossary) == 190, len(glossary)

    stored = read_labels(FAREEZ_LABELS_190)
    selected = set(json.loads((ROOT / "scratch" / "fareez_272_gold_text.json").read_text()))
    files = sorted(f for f in (ROOT / "data" / "whisper_out").glob("*.json") if f.stem in selected)
    fresh, _ = align_and_label.align_files(files, glossary)
    cols = ["encounter_id", "gold_word", "whisper_word", "error_type", "is_error", "tier"]
    fresh["tier"] = fresh["tier"].astype(float)
    same = (len(fresh) == len(stored)
            and fresh[cols].fillna("").astype(str).equals(stored[cols].fillna("").astype(str)))
    print(f"Encounters: {len(files)}  stored rows: {len(stored)}  fresh rows: {len(fresh)}  identical: {same}")

    tiers, n_enc = tier_rates_with_ci(stored)
    print(tiers.to_string(index=False))

    model = joblib.load(NO_TIER_MODEL_PATH)
    df = score_rows(stored, glossary, model)
    n_all, n_scorable = n_critical_errors(stored)
    t, e = df["tier"].to_numpy(), df["is_error"].to_numpy()
    raw_c, _ = caught_at_budget(df["raw_uncertainty"].to_numpy(), t, e, 0.10)
    risk_c, _ = caught_at_budget(df["risk_score"].to_numpy(), t, e, 0.10)
    # Unstable sort, as in scratch/precision_recall_at_budgets.py, for comparison.
    n_flag = int(len(df) * 0.10)
    crit = np.isin(t, [2, 3])
    flagged = np.argsort(-df["risk_score"].to_numpy())[:n_flag]
    risk_unstable = int(e[flagged][crit[flagged]].sum())
    print(f"Critical errors: all={n_all} scorable={n_scorable}")
    print(f"@10%: raw caught {raw_c}, RiskAware caught {risk_c} (stable sort), {risk_unstable} (default sort)")

    checks = []
    for _, r in tiers.iterrows():
        t_ = int(r["tier"])
        checks.append((f"tier{t_}_rate_pct", EXPECTED_RATES[t_], round(r["error_rate_pct"], 2)))
        checks.append((f"tier{t_}_gold_words", EXPECTED_GOLD[t_], int(r["gold_words"])))
    checks += [("encounters", 272, n_enc),
               ("scorable_critical_errors", 245, n_scorable),
               ("raw_caught_at_10pct", 129, raw_c),
               ("riskaware_caught_at_10pct", "154 or 155", risk_c),
               ("realignment_matches_stored", True, same)]
    out = pd.DataFrame(checks, columns=["check", "expected", "observed"])
    out["match"] = [
        (o in (154, 155)) if exp == "154 or 155" else (exp == o) for _, exp, o in checks
    ]
    out.to_csv(OUT_DIR / "r0_baseline_check.csv", index=False)
    print(out.to_string(index=False))
    if not out["match"].all():
        print("\nBASELINE DOES NOT REPRODUCE - STOP.")
        sys.exit(1)
    print("\nBaseline reproduced.")


if __name__ == "__main__":
    main()
