"""
Throwaway diagnostic — not part of the reproducible pipeline.

Bootstrap 95% CI on the fareez per-tier error rates (n=60 encounters),
same method as src/model/bootstrap_significance.py: encounter-level
cluster resampling (not word-level, to respect within-encounter
correlation), 2000 resamples, seed 42. Read-only against
data/processed/word_level_labels_fareez.csv — writes nothing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data" / "processed" / "word_level_labels_fareez.csv"

N_BOOTSTRAP = 2000
RNG_SEED = 42


def tier_error_rate(tier_arr, is_error_arr, tier):
    mask = tier_arr == tier
    n = mask.sum()
    if n == 0:
        return np.nan
    return is_error_arr[mask].sum() / n


def main():
    df = pd.read_csv(LABELS_PATH)
    df = df[df["gold_word"].notna()].copy()  # gold-anchored rows only (excludes pure insertions)

    encounter_ids = df["encounter_id"].to_numpy()
    tier = df["tier"].to_numpy()
    is_error = df["is_error"].to_numpy()

    unique_encounters = np.unique(encounter_ids)
    idx_by_encounter = {eid: np.where(encounter_ids == eid)[0] for eid in unique_encounters}
    print(f"Encounters: {len(unique_encounters)}  Gold-anchored rows: {len(df)}\n")

    point_estimates = {t: tier_error_rate(tier, is_error, t) for t in (1, 2, 3)}

    rng = np.random.default_rng(RNG_SEED)
    diffs = {1: [], 2: [], 3: []}

    for _ in range(N_BOOTSTRAP):
        sampled_ids = rng.choice(unique_encounters, size=len(unique_encounters), replace=True)
        idx = np.concatenate([idx_by_encounter[eid] for eid in sampled_ids])
        t_idx, e_idx = tier[idx], is_error[idx]
        for t in (1, 2, 3):
            rate = tier_error_rate(t_idx, e_idx, t)
            if not np.isnan(rate):
                diffs[t].append(rate)

    print(f"{'Tier':<8}{'Point estimate':>16}{'95% CI':>24}{'Resamples used':>18}")
    print("-" * 66)
    for t in (1, 2, 3):
        arr = np.array(diffs[t])
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        pe = point_estimates[t]
        print(f"{t:<8}{pe*100:>15.2f}%{'[' + f'{ci_low*100:.2f}%, {ci_high*100:.2f}%' + ']':>24}{len(arr):>18} / {N_BOOTSTRAP}")


if __name__ == "__main__":
    main()
