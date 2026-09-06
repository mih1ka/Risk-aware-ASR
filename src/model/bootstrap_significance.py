import numpy as np
import pandas as pd

RISK_SCORES_PATH = "data/processed/risk_scores_test.csv"
N_BOOTSTRAP = 2000
FLAG_PCT = 0.10
RNG_SEED = 42


def critical_recall_at_pct(scores, tiers, is_error, pct):
    n = len(scores)
    n_flag = int(n * pct)
    order = np.argsort(-scores)
    flagged_idx = order[:n_flag]
    critical_mask = np.isin(tiers, [2, 3])
    n_critical_errors = is_error[critical_mask].sum()
    if n_critical_errors == 0:
        return np.nan
    flagged_critical_mask = critical_mask[flagged_idx]
    caught = is_error[flagged_idx][flagged_critical_mask].sum()
    return caught / n_critical_errors


def main():
    df = pd.read_csv(RISK_SCORES_PATH)

    encounter_ids = df["encounter_id"].to_numpy()
    risk_score = df["risk_score"].to_numpy()
    uncertainty = df["learned_uncertainty"].to_numpy()
    tier = df["tier"].to_numpy()
    is_error = df["is_error"].to_numpy()

    unique_encounters = np.unique(encounter_ids)
    idx_by_encounter = {eid: np.where(encounter_ids == eid)[0] for eid in unique_encounters}

    rng = np.random.default_rng(RNG_SEED)
    diffs = []

    for _ in range(N_BOOTSTRAP):
        sampled_ids = rng.choice(unique_encounters, size=len(unique_encounters), replace=True)
        idx = np.concatenate([idx_by_encounter[eid] for eid in sampled_ids])

        recall_risk = critical_recall_at_pct(risk_score[idx], tier[idx], is_error[idx], FLAG_PCT)
        recall_unc = critical_recall_at_pct(uncertainty[idx], tier[idx], is_error[idx], FLAG_PCT)

        if not np.isnan(recall_risk) and not np.isnan(recall_unc):
            diffs.append(recall_risk - recall_unc)

    diffs = np.array(diffs)
    print(f"Bootstrap iterations used: {len(diffs)} / {N_BOOTSTRAP}")
    print(f"Mean recall difference (risk_score - uncertainty_only) at top {FLAG_PCT*100:.0f}%: {diffs.mean():.4f}")
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    pct_positive = (diffs > 0).mean()
    print(f"Fraction of bootstrap samples where risk_score >= uncertainty_only: {pct_positive:.3f}")


if __name__ == "__main__":
    main()