"""
Throwaway diagnostic — not part of the reproducible pipeline.

Compares glossary term density and coverage in fareez gold text vs.
aci-bench gold text (aci + virtassist + virtscribe — the same 207
encounters align_and_label.py's whisper_out/ covers, confirmed by
cross-referencing whisper_out encounter_ids against encounter_texts.jsonl's
dataset field).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "eval"))
from align_and_label import load_glossary, normalize, tokenize  # noqa: E402

ENCOUNTER_TEXTS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "encounter_texts.jsonl"

ACI_BENCH_DATASETS = {"aci", "virtassist", "virtscribe"}


def load_gold_texts():
    fareez_texts = []
    aci_bench_texts = []
    with open(ENCOUNTER_TEXTS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec["dataset"] == "fareez":
                fareez_texts.append(rec["full_text"])
            elif rec["dataset"] in ACI_BENCH_DATASETS:
                aci_bench_texts.append(rec["full_text"])
    return fareez_texts, aci_bench_texts


def analyze(texts, glossary):
    total_tokens = 0
    tier_counts = {1: 0, 2: 0, 3: 0}
    seen_terms = set()

    for text in texts:
        for token in tokenize(text):
            total_tokens += 1
            norm = normalize(token)
            tier = glossary.get(norm, 1)
            tier_counts[tier] += 1
            if norm in glossary:
                seen_terms.add(norm)

    return total_tokens, tier_counts, seen_terms


# Baseline numbers from the original 176-term glossary run, for the delta column.
BASELINE = {
    "fareez": {"total_tokens": 357503, "tier2": 178, "tier3": 303, "seen": 42},
    "aci": {"total_tokens": 236274, "tier2": 435, "tier3": 946, "seen": 176},
}


def main():
    glossary = load_glossary()
    print(f"Loaded glossary: {len(glossary)} terms (baseline run used 176)")

    fareez_texts, aci_bench_texts = load_gold_texts()
    print(f"fareez encounters: {len(fareez_texts)}")
    print(f"aci-bench-gold encounters (aci + virtassist + virtscribe): {len(aci_bench_texts)}")
    print()

    fareez_total, fareez_tiers, fareez_seen = analyze(fareez_texts, glossary)
    aci_total, aci_tiers, aci_seen = analyze(aci_bench_texts, glossary)

    def pct(n, total):
        return 100.0 * n / total if total else 0.0

    def delta(new, old):
        d = new - old
        return f"{'+' if d >= 0 else ''}{d}"

    def delta_pct(new_n, new_total, old_n, old_total):
        d = pct(new_n, new_total) - pct(old_n, old_total)
        return f"{'+' if d >= 0 else ''}{d:.2f}pp"

    def print_table(label_title, fareez_val, aci_val, fareez_delta, aci_delta):
        rows_local.append((label_title, fareez_val, aci_val, fareez_delta, aci_delta))

    rows_local = []
    print_table(
        "Total tokens",
        f"{fareez_total:,}", f"{aci_total:,}",
        delta(fareez_total, BASELINE["fareez"]["total_tokens"]),
        delta(aci_total, BASELINE["aci"]["total_tokens"]),
    )
    print_table(
        "Tier 2 count",
        f"{fareez_tiers[2]:,}", f"{aci_tiers[2]:,}",
        delta(fareez_tiers[2], BASELINE["fareez"]["tier2"]),
        delta(aci_tiers[2], BASELINE["aci"]["tier2"]),
    )
    print_table(
        "Tier 2 %",
        f"{pct(fareez_tiers[2], fareez_total):.2f}%", f"{pct(aci_tiers[2], aci_total):.2f}%",
        delta_pct(fareez_tiers[2], fareez_total, BASELINE["fareez"]["tier2"], BASELINE["fareez"]["total_tokens"]),
        delta_pct(aci_tiers[2], aci_total, BASELINE["aci"]["tier2"], BASELINE["aci"]["total_tokens"]),
    )
    print_table(
        "Tier 3 count",
        f"{fareez_tiers[3]:,}", f"{aci_tiers[3]:,}",
        delta(fareez_tiers[3], BASELINE["fareez"]["tier3"]),
        delta(aci_tiers[3], BASELINE["aci"]["tier3"]),
    )
    print_table(
        "Tier 3 %",
        f"{pct(fareez_tiers[3], fareez_total):.2f}%", f"{pct(aci_tiers[3], aci_total):.2f}%",
        delta_pct(fareez_tiers[3], fareez_total, BASELINE["fareez"]["tier3"], BASELINE["fareez"]["total_tokens"]),
        delta_pct(aci_tiers[3], aci_total, BASELINE["aci"]["tier3"], BASELINE["aci"]["total_tokens"]),
    )
    print_table(
        f"Glossary terms seen (of {len(glossary)})",
        f"{len(fareez_seen)}", f"{len(aci_seen)}",
        delta(len(fareez_seen), BASELINE["fareez"]["seen"]),
        delta(len(aci_seen), BASELINE["aci"]["seen"]),
    )

    label_w = max(len(r[0]) for r in rows_local)
    col_w = 18
    delta_w = 12
    header = (f"{'':<{label_w}}  {'fareez':>{col_w}}  {'Δ vs 176':>{delta_w}}"
              f"  {'aci-bench-gold':>{col_w}}  {'Δ vs 176':>{delta_w}}")
    print(header)
    print("-" * len(header))
    for label, fareez_val, aci_val, fareez_d, aci_d in rows_local:
        print(f"{label:<{label_w}}  {fareez_val:>{col_w}}  {fareez_d:>{delta_w}}"
              f"  {aci_val:>{col_w}}  {aci_d:>{delta_w}}")


if __name__ == "__main__":
    main()
