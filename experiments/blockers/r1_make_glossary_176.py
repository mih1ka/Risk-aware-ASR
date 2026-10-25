"""
R1 step 1 - build data/glossary/tiered_glossary_176.csv from the 190-term
glossary by removing the 14 terms that were added from Fareez transcripts.
"""
import pandas as pd

from common import GLOSSARY_190, GLOSSARY_176

FAREEZ_TERMS = ["ramipril", "reactine", "rosuvastatin", "spiriva", "ventilin",
                "corticosteroids", "cortico", "steroid", "steroids", "opioids",
                "coronavirus", "clotting", "rheumatological", "antalgic"]


def main():
    g = pd.read_csv(GLOSSARY_190)
    drop = g["term"].str.lower().isin(FAREEZ_TERMS)
    assert drop.sum() == len(FAREEZ_TERMS), sorted(set(FAREEZ_TERMS) - set(g["term"].str.lower()))
    out = g[~drop]
    counts = out["tier"].value_counts().to_dict()
    assert len(out) == 176 and counts == {2: 40, 3: 136}, (len(out), counts)
    out.to_csv(GLOSSARY_176, index=False)
    print(f"Wrote {GLOSSARY_176.name}: {len(out)} rows, tier counts {counts}")


if __name__ == "__main__":
    main()
