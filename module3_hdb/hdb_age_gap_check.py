"""Did old flats rise as fast as young ones?  Median resale price of each flat-age band, year by year (a check, prints only).

Reads hdb_data/hdb_resale_all.csv (from hdb_resale_1990_2026.py). For each sale year 2001-2026 and each age band
(0-9, 10-19, 20-29, 30-39, 40-49, 50+ years at the sale):
  1. median price per m2 (all flat types): takes out the flat size, which differs a lot between old and new flats
  2. median price of 4-room flats only: a like-for-like flat type
  3. old vs new: price per m2 of 40-49 and 50+ flats as a % of 0-9 flats, and the dollar gap for a 90 m2 flat
Cells with fewer than 100 sales are left blank. These are raw medians: town, storey and the market mix are
NOT held constant (the model does that), so read them as what buyers actually paid.

Writes hdb_data/age_gap_by_year.csv and hdb_data/age_gap_by_year.png (charts, optional). Nothing else changes.
    conda activate dsai-m3
    python hdb_age_gap_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("hdb_data")
MIN_N = 100
BANDS = [-1, 9, 19, 29, 39, 49, 99]
LABELS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50+"]

d = pd.read_csv(OUT / "hdb_resale_all.csv", usecols=["year", "flat_type", "floor_area_sqm", "flat_age", "resale_price"],
                low_memory=False)
d = d[d["year"] >= 2001].copy()
d["band"] = pd.cut(d["flat_age"], BANDS, labels=LABELS)
d["psm"] = d["resale_price"] / d["floor_area_sqm"]
d["flat_type"] = d["flat_type"].str.upper().str.replace("MULTI GENERATION", "MULTI-GENERATION")


def table(df, value):
    g = df.groupby(["year", "band"], observed=True)[value].agg(["median", "size"])
    med = g["median"].where(g["size"] >= MIN_N).unstack("band")
    return med.reindex(columns=LABELS)


n = d.groupby(["year", "band"], observed=True).size().unstack("band").reindex(columns=LABELS).fillna(0).astype(int)
psm = table(d, "psm").round(0)
four = (table(d[d["flat_type"] == "4 ROOM"], "resale_price") / 1000).round(0)

pd.set_option("display.width", 200)
print("Sales by year and flat age at sale:")
print(n.to_string())
print(f"\n1. Median price per m2 ($), all flat types (blank = fewer than {MIN_N} sales):")
print(psm.to_string())
print(f"\n2. Median price of 4-room flats ($k):")
print(four.to_string())

cmp = pd.DataFrame({
    "new 0-9 $/m2": psm["0-9"],
    "40-49 $/m2": psm["40-49"],
    "40-49 as % of new": (psm["40-49"] / psm["0-9"] * 100).round(1),
    "50+ as % of new": (psm["50+"] / psm["0-9"] * 100).round(1),
    "gap 40-49 vs new, 90 m2 ($k)": ((psm["40-49"] - psm["0-9"]) * 90 / 1000).round(0),
})
print("\n3. Old flats compared with new ones (price per m2):")
print(cmp.to_string())

base = int(psm["40-49"].first_valid_index())          # first year with enough 40-49-year-old sales to compare
growth = (psm / psm.loc[base] * 100).round(0)
print(f"\n4. Price per m2 as % of its own {base} level (the first year with {MIN_N}+ sales of 40-49-year-old flats):")
print(growth.loc[[y for y in sorted(set([base] + list(range(base, 2027, 2)) + [2013, 2014, 2026])) if y in growth.index]].to_string())

out = psm.add_prefix("psm_").join(four.add_prefix("med4room_k_")).join(n.add_prefix("n_")).join(cmp)
out.to_csv(OUT / "age_gap_by_year.csv")
print(f"\nSaved {OUT / 'age_gap_by_year.csv'}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for b in LABELS:
        ax[0].plot(psm.index, psm[b], marker="o", ms=3, label=f"age {b}")
    ax[0].axvline(2013.5, color="grey", ls="--"); ax[0].text(2013.6, ax[0].get_ylim()[1] * 0.95, "1 Jul 2013 rule", color="grey")
    ax[0].set_title("Median resale price per m2 by flat age"); ax[0].set_ylabel("$ per m2"); ax[0].legend()
    ax[1].plot(cmp.index, cmp["40-49 as % of new"], marker="o", label="40-49 years")
    ax[1].plot(cmp.index, cmp["50+ as % of new"], marker="o", label="50+ years")
    ax[1].axvline(2013.5, color="grey", ls="--")
    ax[1].set_title("Old flats' price per m2 as % of new flats (0-9 years)"); ax[1].set_ylabel("%"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "age_gap_by_year.png", dpi=130)
    print(f"Saved {OUT / 'age_gap_by_year.png'}")
except Exception as e:
    print("Chart skipped:", e)
