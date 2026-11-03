"""
Fit the 2-input HDB planes shown on hdb-3d-planes.html and save them to hdb_pairs.js.

Each "pair" is two inputs -> one plane:  price = beta + omega1 x input1 + omega2 x input2
The page draws the plane in 3D with a sample of real sales as dots.

Needs hdb_data/hdb_resale_all.csv from hdb_resale_1990_2026.py (run that first).
    conda activate dsai-m3
    python hdb_pairs.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

SRC = Path("hdb_data/hdb_resale_all.csv")
if not SRC.exists():
    raise SystemExit("hdb_data/hdb_resale_all.csv not found - run hdb_resale_1990_2026.py first")

# ------------------------------------------------------------------ 1. load and add columns
hdb = pd.read_csv(SRC, usecols=["month", "year", "town", "storey_range",
                                "floor_area_sqm", "flat_age", "resale_price"])
ends = hdb["storey_range"].str.split(" TO ", expand=True).astype(int)   # "10 TO 12" -> 10, 12
hdb["storey"] = (ends[0] + ends[1]) / 2                                 # middle floor: 11
hdb["price_k"] = hdb["resale_price"] / 1000                              # price in $k

RECENT_FROM = 2023
recent = hdb[hdb["year"] >= RECENT_FROM]
PERIODS = {"recent": recent, "all": hdb}

# input 1, input 2, which sales the plane is fitted on
PAIRS = {
    "area_age":    ("floor_area_sqm", "flat_age", "recent"),
    "area_storey": ("floor_area_sqm", "storey",   "recent"),
    "age_storey":  ("flat_age",       "storey",   "recent"),
    "area_year":   ("floor_area_sqm", "year",     "all"),
    "age_year":    ("flat_age",       "year",     "all"),
}
MIN_ROWS = 30          # a town needs at least this many sales to get its own plane
SAMPLE_TOWN = 120      # dots per town on the page
SAMPLE_ALL = 500       # dots when "All towns" is chosen
COLS = ["floor_area_sqm", "flat_age", "storey", "year", "price_k"]   # order of each dot


def fit(df, a, b):
    X, y = df[[a, b]], df["price_k"]
    m = LinearRegression().fit(X, y)
    return {"n": len(df), "beta": round(m.intercept_, 3), "w1": round(m.coef_[0], 4),
            "w2": round(m.coef_[1], 4), "r2": round(m.score(X, y), 3)}


# ------------------------------------------------------------------ 2. fit every pair
fits = {}
for key, (a, b, period) in PAIRS.items():
    data = PERIODS[period]
    fits[key] = {"ALL": fit(data, a, b)}
    for town, g in data.groupby("town"):
        if len(g) >= MIN_ROWS:
            fits[key][town] = fit(g, a, b)

# ------------------------------------------------------------------ 3. dots: a sample of real sales
samples = {}
for period, data in PERIODS.items():
    rows = {"ALL": data.sample(min(SAMPLE_ALL, len(data)), random_state=0)}
    for town, g in data.groupby("town"):
        rows[town] = g.sample(min(SAMPLE_TOWN, len(g)), random_state=0)
    samples[period] = {t: df[COLS].round(1).values.tolist() for t, df in rows.items()}

# ------------------------------------------------------------------ 4. axis ranges from the data
def span(s, step, lo_q=0.005, hi_q=0.995):
    lo, hi = s.quantile(lo_q), s.quantile(hi_q)
    return [float(step * (lo // step)), float(step * -(-hi // step))]

ranges = {
    "floor_area_sqm": span(recent["floor_area_sqm"], 10),
    "flat_age": [0.0, float(5 * -(-hdb["flat_age"].max() // 5))],
    "storey": [0.0, float(5 * -(-recent["storey"].quantile(0.999) // 5))],
    "year": [float(hdb["year"].min()), float(hdb["year"].max())],
    "price_recent": [0.0, float(100 * -(-recent["price_k"].quantile(0.995) // 100))],
    "price_all": [0.0, float(100 * -(-hdb["price_k"].quantile(0.995) // 100))],
}
when = {p: f"{d['month'].min()[:7]} to {d['month'].max()[:7]}" for p, d in PERIODS.items()}

out = {"pairs": {k: {"x1": a, "x2": b, "period": p} for k, (a, b, p) in PAIRS.items()},
       "fits": fits, "samples": samples, "ranges": ranges, "when": when,
       "rows": {p: len(d) for p, d in PERIODS.items()}}
Path("hdb_pairs.js").write_text("window.HDB_PAIRS = " + json.dumps(out, separators=(",", ":")) + ";\n")

# ------------------------------------------------------------------ 5. print the all-towns planes
print(f"recent = {when['recent']} ({len(recent):,} sales);  all = {when['all']} ({len(hdb):,} sales)")
print(f"{'pair':12} {'period':7} {'beta':>9} {'omega1':>9} {'omega2':>9} {'R2':>6}")
for key, (a, b, period) in PAIRS.items():
    f = fits[key]["ALL"]
    print(f"{key:12} {period:7} {f['beta']:9.2f} {f['w1']:9.3f} {f['w2']:9.3f} {f['r2']:6.3f}")
print("ranges:", ranges)
print(f"saved hdb_pairs.js ({Path('hdb_pairs.js').stat().st_size / 1024:.0f} KB)")
