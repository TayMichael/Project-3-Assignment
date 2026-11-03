"""Replay: the page model refitted on sales up to each year E (2006-2025), so the page can forecast
from the past, plus the actual median price of every block and flat type in every year 1990-2026.

How it works
  1. Runs hdb_location_model.py exactly as it is (same sales, same features, same page model; it also
     rewrites hdb_location.js, unchanged). Nothing in that script is modified.
  2. For each training end year E it fits the page model's plane again on sales up to E only.
     The page then prices a flat "as a forecaster at the end of E": every year after E is a forecast.
  3. Scores each refit on the 5 years after E with the drivers that really happened (hindsight),
     so the page can say how well a model trained to E did on E+1..E+5.
  4. Takes the median resale price of each block + flat type in each year from ALL sales, 1990-2026,
     so the page can draw the block's real prices next to the forecasts.

Writes hdb_replay.js (window.HDB_REPLAY), read by hdb-future-price.html if present.

    conda activate dsai-m3
    python hdb_replay_models.py          # about 6-8 minutes (most of it is hdb_location_model.py)
"""
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

CUTS = list(range(2006, 2026))          # training end years; 2001-2005 is too short to fit the plane
NEXT = 5                                # years scored after each training end year

print("Running hdb_location_model.py (unchanged) to build the sales and features ...")
g = runpy.run_path("hdb_location_model.py", run_name="replay")
hdb, X, y, cols = g["hdb"], g["X"], g["y"], g["cols"]
BASE0, AGE, AGE_KNOTS, LOCCOLS = g["BASE0"], g["AGE"], g["AGE_KNOTS"], g["LOC"] + g["MRT"]
sups, STANCE, page = g["sups"], g["STANCE"], g["model"]
r4 = lambda v: round(float(v), 4)
print(f"\nPage model: {page['label']} ({len(hdb):,} sales, {len(X.columns)} columns)")


def as_page_model(m, rows):
    """the same dictionary layout as model S in hdb_location.js, so the page can swap it in"""
    coef = dict(zip(X.columns, m.coef_))
    keep = lambda c: c in BASE0 + sups + STANCE or not (c.startswith("type_") or c.startswith("model_") or c in LOCCOLS or c in AGE or c == "rfs4")
    return {"beta": round(float(m.intercept_), 3),
            "coef": {c: r4(v) for c, v in coef.items() if keep(c)},
            "age_coef": {str(k): r4(coef[a]) for k, a in zip(AGE_KNOTS, AGE) if a in coef},
            "loc_coef": {c: r4(coef[c]) for c in LOCCOLS},
            "type_coef": {c[5:]: r4(v) for c, v in coef.items() if c.startswith("type_")} | {"4 ROOM": 0.0},
            "model_coef": {c[6:]: r4(v) for c, v in coef.items() if c.startswith("model_")} | {"MODEL A": 0.0},
            "n": int(rows.sum()), "period": f"{hdb.loc[rows, 'month'].min()[:7]} to {hdb.loc[rows, 'month'].max()[:7]}"}


cuts, table = {}, []
yr = hdb["year"]
for E in CUTS:
    tr = yr <= E
    m = LinearRegression().fit(X[tr], y[tr])
    d = as_page_model(m, tr)
    nx = (yr > E) & (yr <= E + NEXT)
    if nx.any():
        e = pd.Series(m.predict(X[nx]), index=y[nx].index) - y[nx]
        by = e.groupby(yr[nx]).agg(["mean", lambda s: s.abs().mean()]).round(1)
        d["next"] = {"years": f"{E + 1}-{int(yr[nx].max())}", "mae": round(float(e.abs().mean()), 1),
                     "bias": round(float(e.mean()), 1), "by_year": [[int(k), float(r.iloc[1]), float(r.iloc[0])] for k, r in by.iterrows()]}
        table.append((E, d["n"], d["next"]["years"], d["next"]["mae"], d["next"]["bias"]))
    cuts[str(E)] = d
    print(f"  trained on sales up to {E}: {d['n']:>8,} sales" + (f"; next {NEXT} years {d['next']['years']}: MAE {d['next']['mae']}k, bias {d['next']['bias']:+}k" if "next" in d else ""))

# ---------------------------------------------------------------- actual prices, every block and flat type, 1990-2026
norm = lambda s: s.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
allsales = pd.read_csv(Path("hdb_data") / "hdb_resale_all.csv", usecols=["year", "block", "street_name", "flat_type", "resale_price"], low_memory=False)
allsales["key"] = norm(allsales["block"]) + "|" + norm(allsales["street_name"])
allsales["flat_type"] = norm(allsales["flat_type"]).str.replace("MULTI GENERATION", "MULTI-GENERATION")   # as in hdb_location_model.py
med = (allsales.groupby(["key", "flat_type", "year"])["resale_price"].agg(["median", "size"]).reset_index())
actual = {}
for (key, ft), grp in med.groupby(["key", "flat_type"]):
    actual.setdefault(key, {})[ft] = [v for _, _, yy, md, n in grp.itertuples(index=False, name=None) for v in (int(yy), int(round(md / 1000)), int(n))]
print(f"\nActual prices: {len(med):,} block + flat type + year medians, {len(actual):,} blocks, "
      f"{int(allsales['year'].min())}-{int(allsales['year'].max())}")

out = {"cut_years": CUTS, "next_years": NEXT, "page_label": page["label"], "cols": page["cols"], "cuts": cuts, "actual": actual}
Path("hdb_replay.js").write_text("window.HDB_REPLAY = " + json.dumps(out, separators=(",", ":")) + ";\n")
print(f"Saved hdb_replay.js ({Path('hdb_replay.js').stat().st_size / 1e6:.1f} MB)")
print("\nHow a model trained up to each year did on the next 5 years (actual drivers, $k):")
print(pd.DataFrame(table, columns=["trained_to", "sales", "next_years", "MAE", "bias"]).to_string(index=False))
