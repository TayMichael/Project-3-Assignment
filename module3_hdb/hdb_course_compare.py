"""The course HDB data (2017-2019) scored four ways: the course's features vs model S's flat + location features.

The course notebook (su-ntu-ctp/6m-data-C3.2-HDB_price_predictor) uses town, flat type, floor area, storey and lease
start year, and one random 80/20 split (random_state=42). This script keeps the course's data, models and settings,
and only swaps in model S's flat and location features (flat model, flat age with the 10 lease-decay bends, km to the
CBD, a mall and a popular school, the school-within-1km flag, km to the nearest MRT station open at the sale month,
and a time trend inside the window). Four tests:
  1. the course's own split: random 80/20, random_state=42 (one sample)
  2. the same split repeated with 5 different random seeds: average and spread (many samples, as on the whiteboard)
  3. time-based: train 2017 -> test 2018, and train 2017-2018 -> test 2019 (forecasting the next year)
  4. adding the feature groups one at a time, to see which ones win (course split and the 2019 test)
Malls and popular schools are DATED, as in model S since 31 October 2026: a mall counts from its opening month (closed
months excluded) and a popular school from the month it started at its home site (hdb_data/malls_dated.csv,
school_home_sites.csv, positions in mall_school_locations.csv). "model S, today's malls/schools" keeps the old undated
distances from Blocks.csv, for comparison. If the three files are not found, the dated set is skipped with a warning.
Prints only; nothing is written. Needs lightgbm (already in dsai-m3).
    conda activate dsai-m3
    python hdb_course_compare.py <course hdb_resale_2017_2019.csv> <Blocks.csv> <MRT_Stations.csv> [hdb_data folder]
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

COURSE_CSV, BLOCKS_CSV, MRT_CSV = sys.argv[1:4]
LOC_DIR = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("hdb_data")
SEEDS = [42, 1, 2, 3, 4]

# ---------------------------------------------------------------- the course data, plus model S's features
df = pd.read_csv(COURSE_CSV)
b = df["storey_range"].str.split(" TO ", expand=True).astype(int)
df["storey"] = (b[0] + b[1]) / 2
df["year"] = df["month"].str[:4].astype(int)
df["date"] = df["year"] + (df["month"].str[5:7].astype(int) - 0.5) / 12
df["flat_age"] = df["year"] - df["lease_commence_date"]
norm = lambda s: s.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
df["block_key"] = norm(df["block"]) + " " + norm(df["street_name"])
fm = norm(df["flat_model"])
df["flat_model"] = fm.where(fm.map(fm.value_counts()) >= 300, "OTHER")        # rare flat models grouped
blocks = pd.read_csv(BLOCKS_CSV)
df = df.merge(blocks[["block_key", "latitude", "longitude", "dist_to_cbd_km", "dist_to_mall_km",
                      "dist_to_school_km", "school_within_1km"]], on="block_key", how="left")
print(f"{len(df):,} sales 2017-2019; matched to a located block: {df['latitude'].notna().sum():,}")

st = pd.read_csv(MRT_CSV)                                                      # nearest station open at the sale month
r = np.pi / 180
la, lo = df["latitude"].values[:, None] * r, df["longitude"].values[:, None] * r
sl, so = st["latitude"].values[None, :] * r, st["longitude"].values[None, :] * r
d = 6371 * 2 * np.arcsin(np.sqrt(np.sin((sl - la) / 2) ** 2 + np.cos(la) * np.cos(sl) * np.sin((so - lo) / 2) ** 2))
d = np.where(st["opened_year"].values[None, :] <= df["date"].values[:, None], d, np.inf)
df["mrt_open_km"] = np.minimum(d.min(axis=1), 3)                               # capped at 3 km, as in model S
for k in range(10, 60, 5):
    df[f"age_over_{k}"] = (df["flat_age"] - k).clip(lower=0)                   # the 10 lease-decay bends
df["t"] = df["date"] - 2017                                                    # time trend inside the window

# ---------------------------------------------------------------- malls and popular schools at the sale date (as model S)
LOC_FILES = [LOC_DIR / "mall_school_locations.csv", LOC_DIR / "malls_dated.csv", LOC_DIR / "school_home_sites.csv"]
DATED = all(f.exists() for f in LOC_FILES)
if DATED:
    key = lambda s: s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)

    def month(v):
        """'2019-08-30', '2014-12' or '1997' -> first day of that month (a year alone counts from January)"""
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        s = str(v).strip()
        return pd.Timestamp((s[:7] if len(s) >= 7 and s[4] == "-" else s[:4] + "-01") + "-01")

    loc = pd.read_csv(LOC_FILES[0]).dropna(subset=["lat", "lon"])
    M = loc[loc["kind"] == "mall"].assign(k=lambda d: key(d["name"])).merge(
        pd.read_csv(LOC_FILES[1]).assign(k=lambda d: key(d["name"]))[["k", "opened", "closed", "reopened"]], on="k", how="left")
    Sc = loc[loc["kind"] == "school"].assign(k=lambda d: key(d["name"])).merge(
        pd.read_csv(LOC_FILES[2]).assign(k=lambda d: key(d["name"]))[["k", "home_site_from_ym"]], on="k", how="left")
    m_open, m_close, m_re = M["opened"].map(month), M["closed"].map(month), M["reopened"].map(month)
    m_re = m_re.where(~(m_re <= m_close))                         # an older 'reopened' (a renaming) does not end a closure
    s_open = Sc["home_site_from_ym"].map(month)
    hv = lambda la1, lo1, la2, lo2: 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) * r / 2) ** 2 + np.cos(la1 * r) * np.cos(la2 * r) * np.sin((lo2 - lo1) * r / 2) ** 2))
    dM = hv(df["latitude"].values[:, None], df["longitude"].values[:, None], M["lat"].values[None, :], M["lon"].values[None, :])
    dS = hv(df["latitude"].values[:, None], df["longitude"].values[:, None], Sc["lat"].values[None, :], Sc["lon"].values[None, :])
    mall_d, sch_d = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for mo, rows in df.groupby("month").indices.items():
        ts = pd.Timestamp(mo + "-01")
        closed = (m_close.notna() & (m_close <= ts) & (m_re.isna() | (m_re > ts))).values
        m_ok = (m_open.isna() | (m_open <= ts)).values & ~closed
        s_ok = (s_open.isna() | (s_open <= ts)).values
        mall_d[rows] = np.where(m_ok, dM[rows], np.inf).min(axis=1)
        sch_d[rows] = np.where(s_ok, dS[rows], np.inf).min(axis=1)
    df["mall_dated_km"], df["school_dated_km"] = mall_d, sch_d
    df["school_1km_dated"] = (sch_d <= 1.0).astype(float)
    have = df["latitude"].notna()
    farther = have & (mall_d > dM.min(axis=1) + 1e-9)          # vs today's malls from the same positions (Blocks.csv is rounded to 3 decimals)
    s_off = have & (dS.min(axis=1) <= 1.0) & (sch_d > 1.0)
    print(f"Dated: {len(M)} malls ({m_open.notna().sum()} dated), {len(Sc)} popular schools ({s_open.notna().sum()} dated)")
    print("  nearest mall farther than today's (a nearer one not open yet, or closed):",
          {y: f"{farther[df['year'] == y].mean():.1%}" for y in (2017, 2018, 2019)})
    print("  1 km school flag off (school not at its home site yet):",
          {y: f"{s_off[df['year'] == y].mean():.1%}" for y in (2017, 2018, 2019)})
    print("  check: undated distances recomputed from the positions match Blocks.csv for "
          f"{(np.abs(np.where(have, dM.min(axis=1), np.nan) - df['dist_to_mall_km']) < 0.01)[have].mean():.1%} (malls) and "
          f"{(np.abs(np.where(have, dS.min(axis=1), np.nan) - df['dist_to_school_km']) < 0.01)[have].mean():.1%} (schools) of sales")
else:
    print("Dated malls/schools: " + ", ".join(str(f) for f in LOC_FILES if not f.exists()) + " not found - only today's malls and schools")

y = df["resale_price"].values
ok = df["latitude"].notna().values
FEATS = {"course": (["town", "flat_type"], ["floor_area_sqm", "storey", "lease_commence_date"]),
         "model S": (["town", "flat_type", "flat_model"],
                     ["floor_area_sqm", "storey", "flat_age"] + [f"age_over_{k}" for k in range(10, 60, 5)]
                     + ["dist_to_cbd_km", "dist_to_mall_km", "dist_to_school_km", "school_within_1km", "mrt_open_km", "t"])}
if DATED:                                                       # model S now: dated malls and schools; the old set kept for comparison
    FEATS["model S, today's malls/schools"] = FEATS.pop("model S")
    c, n = FEATS["model S, today's malls/schools"]
    FEATS = {"course": FEATS["course"],
             "model S": (c, [{"dist_to_mall_km": "mall_dated_km", "dist_to_school_km": "school_dated_km",
                              "school_within_1km": "school_1km_dated"}.get(x, x) for x in n]),
             "model S, today's malls/schools": (c, n)}
MODELS = {"Linear regression": lambda: LinearRegression(),
          "Random forest": lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42),
          "Gradient boosting": lambda: GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.1, random_state=42),
          "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)}


def score(model, feats, tr, te):
    cat, num = FEATS[feats]
    pipe = Pipeline([("prep", ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")),
                     ("model", MODELS[model]())])
    pipe.fit(df.loc[tr, cat + num], y[tr])
    e = pipe.predict(df.loc[te, cat + num]) - y[te]
    return {"mae": np.abs(e).mean(), "mape": (np.abs(e) / y[te]).mean() * 100, "bias": e.mean()}


def show(title, rows):
    print(f"\n{title}")
    print(pd.DataFrame(rows).round({"mae": 0, "mape": 1, "bias": 0, "sd": 0}).to_string(index=False))


# ---------------------------------------------------------------- 1. the course's own split (one sample)
idx = np.arange(len(df))
tr_i, te_i = train_test_split(idx, test_size=0.2, random_state=42)
tr, te = np.isin(idx, tr_i) & ok, np.isin(idx, te_i) & ok
rows = []
for m in MODELS:
    for f in FEATS:
        t0 = time.time()
        rows.append({"model": m, "features": f, **score(m, f, tr, te), "secs": round(time.time() - t0)})
show(f"1. Course split, random 80/20, random_state=42 (train {tr.sum():,}, test {te.sum():,})", rows)

# ---------------------------------------------------------------- 2. the same split with 5 seeds (many samples)
rows = []
for m in ("Linear regression", "Random forest", "LightGBM"):
    for f in ("course", "model S"):
        maes = []
        for s in SEEDS:
            a, b_ = train_test_split(idx, test_size=0.2, random_state=s)
            maes.append(score(m, f, np.isin(idx, a) & ok, np.isin(idx, b_) & ok)["mae"])
        rows.append({"model": m, "features": f, "mae": np.mean(maes), "sd": np.std(maes, ddof=1),
                     "lowest": round(min(maes)), "highest": round(max(maes))})
show(f"2. Random 80/20 split repeated with seeds {SEEDS}: mean MAE and spread", rows)

# ---------------------------------------------------------------- 3. time-based: forecast the next year
rows = []
yr = df["year"].values
for train_years, test_year in (((2017,), 2018), ((2017, 2018), 2019)):
    tr, te = np.isin(yr, train_years) & ok, (yr == test_year) & ok
    for m in ("Linear regression", "Random forest", "LightGBM"):
        for f in FEATS:
            rows.append({"train": "-".join(map(str, train_years)), "test": test_year, "model": m, "features": f,
                         **score(m, f, tr, te)})
show("3. Time-based: train on earlier years, test on the next year", rows)

# ---------------------------------------------------------------- 4. which features win: one group at a time
BASE_CAT, BASE_NUM = FEATS["course"]
BENDS = ["flat_age"] + [f"age_over_{k}" for k in range(10, 60, 5)]
LOC = ["dist_to_cbd_km"] + (["mall_dated_km", "school_dated_km", "school_1km_dated"] if DATED else ["dist_to_mall_km", "dist_to_school_km", "school_within_1km"])
STEPS = [("course features", BASE_CAT, BASE_NUM),
         ("+ flat model", BASE_CAT + ["flat_model"], BASE_NUM),
         ("+ lease-decay bends", BASE_CAT + ["flat_model"], BASE_NUM + BENDS),
         ("+ CBD, mall, school", BASE_CAT + ["flat_model"], BASE_NUM + BENDS + LOC),
         ("+ MRT", BASE_CAT + ["flat_model"], BASE_NUM + BENDS + LOC + ["mrt_open_km"]),
         ("+ time trend (all of ours)", BASE_CAT + ["flat_model"], BASE_NUM + BENDS + LOC + ["mrt_open_km", "t"]),
         ("course + CBD, mall, school only", BASE_CAT, BASE_NUM + LOC),
         ("course + MRT only", BASE_CAT, BASE_NUM + ["mrt_open_km"])]
a, b_ = train_test_split(idx, test_size=0.2, random_state=42)
random_split, next_year = (np.isin(idx, a) & ok, np.isin(idx, b_) & ok), ((yr < 2019) & ok, (yr == 2019) & ok)
rows = []
for name, cat, num in STEPS:
    FEATS[name] = (cat, num)
    row = {"features": name}
    for m in ("Linear regression", "LightGBM"):
        row[f"{m}, random"] = score(m, name, *random_split)["mae"]
        row[f"{m}, 2019"] = score(m, name, *next_year)["mae"]
    rows.append(row)
print("\n4. Adding feature groups one at a time (MAE, $): course split and the 2019 time test")
print(pd.DataFrame(rows).round(0).to_string(index=False))

print("\nMedian resale price by year ($):", df.groupby("year")["resale_price"].median().astype(int).to_dict())
