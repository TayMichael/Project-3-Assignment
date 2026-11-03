"""Validate the hybrid (plane S + LightGBM on S's errors) the same way plane S was chosen.

In hdb_tree_models.py the hybrid beat plane S on the 2020-2026 test ($63.3k vs $73.9k), but that
test is the sealed test: picking a model because it won there is choosing on the test.
Here the hybrid gets the same time-series cross-validation that chose S in hdb_location_model.py:

  fold 2015: train 2001-2014, validate 2015
  fold 2016: train 2001-2015, validate 2016
  ...
  fold 2019: train 2001-2018, validate 2019

The model with the lowest average validation MAE is chosen; a hybrid must beat S by at least
$0.5k on that average, or the simpler plane S stays. Only then is the 2020-2026 test
scored, once, for the chosen model and for plane S.

Models compared (data preparation copied unchanged from hdb_tree_models.py):
  plane_S            the page model (linear plane S)
  hybrid_$_default   S + LightGBM learning S's error in dollars, from flat and location only
  hybrid_%_default   S x LightGBM learning S's error in percent, from flat and location only
  with --grid, also four other LightGBM settings for the dollar hybrid (tuning, assignment point 7)

Writes:
  hdb_data/hybrid_cv.csv        every fold and model
  hdb_data/hybrid_sealed.csv    the one 2020-2026 score for the chosen model and plane S

    conda activate dsai-m3
    python hdb_hybrid_cv.py            (about 15-25 minutes)
    python hdb_hybrid_cv.py --grid     (also tunes the dollar hybrid, about 40-60 minutes)
"""
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb

OUT = Path("hdb_data")
TRAIN_FROM, TEST_FROM, BASE_TOWN = 2001, 2020, "ANG MO KIO"   # 2001: same sales as page model H
CAP_KM = 3.0
RF_ROWS = 150_000               # the random forest trains on a random sample of this many sales (8 GB laptop)
SEED = 42
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")     # LightGBM 4.7 renamed an argument; the old one still works

norm = lambda s: s.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
t0 = time.time()
clock = lambda: f"[{(time.time() - t0) / 60:4.1f} min]"

# ---------------------------------------------------------------- sales, drivers, block lookup (as hdb_location_model.py)
hdb = pd.read_csv(OUT / "hdb_resale_all.csv", low_memory=False,
                  usecols=["month", "year", "town", "flat_type", "flat_model", "block", "street_name",
                           "storey_range", "floor_area_sqm", "flat_age", "resale_price"])
hdb = hdb[hdb["year"] >= TRAIN_FROM].copy()
ends = hdb["storey_range"].str.split(" TO ", expand=True).astype(int)
hdb["storey"] = (ends[0] + ends[1]) / 2
hdb["price_k"] = hdb["resale_price"] / 1000
hdb["date"] = pd.to_datetime(hdb["month"].str[:7])
macro = pd.read_csv(OUT / "macro_annual.csv", index_col="year")
hdb["income_k"] = (hdb["year"] - 1).map(macro["income"]) / 1000          # year Y sale <- year Y-1 income
# supply as in page model H: RFS flats offered a year 1997-2000, then BTO launched, 4 years before the sale,
# with rfs4 = 1 for the sales whose figure is an RFS year (2001-2004), 0 from 2005 and in every future year
hdb["avl4_k"] = (hdb["year"] - 4).map(macro["avl"]) / 1000
hdb["rfs4"] = (hdb["year"] - 4 <= 2000).astype(float)
# as page model S: also HDB flats sold 4 years before (SingStat) and the policy stance known at the sale date
hdb["sold4_k"] = (hdb["year"] - 4).map(macro["sold"]) / 1000 if "sold" in macro else np.nan
pol = pd.read_csv("policy_events.csv", parse_dates=["announce_date"])
pol["w"] = pol["direction"] * pol["intensity"]
_m = pd.to_datetime(np.sort(hdb["date"].unique()))
_st = {m: float(pol.loc[(pol["announce_date"] < m) & (pol["announce_date"] >= m - pd.DateOffset(months=36)), "w"].sum()) for m in _m}
hdb["tight36"] = hdb["date"].map(_st)
hdb["flat_type"] = norm(hdb["flat_type"]).str.replace("MULTI GENERATION", "MULTI-GENERATION")
fm = norm(hdb["flat_model"])
hdb["flat_model"] = fm.where(fm.map(fm.value_counts()) >= 2000, "OTHER")   # rare models grouped
hdb = hdb.drop(columns=["storey_range", "resale_price"])

LOOKUP = OUT / "hdb_block_lookup_filled.csv"
if not LOOKUP.exists():
    LOOKUP = OUT / "hdb_block_lookup_from_V3F.csv"
blk = pd.read_csv(LOOKUP, dtype={"postal_code": str}, low_memory=False)
blk["block"], blk["street_name"] = norm(blk["block"]), norm(blk["street_name"])
blk = blk.drop_duplicates(["block", "street_name"]).dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
blk["bid"] = blk.index
LOC = ["dist_to_cbd_km", "dist_to_mall_km", "dist_to_nearest_high_demand_primary_school_km",
       "within_1km_high_demand_primary_school_flag"]
hdb["block"], hdb["street_name"] = norm(hdb["block"]), norm(hdb["street_name"])
hdb = hdb.merge(blk[["block", "street_name", "bid"] + LOC], on=["block", "street_name"], how="inner")

# ---------------------------------------------------------------- MRT at the sale date (same rules as hdb_location_model.py)
st = pd.read_csv(OUT / "mrt_stations_corrected.csv").dropna(subset=["latitude", "longitude"])
st_ann = pd.to_datetime(st["announcement_date"]).values
st_opn = pd.to_datetime(st["opening_date"]).values
st_mid = st_ann + (st_opn - st_ann) / 2

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

dist = haversine(blk["latitude"].values[:, None], blk["longitude"].values[:, None],
                 st["latitude"].values[None, :], st["longitude"].values[None, :])   # blocks x stations, km
MRT = ["mrt_open_km", "mrt_new_gain", "stage_early", "stage_late"]

def mrt_features(bids, d):
    """MRT features for blocks `bids` on date d: open stations count, announced ones add a gain"""
    is_open = st_opn <= d
    is_ann = (st_ann <= d) & ~is_open
    d_open = np.where(is_open, dist[bids], np.inf).min(axis=1)
    d_ann = np.where(is_ann, dist[bids], np.inf)
    j = d_ann.argmin(axis=1)
    near_ann = d_ann[np.arange(len(bids)), j]
    closer = near_ann < d_open
    late = st_mid[j] <= d
    return {"mrt_open_km": np.minimum(d_open, CAP_KM),
            "mrt_new_gain": np.where(closer, np.minimum(d_open, CAP_KM) - np.minimum(near_ann, CAP_KM), 0.0),
            "stage_early": (closer & ~late).astype(float), "stage_late": (closer & late).astype(float)}

for c in MRT:
    hdb[c] = np.nan
for d in np.unique(hdb["date"].values):
    rows = np.where(hdb["date"].values == d)[0]
    for c, v in mrt_features(hdb["bid"].values[rows], d).items():
        hdb.iloc[rows, hdb.columns.get_loc(c)] = v

# ---------------------------------------------------------------- malls and popular schools at the sale date
# A mall counts from its opening month (closed months excluded, e.g. Century Square Sep 2017 - Jun 2018); a popular
# primary school from the month it started at its HOME site (holding sites during rebuilding are ignored: MOE measures
# the 1 km rule from the registered site). Tested in hdb_dated_location_test.py (S1): same accuracy as undated, but no
# sale is credited with a mall or school that did not exist yet. Dates: hdb_data/malls_dated.csv, school_home_sites.csv.
LOC_FILES = [OUT / "mall_school_locations.csv", OUT / "malls_dated.csv", OUT / "school_home_sites.csv"]
if all(f.exists() for f in LOC_FILES):
    _loc = pd.read_csv(LOC_FILES[0]).dropna(subset=["lat", "lon"])
    _key = lambda s: s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)

    def _month(v):
        """'2019-08-30', '2014-12' or '1997' -> first day of that month (a year alone counts from January)"""
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        s = str(v).strip()
        return pd.Timestamp((s[:7] if len(s) >= 7 and s[4] == "-" else s[:4] + "-01") + "-01")

    _M = _loc[_loc["kind"] == "mall"].assign(k=lambda d: _key(d["name"]))
    _S = _loc[_loc["kind"] == "school"].assign(k=lambda d: _key(d["name"]))
    _md = pd.read_csv(LOC_FILES[1]).assign(k=lambda d: _key(d["name"]))
    _sd = pd.read_csv(LOC_FILES[2]).assign(k=lambda d: _key(d["name"]))
    _M = _M.merge(_md[["k", "opened", "closed", "reopened"]], on="k", how="left")
    _S = _S.merge(_sd[["k", "home_site_from_ym"]], on="k", how="left")
    _m_open, _m_close, _m_re = _M["opened"].map(_month), _M["closed"].map(_month), _M["reopened"].map(_month)
    _m_re = _m_re.where(~(_m_re <= _m_close))                     # an older 'reopened' (a renaming) does not end a closure
    _s_open = _S["home_site_from_ym"].map(_month)
    print(f"\nDated malls and schools: {len(_M)} malls ({_m_open.notna().sum()} dated), {len(_S)} popular schools "
          f"({_s_open.notna().sum()} dated); undated ones count as always there")
    _blk = blk
    _dM = haversine(_blk["latitude"].values[:, None], _blk["longitude"].values[:, None], _M["lat"].values[None, :], _M["lon"].values[None, :])
    _dS = haversine(_blk["latitude"].values[:, None], _blk["longitude"].values[:, None], _S["lat"].values[None, :], _S["lon"].values[None, :])
    _b = hdb["bid"].values.astype(int)
    _mall, _sch = np.full(len(hdb), np.nan), np.full(len(hdb), np.nan)
    for d in np.unique(hdb["date"].values):
        rows = np.where(hdb["date"].values == d)[0]
        ts = pd.Timestamp(d)
        closed = (_m_close.notna() & (_m_close <= ts) & (_m_re.isna() | (_m_re > ts))).values
        m_ok = (_m_open.isna() | (_m_open <= ts)).values & ~closed
        s_ok = (_s_open.isna() | (_s_open <= ts)).values
        _mall[rows] = np.where(m_ok, _dM[_b[rows]], np.inf).min(axis=1)
        _sch[rows] = np.where(s_ok, _dS[_b[rows]], np.inf).min(axis=1)
    _before = (hdb["within_1km_high_demand_primary_school_flag"].fillna(0).values == 1)
    hdb["dist_to_mall_km"] = _mall
    hdb["dist_to_nearest_high_demand_primary_school_km"] = _sch
    hdb["within_1km_high_demand_primary_school_flag"] = (_sch <= 1.0).astype(float)
    print(f"Dated: nearest mall farther than today's for {(hdb['dist_to_mall_km'].values > np.nan_to_num(_dM[_b].min(axis=1)) + 1e-9).mean():.1%} "
          f"of sales; 1 km school flag off (school not there yet) for {(_before & (_sch > 1.0)).mean():.1%}")
else:
    print("\nDated malls/schools: " + ", ".join(str(f) for f in LOC_FILES if not f.exists())
          + " missing - using today's malls and schools for every sale")


NUM = ["floor_area_sqm", "flat_age", "storey", "income_k", "avl4_k", "rfs4", "sold4_k", "tight36"] + LOC + MRT
CAT = ["town", "flat_type", "flat_model"]
hdb = hdb.dropna(subset=NUM).reset_index(drop=True)
AGE_KNOTS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
AGE = [f"age_over_{k}" for k in AGE_KNOTS]
CATS = {c: sorted(hdb[c].unique()) for c in CAT}
print(f"{clock()} {len(hdb):,} sales {hdb['year'].min()}-{hdb['year'].max()}, {len(NUM)} number columns + town, flat type, flat model")

# ---------------------------------------------------------------- the inputs each model sees
def X_linear(d, cols=None):
    """model F: the plane, with the age bands and 0/1 columns for town, flat type and flat model"""
    ages = pd.DataFrame({a: (d["flat_age"] - k).clip(lower=0) for k, a in zip(AGE_KNOTS, AGE)}, index=d.index)
    X = pd.concat([d[NUM], ages,
                   pd.get_dummies(pd.Categorical(d["town"], CATS["town"]), dtype=float).set_axis(CATS["town"], axis=1).set_index(d.index).drop(columns=BASE_TOWN),
                   pd.get_dummies(pd.Categorical(d["flat_type"], CATS["flat_type"]), prefix="type", dtype=float).set_index(d.index).drop(columns="type_4 ROOM"),
                   pd.get_dummies(pd.Categorical(d["flat_model"], CATS["flat_model"]), prefix="model", dtype=float).set_index(d.index).drop(columns="model_MODEL A")],
                  axis=1)
    return X if cols is None else X.reindex(columns=cols, fill_value=0.0)

def X_tree(d):
    """boosting models: plain numbers + town / flat type / flat model as categories (no age bands)"""
    X = d[NUM].astype("float32").copy()
    for c in CAT:
        X[c] = pd.Categorical(d[c], categories=CATS[c])
    return X

def X_forest(d):
    """random forest (scikit-learn needs numbers): plain numbers + 0/1 columns for the categories"""
    parts = [d[NUM].astype("float32")]
    for c in CAT:
        parts.append(pd.get_dummies(pd.Categorical(d[c], CATS[c]), prefix=c, dtype="float32").set_index(d.index))
    return pd.concat(parts, axis=1)

HYB_DROP = ["income_k", "avl4_k", "rfs4", "sold4_k", "tight36"]   # the hybrid's trees get flat + location only: the plane alone carries the market level

def scores(y, p):
    e = p - y
    return {"mae": round(float(np.abs(e).mean()), 1), "rmse": round(float(np.sqrt((e ** 2).mean())), 1),
            "mape": round(float((np.abs(e) / y).mean() * 100), 1), "bias": round(float(e.mean()), 1),
            "r2": round(float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)}

def early_split(n):
    """10% of the training rows held back so the boosting models know when to stop adding trees"""
    rng = np.random.default_rng(SEED)
    val = rng.random(n) < 0.10
    return ~val, val

# ---------------------------------------------------------------- fit every model on one split, score on the other
LGB_P = dict(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, random_state=SEED, verbose=-1)
XGB_P = dict(n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
             enable_categorical=True, max_cat_to_onehot=1, early_stopping_rounds=50, random_state=SEED)

# ================================================================ time-series cross-validation of the hybrid
import sys
GRID = "--grid" in sys.argv
VAL_YEARS = [2015, 2016, 2017, 2018, 2019]
y_all, yr = hdb["price_k"].values, hdb["year"].values
XH = X_tree(hdb).drop(columns=HYB_DROP)         # the hybrid's trees: flat and location only
CONFIGS = {"default": {}}                       # default = the settings used on the page's tree section
if GRID:
    CONFIGS |= {"leaves31": dict(num_leaves=31), "leaves127": dict(num_leaves=127),
                "minleaf400": dict(min_child_samples=400), "slow": dict(learning_rate=0.02, n_estimators=8000)}

def fold(tr_mask, te_mask, only=None):
    """train on tr_mask rows, predict te_mask rows; returns actual prices and each model's predictions"""
    tr, te = np.where(tr_mask)[0], np.where(te_mask)[0]
    dtr, dte = hdb.iloc[tr], hdb.iloc[te]
    ytr = y_all[tr]
    Xl = X_linear(dtr); cols = list(Xl.columns)
    F = LinearRegression().fit(Xl, ytr)
    f_tr, f_te = F.predict(Xl), F.predict(X_linear(dte, cols))
    del Xl; gc.collect()
    out = {"plane_S": f_te}
    fit_, val_ = early_split(len(tr))           # 10% of the TRAINING rows tell LightGBM when to stop
    Xh, Xhte = XH.iloc[tr], XH.iloc[te]
    r, lr = ytr - f_tr, np.log(ytr / np.maximum(f_tr, 50.0))
    for name, extra in CONFIGS.items():
        if only and f"hybrid_$_{name}" not in only and not (name == "default" and "hybrid_%_default" in only):
            continue
        p = LGB_P | extra
        if not only or f"hybrid_$_{name}" in only:
            hy = lgb.LGBMRegressor(**p).fit(Xh[fit_], r[fit_], eval_set=[(Xh[val_], r[val_])],
                                            callbacks=[lgb.early_stopping(50, verbose=False)])
            out[f"hybrid_$_{name}"] = f_te + hy.predict(Xhte)
        if name == "default" and (not only or "hybrid_%_default" in only):
            hr = lgb.LGBMRegressor(**p).fit(Xh[fit_], lr[fit_], eval_set=[(Xh[val_], lr[val_])],
                                            callbacks=[lgb.early_stopping(50, verbose=False)])
            out["hybrid_%_default"] = f_te * np.exp(hr.predict(Xhte))
    return y_all[te], yr[te], out

rows = []
for v in VAL_YEARS:
    y, _, out = fold((yr >= TRAIN_FROM) & (yr < v), yr == v)
    for m, p in out.items():
        s = scores(y, p)
        rows.append({"val_year": v, "model": m, "n": len(y)} | s)
        print(f"{clock()} validate {v}  {m:22} MAE {s['mae']:6}  RMSE {s['rmse']:6}  bias {s['bias']:+7}")
cv = pd.DataFrame(rows)
cv.to_csv(OUT / "hybrid_cv.csv", index=False)

mean = cv.groupby("model")["mae"].mean().sort_values()
wide = cv.pivot(index="model", columns="val_year", values="mae").loc[mean.index]
wide["CV mean"] = mean.round(1)
print("\nValidation MAE ($k) by year, 2015-2019 (lower is better):")
print(wide.to_string())
MARGIN = 0.5    # $k: a hybrid must beat plane S by at least this on the CV mean, or the simpler S stays
best = mean.index[0]
chosen = best if best != "plane_S" and mean[best] <= mean["plane_S"] - MARGIN else "plane_S"
print(f"\nBest CV mean: {best} ${mean[best]:.1f}k; plane S ${mean['plane_S']:.1f}k (gap ${mean['plane_S'] - mean[best]:.1f}k)")
print(f"Chosen by cross-validation (a hybrid must beat S by ${MARGIN}k or more): {chosen}")

# ---------------------------------------------------------------- the sealed test, scored once
only = {chosen, "plane_S"}
y, years, out = fold((yr >= TRAIN_FROM) & (yr < TEST_FROM), yr >= TEST_FROM, only=only)
sealed = []
for m in ["plane_S"] + ([chosen] if chosen != "plane_S" else []):
    s = scores(y, out[m])
    by_year = pd.Series(np.abs(out[m] - y)).groupby(years).mean().round(1)
    sealed.append({"model": m, "chosen": m == chosen, "n": len(y)} | s | {f"mae_{k}": v for k, v in by_year.items()})
sealed = pd.DataFrame(sealed)
sealed.to_csv(OUT / "hybrid_sealed.csv", index=False)
print(f"\nSealed test {TEST_FROM}-{yr.max()}, scored once:")
print(sealed.to_string(index=False))
print(f"\n{clock()} saved hdb_data/hybrid_cv.csv and hdb_data/hybrid_sealed.csv")
