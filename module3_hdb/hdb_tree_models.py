"""Lesson 3.4 models on the HDB data: random forest, gradient boosting, XGBoost, LightGBM,
and a hybrid (the linear plane F + LightGBM on F's errors), all compared with model F.

Same sales and same features as model F in hdb_location_model.py (2005 on, BTO lag 4),
except the tree models get plain flat_age instead of the age bands: they find the
lease-decay curve by themselves.

Two tests, both scored on sales the models never saw:
  forecast   train 2005-2019, test 2020-2026   (the same back-test as the page)
  valuation  random 80% / 20% of all 2005-2026 sales (value a flat in a year the model has seen)

Writes:
  hdb_trees.js                    for hdb-future-price.html: the comparison table, feature importance,
                                  the age curve each model learned, and each model's price today for
                                  the typical flat of every block and flat type
  hdb_data/tree_backtest.csv      the comparison table
  hdb_data/tree_importance.png    feature importance chart (for slides)
  hdb_data/tree_age_curve.png     age curve chart (for slides)

Needs the same files as hdb_location_model.py (run hdb_drivers_model.py first for macro_annual.csv).
    conda activate dsai-m3
    pip install -U scikit-learn xgboost lightgbm      (once)
    python hdb_tree_models.py                         (about 10-25 minutes on a laptop)
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

def run(name, tr, te):
    """train on rows tr, score on rows te; returns {model: {train scores, test scores, per-year test MAE}} and the fitted models"""
    dtr, dte = hdb.loc[tr], hdb.loc[te]
    ytr, yte = dtr["price_k"].values, dte["price_k"].values
    fit_, val_ = early_split(len(dtr))
    res, fitted = {}, {}

    def keep(key, ptr, pte, extra=None):
        res[key] = {"train": scores(ytr, ptr), "test": scores(yte, pte),
                    "years": {int(k): float(v) for k, v in pd.Series(np.abs(pte - yte)).groupby(dte["year"].values).mean().round(1).items()}}
        if extra:
            res[key].update(extra)
        print(f"{clock()} {name:9} {key:24} train MAE {res[key]['train']['mae']:6}  test MAE {res[key]['test']['mae']:6}"
              f"  RMSE {res[key]['test']['rmse']:6}  bias {res[key]['test']['bias']:+7}")

    # 1. linear plane F
    Xl = X_linear(dtr); cols = list(Xl.columns)
    F = LinearRegression().fit(Xl, ytr)
    f_tr, f_te = F.predict(Xl), F.predict(X_linear(dte, cols))
    keep("linear_F", f_tr, f_te); fitted["linear_F"] = (F, cols)
    del Xl; gc.collect()

    # 2. random forest (bagging), on a random sample of the training sales
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(dtr), size=min(RF_ROWS, len(dtr)), replace=False)
    Xf = X_forest(dtr.iloc[sub])
    rf = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=5, max_features=0.4,
                               bootstrap=True, n_jobs=-1, random_state=SEED).fit(Xf, ytr[sub])
    del Xf; gc.collect()
    keep("random_forest", rf.predict(X_forest(dtr)), rf.predict(X_forest(dte)), {"trees": 150, "rows": int(len(sub))})
    fitted["random_forest"] = rf

    Xt, Xte = X_tree(dtr), X_tree(dte)
    # 3. gradient boosting (scikit-learn's fast histogram version)
    hgb = HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
                                        validation_fraction=0.1, n_iter_no_change=30, categorical_features="from_dtype",
                                        random_state=SEED).fit(Xt, ytr)
    keep("gradient_boosting", hgb.predict(Xt), hgb.predict(Xte), {"trees": int(hgb.n_iter_)}); fitted["gradient_boosting"] = hgb

    # 4. XGBoost
    xg = xgb.XGBRegressor(**XGB_P).fit(Xt[fit_], ytr[fit_], eval_set=[(Xt[val_], ytr[val_])], verbose=False)
    keep("xgboost", xg.predict(Xt), xg.predict(Xte), {"trees": int(xg.best_iteration) + 1}); fitted["xgboost"] = xg

    # 5. LightGBM
    lg = lgb.LGBMRegressor(**LGB_P).fit(Xt[fit_], ytr[fit_], eval_set=[(Xt[val_], ytr[val_])],
                                        callbacks=[lgb.early_stopping(50, verbose=False)])
    keep("lightgbm", lg.predict(Xt), lg.predict(Xte), {"trees": int(lg.best_iteration_)}); fitted["lightgbm"] = lg

    # 6. hybrid: the plane F carries the market level and the trend; LightGBM learns what F gets wrong (its errors)
    #    from the flat and its location only. Income and BTO are left out of the trees: given them, the trees learned
    #    "in these income years F was off by this much", which does not carry into years with higher income.
    r_tr = ytr - f_tr
    Xh, Xhte = Xt.drop(columns=HYB_DROP), Xte.drop(columns=HYB_DROP)
    hy = lgb.LGBMRegressor(**LGB_P).fit(Xh[fit_], r_tr[fit_], eval_set=[(Xh[val_], r_tr[val_])],
                                        callbacks=[lgb.early_stopping(50, verbose=False)])
    keep("hybrid_F_lightgbm", f_tr + hy.predict(Xh), f_te + hy.predict(Xhte), {"trees": int(hy.best_iteration_)})
    fitted["hybrid_F_lightgbm"] = hy

    # 7. hybrid in percent: the trees learn by what PERCENT the plane is off (log of actual / plane), so the
    #    correction grows with the market: price = plane x ratio, instead of plane + dollars
    lr_tr = np.log(ytr / np.maximum(f_tr, 50.0))
    hr = lgb.LGBMRegressor(**LGB_P).fit(Xh[fit_], lr_tr[fit_], eval_set=[(Xh[val_], lr_tr[val_])],
                                        callbacks=[lgb.early_stopping(50, verbose=False)])
    keep("hybrid_ratio", f_tr * np.exp(hr.predict(Xh)), f_te * np.exp(hr.predict(Xhte)), {"trees": int(hr.best_iteration_)})
    fitted["hybrid_ratio"] = hr
    del Xt, Xte; gc.collect()
    return res, fitted

LABEL = {"linear_F": "Linear plane S (page model)", "random_forest": "Random forest (bagging)",
         "gradient_boosting": "Gradient boosting", "xgboost": "XGBoost", "lightgbm": "LightGBM",
         "hybrid_F_lightgbm": "Hybrid $: plane + LightGBM dollars (flat and location only)",
         "hybrid_ratio": "Hybrid %: plane x LightGBM percent (flat and location only)"}
ORDER = list(LABEL)

is_test = hdb["year"] >= TEST_FROM
print(f"\n{clock()} Forecast test: train {TRAIN_FROM}-{TEST_FROM - 1} ({(~is_test).sum():,} sales), test {TEST_FROM}-{hdb['year'].max()} ({is_test.sum():,} sales)")
fc, fc_models = run("forecast", hdb.index[~is_test], hdb.index[is_test])
del fc_models; gc.collect()

rng = np.random.default_rng(SEED)
rand_test = rng.random(len(hdb)) < 0.20
print(f"\n{clock()} Valuation test: random 80% / 20% of all {TRAIN_FROM}-{hdb['year'].max()} sales")
va, va_models = run("valuation", hdb.index[~rand_test], hdb.index[rand_test])
del va_models; gc.collect()

table = pd.DataFrame([{"model": LABEL[k],
                       "forecast_train_mae": fc[k]["train"]["mae"], "forecast_test_mae": fc[k]["test"]["mae"],
                       "forecast_test_rmse": fc[k]["test"]["rmse"], "forecast_test_bias": fc[k]["test"]["bias"],
                       "valuation_test_mae": va[k]["test"]["mae"], "valuation_test_rmse": va[k]["test"]["rmse"],
                       "trees_forecast": fc[k].get("trees", "")} for k in ORDER])
print("\nComparison ($k per sale; lower is better):")
print(table.to_string(index=False))
print("\nForecast test, MAE by year ($k):")
print(pd.DataFrame({k: fc[k]["years"] for k in ORDER}).round(1).to_string())
table.to_csv(OUT / "tree_backtest.csv", index=False)

# ---------------------------------------------------------------- final models on ALL sales: importance, age curve, price today
print(f"\n{clock()} Refitting every model on all {len(hdb):,} sales for the page ...")
all_idx = hdb.index
F_all, cols_all = None, None
Xl = X_linear(hdb); cols_all = list(Xl.columns)
F_all = LinearRegression().fit(Xl, hdb["price_k"].values)
f_all = F_all.predict(Xl); del Xl; gc.collect()
y_all = hdb["price_k"].values
sub = np.random.default_rng(SEED).choice(len(hdb), size=min(RF_ROWS, len(hdb)), replace=False)
rf_all = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=5, max_features=0.4,
                               n_jobs=-1, random_state=SEED).fit(X_forest(hdb.iloc[sub]), y_all[sub])
Xt = X_tree(hdb)
hgb_all = HistGradientBoostingRegressor(max_iter=fc["gradient_boosting"]["trees"], learning_rate=0.1, max_leaf_nodes=63,
                                        early_stopping=False, categorical_features="from_dtype", random_state=SEED).fit(Xt, y_all)
xgp = {k: v for k, v in XGB_P.items() if k != "early_stopping_rounds"} | {"n_estimators": fc["xgboost"]["trees"]}
xg_all = xgb.XGBRegressor(**xgp).fit(Xt, y_all, verbose=False)
lg_all = lgb.LGBMRegressor(**(LGB_P | {"n_estimators": fc["lightgbm"]["trees"]})).fit(Xt, y_all)
hy_all = lgb.LGBMRegressor(**(LGB_P | {"n_estimators": max(fc["hybrid_F_lightgbm"]["trees"], 1)})).fit(Xt.drop(columns=HYB_DROP), y_all - f_all)
hr_all = lgb.LGBMRegressor(**(LGB_P | {"n_estimators": max(fc["hybrid_ratio"]["trees"], 1)})).fit(Xt.drop(columns=HYB_DROP), np.log(y_all / np.maximum(f_all, 50.0)))
del Xt; gc.collect()
print(f"{clock()} done")

def predict_all(d):
    """price in $k from every final model, for the rows of d"""
    Xt = X_tree(d); fl = F_all.predict(X_linear(d, cols_all))
    return {"linear_F": fl, "random_forest": rf_all.predict(X_forest(d)), "gradient_boosting": hgb_all.predict(Xt),
            "xgboost": xg_all.predict(Xt), "lightgbm": lg_all.predict(Xt), "hybrid_F_lightgbm": fl + hy_all.predict(Xt.drop(columns=HYB_DROP)),
            "hybrid_ratio": fl * np.exp(hr_all.predict(Xt.drop(columns=HYB_DROP)))}

# feature importance: share of all the improvement each feature gave the trees (category columns added back together)
NICE = {"floor_area_sqm": "Floor area", "flat_age": "Flat age", "storey": "Storey", "income_k": "Income (last year)",
        "avl4_k": "RFS lag 4 + BTO lag 4", "rfs4": "RFS lag 4 column (2001-04 sales)", "sold4_k": "Flats sold lag 4", "tight36": "Policy stance", "dist_to_cbd_km": "Distance to CBD", "dist_to_mall_km": "Distance to mall",
        "dist_to_nearest_high_demand_primary_school_km": "Distance to popular school",
        "within_1km_high_demand_primary_school_flag": "Within 1 km of popular school", "mrt_open_km": "Distance to open MRT",
        "mrt_new_gain": "Announced MRT closer", "stage_early": "MRT announced (early stage)", "stage_late": "MRT announced (late stage)",
        "town": "Town", "flat_type": "Flat type", "flat_model": "Flat model"}
def group(names, vals):
    s = pd.Series(vals, index=[next((c for c in CAT if n.startswith(c + "_")), n) for n in names]).groupby(level=0).sum()
    return (s / s.sum() * 100).round(1)
imp = pd.DataFrame({
    "random_forest": group(X_forest(hdb.iloc[:5]).columns, rf_all.feature_importances_),
    "lightgbm": group(lg_all.booster_.feature_name(), lg_all.booster_.feature_importance("gain")),
    "xgboost": pd.Series(xg_all.get_booster().get_score(importance_type="total_gain")).pipe(lambda s: (s / s.sum() * 100).round(1)),
}).reindex(NUM + CAT).fillna(0.0)
imp.index = [NICE[c] for c in imp.index]
imp = imp.sort_values("lightgbm", ascending=False)
print("\nFeature importance (% of the total; random forest = impurity, LightGBM / XGBoost = gain):")
print(imp.to_string())

# age curve each model learned: same sample of recent sales, age set to 0, 2, ... 60, average price, minus age 0
ages = list(range(0, 61, 2))
samp = hdb[hdb["year"] == hdb["year"].max()].sample(n=min(3000, (hdb["year"] == hdb["year"].max()).sum()), random_state=SEED)
curve = {k: [] for k in ORDER}
for a in ages:
    p = predict_all(samp.assign(flat_age=float(a)))
    for k in ORDER:
        curve[k].append(float(p[k].mean()))
curve = {k: [round(v - c[0], 1) for v in c] for k, c in curve.items()}
print(f"\nAge curve learned by each model ($k vs a new flat, same flats otherwise; {len(samp):,} sales from {hdb['year'].max()}):")
print(pd.DataFrame(curve, index=ages).iloc[::5].to_string())

# price today for the typical flat of every block and flat type
Ymax, Dmax = int(hdb["year"].max()), hdb["date"].max()
hdb["age_now"] = hdb["flat_age"] + (Ymax - hdb["year"])
g = hdb[hdb["year"] >= 2010].groupby(["bid", "flat_type"])
typ = g.agg(floor_area_sqm=("floor_area_sqm", "median"), storey=("storey", "median"), flat_age=("age_now", "median"),
            flat_model=("flat_model", lambda s: s.mode().iat[0]), sales=("price_k", "size")).reset_index()
typ = typ.merge(blk[["bid", "block", "street_name", "town"] + LOC], on="bid")
typ["flat_age"] = typ["flat_age"].round()
typ["storey"] = (typ["storey"] / 3).round() * 3 - 1                       # middle of a 3-storey band, like the data
typ["storey"] = typ["storey"].clip(lower=2)
typ["income_k"] = macro["income"].get(Ymax - 1, np.nan) / 1000
typ["avl4_k"] = macro["avl"].get(Ymax - 4, np.nan) / 1000
typ["rfs4"] = 0.0
typ["sold4_k"] = macro["sold"].get(Ymax - 4, np.nan) / 1000 if "sold" in macro else np.nan
typ["tight36"] = _st.get(pd.Timestamp(Dmax), float(pol.loc[(pol["announce_date"] < Dmax) & (pol["announce_date"] >= Dmax - pd.DateOffset(months=36)), "w"].sum()))
for c, v in mrt_features(typ["bid"].values, np.datetime64(Dmax)).items():
    typ[c] = v
typ = typ.dropna(subset=NUM)
pt = predict_all(typ)
today = {}
for i, r in enumerate(typ.itertuples(index=False)):
    key = f"{r.block} {r.street_name}"
    today.setdefault(key, {})[r.flat_type] = [round(float(r.floor_area_sqm)), int(r.flat_age), int(r.storey), r.flat_model] + \
        [round(float(pt[k][i])) for k in ORDER]
print(f"\n{clock()} Price today ({Dmax:%Y-%m}) for {len(typ):,} block + flat type pairs in {len(today):,} blocks")

# ---------------------------------------------------------------- charts for slides
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    top = imp.head(12).iloc[::-1]
    yy = np.arange(len(top))
    ax.barh(yy + 0.2, top["lightgbm"], 0.4, label="LightGBM", color="#0d6e8a")
    ax.barh(yy - 0.2, top["random_forest"], 0.4, label="Random forest", color="#a0418f")
    ax.set_yticks(yy, top.index); ax.set_xlabel("% of total importance"); ax.legend(); ax.set_title("What the tree models rely on")
    fig.tight_layout(); fig.savefig(OUT / "tree_importance.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, col in zip(ORDER, ["#14202b", "#a0418f", "#2f8a3e", "#b4462b", "#0d6e8a", "#1f6fd1", "#5b6b78"]):
        ax.plot(ages, curve[k], label=LABEL[k], color=col, lw=3 if k == "linear_F" else 2)
    ax.set_xlabel("Flat age (years)"); ax.set_ylabel("$k vs a new flat"); ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax.set_title("Lease decay: the plane's age bands vs what the trees learned")
    fig.tight_layout(); fig.savefig(OUT / "tree_age_curve.png", dpi=150); plt.close(fig)
    print("Saved hdb_data/tree_importance.png and hdb_data/tree_age_curve.png")
except Exception as e:                                       # charts are optional
    print("Charts skipped:", e)

# ---------------------------------------------------------------- hdb_trees.js for the page
out = {"order": ORDER, "labels": LABEL,
       "forecast": {"train": f"{TRAIN_FROM}-{TEST_FROM - 1}", "test": f"{TEST_FROM}-{Ymax}", "res": fc},
       "valuation": {"desc": f"random 80% / 20% of {TRAIN_FROM}-{Ymax} sales", "res": va},
       "importance": {"features": list(imp.index), "random_forest": imp["random_forest"].tolist(),
                      "lightgbm": imp["lightgbm"].tolist(), "xgboost": imp["xgboost"].tolist()},
       "age": {"ages": ages, "curves": curve, "year": Ymax},
       "today": {"month": f"{Dmax:%Y-%m}", "cols": ["area", "age", "storey", "flat_model"] + ORDER, "blocks": today},
       "rf_rows": RF_ROWS}
Path("hdb_trees.js").write_text("window.HDB_TREES = " + json.dumps(out, separators=(",", ":")) + ";\n")
print(f"{clock()} Saved hdb_trees.js ({Path('hdb_trees.js').stat().st_size / 1e6:.1f} MB) and hdb_data/tree_backtest.csv")
