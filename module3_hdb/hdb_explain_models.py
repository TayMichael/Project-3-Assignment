"""Look inside the models: one real flat, followed through every model, and where the models miss.

Part 1 - one real sale (the latest 4 ROOM sale of Blk 267 Tampines St 21 in 2020-2026), priced by
every model trained on 2001-2019 only (the sealed-test setup), with what each model did inside:
  plane S            each term of the formula: coefficient x the flat's value
  random forest      the questions tree 1 asked, its leaf, and all 150 trees' answers (their average = the price)
  gradient boosting  the running total after 1, 10, 100 ... trees
  XGBoost            the questions of trees 1-3, their leaf values, the running total, contributions per feature
  LightGBM           the same, read from LightGBM's own tree dump
  hybrid $ (tuned)   plane S + LightGBM (127 leaves) correction: the same walkthrough for the correction
  demo tree          a small LightGBM (8 leaves) on plane S's errors: its whole first tree, for drawing,
                     with the average error of the training sales in every leaf
Part 2 - sealed-test error (2020-2026) of plane S and the tuned hybrid by flat type, town, age, price band,
storey and year (assignment point 8: where does the model fail, and for which segments?)

Data preparation copied unchanged from hdb_tree_models.py.
Writes hdb_data/explain_models.json (send this file back for the guide).
    conda activate dsai-m3
    python hdb_explain_models.py        (about 10-15 minutes)
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

# ================================================================ look inside the models
EX_BLOCK, EX_STREET, EX_TYPE = "267", "TAMPINES ST 21", "4 ROOM"
HYB_TUNED = LGB_P | dict(num_leaves=127)               # the setting chosen by hdb_hybrid_cv.py --grid
CHECKS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 8000]
R = {}                                                 # everything saved to explain_models.json

is_tr = (hdb["year"] < TEST_FROM).values
dtr, dte = hdb[is_tr], hdb[~is_tr]
ytr, yte = dtr["price_k"].values, dte["price_k"].values
cand = dte[(dte["block"] == EX_BLOCK) & (dte["street_name"] == EX_STREET) & (dte["flat_type"] == EX_TYPE)]
if cand.empty:
    cand = dte[(dte["town"] == "TAMPINES") & (dte["flat_type"] == EX_TYPE)]
ex = cand.sort_values("date").iloc[[-1]]
R["flat"] = {"block": ex["block"].iat[0], "street": ex["street_name"].iat[0], "town": ex["town"].iat[0],
             "flat_type": ex["flat_type"].iat[0], "flat_model": ex["flat_model"].iat[0], "month": f"{ex['date'].iat[0]:%Y-%m}",
             "actual_k": float(ex["price_k"].iat[0]), "features": {c: float(ex[c].iat[0]) for c in NUM}}
print(f"{clock()} Example: Blk {R['flat']['block']} {R['flat']['street']}, {R['flat']['flat_type']}, sold {R['flat']['month']} "
      f"for ${R['flat']['actual_k']:.0f}k; models trained on {TRAIN_FROM}-{TEST_FROM - 1}")
pred = {}

# ---------------------------------------------------------------- plane S: the formula, term by term
Xl = X_linear(dtr); cols = list(Xl.columns)
F = LinearRegression().fit(Xl, ytr)
f_tr, f_te = F.predict(Xl), F.predict(X_linear(dte, cols))
del Xl; gc.collect()
xe = X_linear(ex, cols).iloc[0]
terms = [["start (beta)", float(F.intercept_), 1.0, float(F.intercept_)]]
for c, w in zip(cols, F.coef_):
    if xe[c] != 0:
        terms.append([c, float(w), float(xe[c]), float(w * xe[c])])
pred["plane_S"] = float(F.predict(X_linear(ex, cols))[0])
R["plane"] = {"terms": terms, "price": pred["plane_S"]}
print(f"{clock()} plane S: ${pred['plane_S']:.1f}k ({len(terms)} non-zero terms)")

# ---------------------------------------------------------------- random forest: tree 1's questions + all 150 answers
rng = np.random.default_rng(SEED)
sub = rng.choice(len(dtr), size=min(RF_ROWS, len(dtr)), replace=False)
Xf = X_forest(dtr.iloc[sub]); fcols = list(Xf.columns)
rf = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=5, max_features=0.4,
                           bootstrap=True, n_jobs=-1, random_state=SEED).fit(Xf, ytr[sub])
del Xf; gc.collect()
xf = X_forest(ex).reindex(columns=fcols, fill_value=0).to_numpy(dtype="float32")
t = rf.estimators_[0].tree_
node, path = 0, []
while t.children_left[node] != -1:
    f_, thr, v = fcols[t.feature[node]], float(t.threshold[node]), float(xf[0, t.feature[node]])
    left = v <= thr
    path.append({"feature": f_, "op": "<=", "threshold": thr, "value": v, "yes": bool(left),
                 "sales": int(t.n_node_samples[node])})
    node = t.children_left[node] if left else t.children_right[node]
each = [float(e.predict(xf)[0]) for e in rf.estimators_]
pred["random_forest"] = float(rf.predict(pd.DataFrame(xf, columns=fcols))[0])
R["random_forest"] = {"tree1_path": path, "tree1_leaf": float(t.value[node][0][0]), "tree1_leaf_sales": int(t.n_node_samples[node]),
                      "all_trees": each, "price": pred["random_forest"], "rows": int(len(sub))}
print(f"{clock()} random forest: tree 1 asked {len(path)} questions, leaf ${t.value[node][0][0]:.1f}k; "
      f"150 trees from ${min(each):.0f}k to ${max(each):.0f}k, average ${pred['random_forest']:.1f}k")
del rf; gc.collect()

Xt, Xte, Xex = X_tree(dtr), X_tree(dte), X_tree(ex)
fit_, val_ = early_split(len(dtr))

def running(contribs):
    """running total after the first k trees, at the CHECKS points"""
    cs = np.cumsum(contribs)
    return [[k, float(cs[k - 1])] for k in CHECKS if k <= len(cs)] + ([[len(cs), float(cs[-1])]] if len(cs) not in CHECKS else [])

# ---------------------------------------------------------------- gradient boosting: running total
hgb = HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
                                    validation_fraction=0.1, n_iter_no_change=30, categorical_features="from_dtype",
                                    random_state=SEED).fit(Xt, ytr)
stages = np.array([float(p[0]) for p in hgb.staged_predict(Xex)])
pred["gradient_boosting"] = float(stages[-1])
R["gradient_boosting"] = {"start": float(stages[0] - (stages[1] - stages[0]) if len(stages) > 1 else stages[0]),
                          "running": [[k, float(stages[k - 1])] for k in CHECKS if k <= len(stages)] + [[len(stages), float(stages[-1])]],
                          "trees": int(hgb.n_iter_), "price": pred["gradient_boosting"]}
print(f"{clock()} gradient boosting: {hgb.n_iter_} trees, ${pred['gradient_boosting']:.1f}k")
del hgb; gc.collect()

# ---------------------------------------------------------------- LightGBM walkthrough (used for plain LightGBM, the hybrid and the demo)
def code_of(X):
    """category columns as LightGBM sees them: their category codes"""
    return {c: int(X[c].cat.codes.iat[0]) for c in CAT}

def lgb_paths(model, X, n_paths=3, full=0):
    b = model.booster_
    names = b.feature_name()
    dump = b.dump_model()["tree_info"]
    leaves = b.predict(X, pred_leaf=True, num_iteration=model.best_iteration_ or None)[0]
    codes = code_of(X)
    vals = {c: (codes[c] if c in CAT else float(X[c].iat[0])) for c in names}

    def walk(nd, target, acc):
        if "leaf_index" in nd:
            return acc + [{"leaf": int(nd["leaf_index"]), "value": float(nd["leaf_value"]), "sales": int(nd.get("leaf_count", 0))}] \
                if nd["leaf_index"] == target else None
        f_ = names[nd["split_feature"]]; v = vals[f_]
        if nd["decision_type"] == "==":
            cats = [int(c) for c in str(nd["threshold"]).split("||")]
            left = v in cats
            q = {"feature": f_, "op": "in", "threshold": [CATS[f_][c] for c in cats if c < len(CATS[f_])],
                 "value": CATS[f_][v], "yes": bool(left)}
        else:
            left = v <= nd["threshold"]
            q = {"feature": f_, "op": "<=", "threshold": float(nd["threshold"]), "value": v, "yes": bool(left)}
        q["sales"] = int(nd.get("internal_count", 0))
        nxt = nd["left_child"] if left else nd["right_child"]
        return walk(nxt, target, acc + [q])

    used = len(leaves)
    contribs, paths, ok = [], [], True
    for i in range(used):
        p = walk(dump[i]["tree_structure"], int(leaves[i]), [])
        if p is None:
            ok = False; p = [{"leaf": int(leaves[i]), "value": float("nan"), "sales": 0}]
        contribs.append(p[-1]["value"])
        if i < n_paths:
            paths.append(p)
    raw = float(model.predict(X)[0])
    contrib = b.predict(X, pred_contrib=True, num_iteration=model.best_iteration_ or None)[0]
    by_feat = sorted([[n, float(c)] for n, c in zip(names, contrib[:-1])], key=lambda r: -abs(r[1]))
    asked = pd.Series([q["feature"] for i in range(used) for q in (walk(dump[i]["tree_structure"], int(leaves[i]), []) or [])[:-1]])
    out = {"trees": used, "paths": paths, "running": running(contribs), "sum_of_leaves": float(np.sum(contribs)),
           "predict": raw, "walk_matches_predict": bool(ok and abs(np.sum(contribs) - raw) < 1e-3),
           "contrib_base": float(contrib[-1]), "contrib": by_feat, "questions_by_feature": asked.value_counts().to_dict()}
    if full:
        def strip(nd):
            if "leaf_index" in nd:
                return {"leaf": int(nd["leaf_index"]), "value": float(nd["leaf_value"]), "sales": int(nd.get("leaf_count", 0))}
            f_ = names[nd["split_feature"]]
            thr = ([CATS[f_][int(c)] for c in str(nd["threshold"]).split("||")] if nd["decision_type"] == "==" else float(nd["threshold"]))
            return {"feature": f_, "op": "in" if nd["decision_type"] == "==" else "<=", "threshold": thr,
                    "sales": int(nd.get("internal_count", 0)), "yes": strip(nd["left_child"]), "no": strip(nd["right_child"])}
        out["full_trees"] = [strip(dump[i]["tree_structure"]) for i in range(min(full, used))]
    return out

lg = lgb.LGBMRegressor(**LGB_P).fit(Xt[fit_], ytr[fit_], eval_set=[(Xt[val_], ytr[val_])],
                                    callbacks=[lgb.early_stopping(50, verbose=False)])
R["lightgbm"] = lgb_paths(lg, Xex); pred["lightgbm"] = R["lightgbm"]["predict"]
print(f"{clock()} LightGBM: {R['lightgbm']['trees']} trees, ${pred['lightgbm']:.1f}k (walk matches: {R['lightgbm']['walk_matches_predict']})")
del lg; gc.collect()

# ---------------------------------------------------------------- XGBoost walkthrough (from its tree table)
try:
    xg = xgb.XGBRegressor(**XGB_P).fit(Xt[fit_], ytr[fit_], eval_set=[(Xt[val_], ytr[val_])], verbose=False)
    bst = xg.get_booster(); n_used = int(xg.best_iteration) + 1
    dm = xgb.DMatrix(Xex, enable_categorical=True)
    leaves = bst.predict(dm, pred_leaf=True, iteration_range=(0, n_used))[0].astype(int)
    df = bst.trees_to_dataframe()
    base = float(json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]"))
    codes = code_of(Xex)
    parent = {}
    for r in df[df["Feature"] != "Leaf"].itertuples():
        parent[r.Yes] = (r.ID, True); parent[r.No] = (r.ID, False)
    rows = df.set_index("ID")
    contribs, paths = [], []
    for i in range(n_used):
        lid = f"{i}-{leaves[i]}"
        contribs.append(float(rows.loc[lid, "Gain"]))           # a leaf's "Gain" column holds its value
        if i < 3:
            p, cur = [], lid
            while cur in parent:
                pid, yes = parent[cur]; r = rows.loc[pid]; f_ = r["Feature"]
                if f_ in CAT:
                    cats = [int(c) for c in (r["Category"] if "Category" in rows.columns and isinstance(r["Category"], (list, np.ndarray)) else [])]
                    p.append({"feature": f_, "op": "in", "threshold": [CATS[f_][c] for c in cats if c < len(CATS[f_])],
                              "value": CATS[f_][codes[f_]], "yes": bool(yes), "sales": float(r["Cover"])})
                else:
                    p.append({"feature": f_, "op": "<", "threshold": float(r["Split"]), "value": float(Xex[f_].iat[0]),
                              "yes": bool(yes), "sales": float(r["Cover"])})
                cur = pid
            paths.append(p[::-1] + [{"leaf": int(leaves[i]), "value": contribs[-1], "sales": float(rows.loc[lid, "Cover"])}])
    contrib = bst.predict(dm, pred_contribs=True, iteration_range=(0, n_used))[0]
    pred["xgboost"] = float(xg.predict(Xex)[0])
    R["xgboost"] = {"trees": n_used, "base_score": base, "paths": paths, "running": running(contribs),
                    "sum_of_leaves": float(base + np.sum(contribs)), "predict": pred["xgboost"],
                    "contrib_base": float(contrib[-1]),
                    "contrib": sorted([[n, float(c)] for n, c in zip(bst.feature_names, contrib[:-1])], key=lambda r: -abs(r[1]))}
    print(f"{clock()} XGBoost: {n_used} trees, ${pred['xgboost']:.1f}k (base {base:.1f} + leaves = {R['xgboost']['sum_of_leaves']:.1f})")
    del xg; gc.collect()
except Exception as e:                                   # the XGBoost part is optional
    print(f"{clock()} XGBoost walkthrough skipped: {e}")
    R["xgboost"] = {"error": str(e)}

# ---------------------------------------------------------------- the tuned hybrid: plane S + LightGBM correction
Xh, Xhte, Xhex = Xt.drop(columns=HYB_DROP), Xte.drop(columns=HYB_DROP), Xex.drop(columns=HYB_DROP)
r_tr = ytr - f_tr
hy = lgb.LGBMRegressor(**HYB_TUNED).fit(Xh[fit_], r_tr[fit_], eval_set=[(Xh[val_], r_tr[val_])],
                                        callbacks=[lgb.early_stopping(50, verbose=False)])
R["hybrid"] = lgb_paths(hy, Xhex)
pred["hybrid"] = pred["plane_S"] + R["hybrid"]["predict"]
R["hybrid"]["plane"] = pred["plane_S"]; R["hybrid"]["price"] = pred["hybrid"]
h_te = f_te + hy.predict(Xhte)
print(f"{clock()} hybrid: plane ${pred['plane_S']:.1f}k + correction ${R['hybrid']['predict']:+.1f}k from {R['hybrid']['trees']} trees "
      f"= ${pred['hybrid']:.1f}k (sealed-test MAE {np.abs(h_te - yte).mean():.1f}k)")
del hy; gc.collect()

# ---------------------------------------------------------------- demo: a small LightGBM on the plane's errors, whole tree for drawing
demo = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.1, num_leaves=8, min_child_samples=40, random_state=SEED, verbose=-1).fit(Xh, r_tr)
R["demo"] = lgb_paths(demo, Xhex, n_paths=2, full=2)
leaf_tr = demo.booster_.predict(Xh, pred_leaf=True)[:, 0]
R["demo"]["tree1_leaf_avg_error"] = {int(k): [float(v), int(n)] for k, v, n in
                                    pd.DataFrame({"l": leaf_tr, "r": r_tr}).groupby("l")["r"].agg(["mean", "size"]).reset_index().itertuples(index=False)}
R["demo"]["plane_train_mae"] = float(np.abs(r_tr).mean())
R["demo"]["after_50_trees_train_mae"] = float(np.abs(r_tr - demo.predict(Xh)).mean())
print(f"{clock()} demo tree: 8 leaves; plane's training error {R['demo']['plane_train_mae']:.1f}k -> "
      f"{R['demo']['after_50_trees_train_mae']:.1f}k after 50 small trees")
R["pred"] = pred

# ================================================================ Part 2: where the models miss (sealed test 2020-2026)
seg = dte[["town", "flat_type", "flat_age", "storey", "year"]].copy()
seg["e_plane"], seg["e_hybrid"] = f_te - yte, h_te - yte
seg["age band"] = pd.cut(seg["flat_age"], [0, 10, 20, 30, 40, 50, 70], right=False, labels=["0-9", "10-19", "20-29", "30-39", "40-49", "50+"]).astype(str)
seg["price band"] = pd.cut(yte, [0, 400, 500, 600, 800, 1000, 9e9], labels=["< $400k", "$400-500k", "$500-600k", "$600-800k", "$800k-1m", "$1m +"]).astype(str)
seg["storey band"] = pd.cut(seg["storey"], [0, 6, 12, 20, 99], labels=["1-6", "7-12", "13-20", "21 +"]).astype(str)
R["segments"] = {}
for g in ["flat_type", "age band", "price band", "storey band", "year", "town"]:
    t_ = seg.groupby(g).agg(n=("e_plane", "size"), plane_mae=("e_plane", lambda e: e.abs().mean()), plane_bias=("e_plane", "mean"),
                            hybrid_mae=("e_hybrid", lambda e: e.abs().mean()), hybrid_bias=("e_hybrid", "mean")).round(1)
    t_ = t_[t_["n"] >= 200]
    R["segments"][g] = [[str(k)] + [float(x) for x in r] for k, r in zip(t_.index, t_.values)]
    print(f"\nSealed test by {g} ($k):\n{t_.to_string()}")

def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    return o
(OUT / "explain_models.json").write_text(json.dumps(clean(R)))
print(f"\n{clock()} Prices for the example flat ($k): " + ", ".join(f"{k} {v:.1f}" for k, v in pred.items())
      + f"; actual {R['flat']['actual_k']:.1f}")
print(f"{clock()} saved hdb_data/explain_models.json")
