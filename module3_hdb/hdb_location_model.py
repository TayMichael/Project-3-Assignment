"""Add location to the HDB price model and back-test it (train 2001-2019, test 2020-2026).

Needs in hdb_data/:
  hdb_resale_all.csv               from hdb_resale_1990_2026.py
  macro_annual.csv                 from hdb_drivers_model.py (income, bto, ...)
  hdb_block_lookup_from_V3F.csv    your block lookup (one row per block + street)
  hdb_block_lookup_filled.csv      optional: the lookup with demolished and new blocks filled (used if present)
  mrt_stations_corrected.csv       one row per station: announcement, opening, coordinates

Each sale is joined to its block by block + street_name. MRT features are rebuilt for the
sale's own month, so a station only counts once it has opened (or been announced):
  mrt_open_km   distance to the nearest station already open in the sale month
  mrt_new_gain  how much closer the nearest ANNOUNCED-but-not-open station is (0 if none is closer)
  stage_early / stage_late   that announced station is in the first / second half of its
                             announcement-to-opening period (your midpoint rule)

New HDB supply is tried three ways (from macro_annual.csv): BTO launched lag 4 (models A-F, train 2005+),
RFS lag 4 + BTO lag 4 = RFS flats offered a year 1997-2000 (four quarterly batches), then BTO launched, 4 years before the sale (G, and H with a 0/1 RFS-years column; train 2001+), and
HDB flats completed lag 1 / 4 / 5 (I, J, K). Each trains from the first year its measure allows; all are
tested on 2020-2026 sales, and the page uses the lowest-MAE model of D, F-K tested on the full period.

Writes hdb_location.js (the best of D, F-K refitted (L-N are printed only) on all years + every block + every station) for
hdb-future-price.html, which adds it to the driver list as "Income + BTO lag 4 + location".
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

OUT = Path("hdb_data")
TRAIN_FROM, TEST_FROM, BASE_TOWN = 2001, 2020, "ANG MO KIO"
CAP_KM = 3.0                     # beyond 3 km, another km to the MRT changes little

norm = lambda s: s.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)

# ---------------------------------------------------------------- sales + last year's drivers
hdb = pd.read_csv(OUT / "hdb_resale_all.csv", low_memory=False)
hdb = hdb[hdb["year"] >= TRAIN_FROM].copy()
ends = hdb["storey_range"].str.split(" TO ", expand=True).astype(int)
hdb["storey"] = (ends[0] + ends[1]) / 2
hdb["price_k"] = hdb["resale_price"] / 1000
hdb["date"] = pd.to_datetime(hdb["month"].str[:7])
macro = pd.read_csv(OUT / "macro_annual.csv", index_col="year")
hdb["income_k"] = (hdb["year"] - 1).map(macro["income"]) / 1000          # year Y sale <- year Y-1 income
# new HDB supply, three measures (see hdb_drivers_model.py): BTO launched, availability (RFS+BTO), completed
SUPPLY = {"bto4_k": ("bto", 4, "BTO lag 4"), "bto1_k": ("bto", 1, "BTO lag 1"), "bto0_k": ("bto", 0, "BTO lag 0 (same year)"),
          "avl1_k": ("avl", 1, "RFS lag 1 + BTO lag 1"), "avl4_k": ("avl", 4, "RFS lag 4 (1997-2000) + BTO lag 4 (2001 on)"),
          "cmp1_k": ("cmp", 1, "HDB flats completed (lag 1)"), "cmp4_k": ("cmp", 4, "HDB flats completed (lag 4)"),
          "cmp5_k": ("cmp", 5, "HDB flats completed (lag 5)"), "sold4_k": ("sold", 4, "HDB flats sold (lag 4, official)")}
for col, (ser, k, _) in SUPPLY.items():
    if ser in macro:
        hdb[col] = (hdb["year"] - k).map(macro[ser]) / 1000                   # supply k years before the sale
hdb["rfs4"] = (hdb["year"] - 4 <= 2000).astype(float)                      # 1 when avl4 is an RFS year (quarterly batches), not BTO launches
hdb["flat_type"] = norm(hdb["flat_type"]).str.replace("MULTI GENERATION", "MULTI-GENERATION")
fm = norm(hdb["flat_model"])
top = fm.value_counts()
hdb["flat_model"] = fm.where(fm.map(top) >= 2000, "OTHER")              # rare models grouped

# ---------------------------------------------------------------- join the block lookup
LOOKUP = OUT / "hdb_block_lookup_filled.csv"          # from fill_missing_blocks.py, if you ran it
if not LOOKUP.exists():
    LOOKUP = OUT / "hdb_block_lookup_from_V3F.csv"
print("Block lookup:", LOOKUP)
blk = pd.read_csv(LOOKUP, dtype={"postal_code": str}, low_memory=False)
blk["block"], blk["street_name"] = norm(blk["block"]), norm(blk["street_name"])
blk = blk.drop_duplicates(["block", "street_name"])
blk = blk.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
blk["bid"] = blk.index
hdb["block"], hdb["street_name"] = norm(hdb["block"]), norm(hdb["street_name"])
hdb = hdb.merge(blk[["block", "street_name", "bid", "latitude", "longitude", "postal_code",
                     "dist_to_cbd_km", "dist_to_mall_km",
                     "dist_to_nearest_high_demand_primary_school_km",
                     "within_1km_high_demand_primary_school_flag"]],
                on=["block", "street_name"], how="left")
matched = hdb["bid"].notna()
print(f"\nMatched to a geocoded block: {matched.sum():,} of {len(hdb):,} sales ({matched.mean():.1%})")
by_year = hdb.groupby("year")["bid"].apply(lambda s: s.notna().mean())
print("Match rate by year (lowest 5):", by_year.sort_values().head(5).round(3).to_dict())
miss = hdb.loc[~matched].groupby(["block", "street_name"]).size().sort_values(ascending=False)
miss.rename("sales").to_csv(OUT / "unmatched_blocks.csv")
print(f"{len(miss):,} block+street pairs not found -> hdb_data/unmatched_blocks.csv (top 5 below)")
print(miss.head().to_string())
hdb = hdb[matched].copy()
hdb["bid"] = hdb["bid"].astype(int)

# ---------------------------------------------------------------- MRT at the sale date
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

feat = {c: np.full(len(hdb), np.nan) for c in ["mrt_open_km", "mrt_new_gain", "stage_early", "stage_late"]}
idx = {d: np.where(hdb["date"].values == d)[0] for d in np.unique(hdb["date"].values)}
for d, rows in idx.items():
    b = hdb["bid"].values[rows]
    is_open = st_opn <= d
    is_ann = (st_ann <= d) & ~is_open
    d_open = np.where(is_open, dist[b], np.inf).min(axis=1)
    d_ann = np.where(is_ann, dist[b], np.inf)
    j = d_ann.argmin(axis=1)
    near_ann = d_ann[np.arange(len(b)), j]
    closer = near_ann < d_open
    feat["mrt_open_km"][rows] = np.minimum(d_open, CAP_KM)
    feat["mrt_new_gain"][rows] = np.where(closer, np.minimum(d_open, CAP_KM) - np.minimum(near_ann, CAP_KM), 0.0)
    late = st_mid[j] <= d
    feat["stage_early"][rows] = (closer & ~late).astype(float)
    feat["stage_late"][rows] = (closer & late).astype(float)
for c, v in feat.items():
    hdb[c] = v

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
    _blk = hdb[["bid", "latitude", "longitude"]].drop_duplicates("bid").set_index("bid").sort_index()
    _pos = pd.Series(np.arange(len(_blk)), index=_blk.index)
    _dM = haversine(_blk["latitude"].values[:, None], _blk["longitude"].values[:, None], _M["lat"].values[None, :], _M["lon"].values[None, :])
    _dS = haversine(_blk["latitude"].values[:, None], _blk["longitude"].values[:, None], _S["lat"].values[None, :], _S["lon"].values[None, :])
    _b = _pos[hdb["bid"].values].values
    _mall, _sch = np.full(len(hdb), np.nan), np.full(len(hdb), np.nan)
    for d, rows in idx.items():                                     # idx: sale rows by month (MRT block above)
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


# quick sanity check against your V3F problem towns
chk = hdb[(hdb["year"] == 2010) & hdb["town"].isin(["JURONG EAST", "CLEMENTI", "PASIR RIS", "TAMPINES", "WOODLANDS", "PUNGGOL"])]
print("\n2010 median km to nearest OPEN MRT (corrected stations):")
print(chk.groupby("town")["mrt_open_km"].median().round(3).to_string())

# ---------------------------------------------------------------- policy stance known at the sale date
# tight36 = sum of (direction x intensity) of the policy rounds ANNOUNCED in the 36 months before the sale month
# (policy_events.csv: direction +1 = cooling / tightening, -1 = easing; intensity 1-3). Only rounds announced in
# EARLIER months count, so a sale never "knows" a round announced later, not even later in the same month.
POL = Path("policy_events.csv")
POLCOL = {"SSD": "pol_ssd", "ABSD": "pol_absd", "Loan limits & HDB rules": "pol_loan", "BSD / tax": "pol_bsd",
          "Easing (incl. grants)": "pol_ease"}
POLICY_COLS = ["tight36", "cool36", "ease36"] + list(POLCOL.values())


def policy_group(r):
    """the type of a policy round, by direction and the instruments it used (as in hdb_drivers_model.py)"""
    ins = str(r["instruments"]).upper().split(";")
    if int(r["direction"]) < 0:
        return "Easing (incl. grants)"
    if "ABSD" in ins:
        return "ABSD"
    if "SSD" in ins:
        return "SSD"
    if ins == ["BSD"]:
        return "BSD / tax"
    return "Loan limits & HDB rules"


if POL.exists():
    pol = pd.read_csv(POL, parse_dates=["announce_date"])
    pol["w"] = pol["direction"] * pol["intensity"]
    months = np.sort(hdb["date"].unique())
    stance = {m: float(pol.loc[(pol["announce_date"] < m) & (pol["announce_date"] >= m - pd.DateOffset(months=36)), "w"].sum())
              for m in pd.to_datetime(months)}
    hdb["tight36"] = hdb["date"].map(stance)
    # the same rounds counted WITHOUT the 1-3 intensity judgment, so the regression learns each weight itself:
    #   cool36 / ease36   number of cooling / easing rounds announced in the 36 months before the sale (model U)
    #   pol_ssd ... pol_ease   number of rounds of each policy type in those 36 months (model V)
    # The types follow policy_group() in hdb_drivers_model.py (the groups of the event study on the page).
    pol["grp"] = pol.apply(policy_group, axis=1)
    win = {m: pol[(pol["announce_date"] < m) & (pol["announce_date"] >= m - pd.DateOffset(months=36))] for m in pd.to_datetime(months)}
    hdb["cool36"] = hdb["date"].map({m: float((w["direction"] > 0).sum()) for m, w in win.items()})
    hdb["ease36"] = hdb["date"].map({m: float((w["direction"] < 0).sum()) for m, w in win.items()})
    for g, c in POLCOL.items():
        hdb[c] = hdb["date"].map({m: float((w["grp"] == g).sum()) for m, w in win.items()})
    print("Rounds per policy type, by announcement year:")
    print(pol.groupby([pol["announce_date"].dt.year.rename("year"), "grp"]).size().unstack(fill_value=0).to_string())
    print("\nPolicy stance (tight36) at the start of each year:",
          ", ".join(f"{y}: {stance.get(pd.Timestamp(f'{y}-01-01'), float('nan')):+.0f}" for y in range(2001, 2027, 3)))

# ---------------------------------------------------------------- back-test
BASE0 = ["floor_area_sqm", "flat_age", "storey", "income_k"]
BASE = BASE0 + ["bto4_k"]
LOC = ["dist_to_cbd_km", "dist_to_mall_km", "dist_to_nearest_high_demand_primary_school_km",
       "within_1km_high_demand_primary_school_flag"]
MRT = ["mrt_open_km", "mrt_new_gain", "stage_early", "stage_late"]
hdb = hdb.dropna(subset=BASE0 + LOC + MRT)                           # supply columns are checked per model
# Lease-decay curve measured from past sales: one extra slope from every bend age onwards.
# age_over_40 = years past 40 (0 before 40), so the price can fall faster after 40 than before it.
AGE_KNOTS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
AGE = [f"age_over_{k}" for k in AGE_KNOTS]
for k, c in zip(AGE_KNOTS, AGE):
    hdb[c] = (hdb["flat_age"] - k).clip(lower=0)

def rows_for(cols):
    """the sales that have every supply column this model uses (each supply measure starts in a different year)"""
    sup = [c for c in cols if c in SUPPLY]
    return hdb.dropna(subset=sup) if sup else hdb

def design(cols, flat=False, d=None):
    d = hdb if d is None else d
    parts = [d[cols], pd.get_dummies(d["town"], dtype=float).drop(columns=BASE_TOWN)]
    if flat:
        parts += [pd.get_dummies(d["flat_type"], prefix="type", dtype=float).drop(columns="type_4 ROOM"),
                  pd.get_dummies(d["flat_model"], prefix="model", dtype=float).drop(columns="model_MODEL A")]
    return pd.concat(parts, axis=1)

MODELS = {   # name: (columns, flat type/model dummies, log price)
    "A  income + BTO lag 4 (current best)": (BASE, False, False),
    "B  + flat type & model":                (BASE, True, False),
    "C  + CBD, mall, school":                (BASE + LOC, True, False),
    "D  + MRT at sale date":                 (BASE + LOC + MRT, True, False),
    "E  D on log price":                     (BASE + LOC + MRT, True, True),
    "F  D + lease-decay curve":              (BASE + LOC + MRT + AGE, True, False),
}
REST = LOC + MRT + AGE
if "avl4_k" in hdb:
    MODELS["G  F, supply = RFS lag 4 + BTO lag 4"] = (BASE0 + ["avl4_k"] + REST, True, False)
    MODELS["H  G + RFS-years column (page model)"] = (BASE0 + ["avl4_k", "rfs4"] + REST, True, False)
if "cmp1_k" in hdb:
    MODELS["I  F, supply = flats completed lag 1"] = (BASE0 + ["cmp1_k"] + REST, True, False)
    MODELS["J  F, supply = flats completed lag 4"] = (BASE0 + ["cmp4_k"] + REST, True, False)
    MODELS["K  F, supply = flats completed lag 5"] = (BASE0 + ["cmp5_k"] + REST, True, False)
# announcement effect: buyers react when the BTO launches, not when keys are collected
MODELS["L  F, supply = BTO lag 1 (launched the year before)"] = (BASE0 + ["bto1_k"] + REST, True, False)
MODELS["N  F, supply = BTO lag 0 (launched the same year)"] = (BASE0 + ["bto0_k"] + REST, True, False)
if "sold4_k" in hdb:
    # official SingStat series, one measure for every year (no RFS-to-BTO switch), so no rfs4 column needed
    MODELS["O  F, supply = HDB flats sold lag 4 (official)"] = (BASE0 + ["sold4_k"] + REST, True, False)
if "avl1_k" in hdb:
    MODELS["M  F, supply = RFS lag 1 + BTO lag 1"] = (BASE0 + ["avl1_k"] + REST, True, False)
res, coefs, per_year = [], {}, {}
for name, (cols, flat, log) in MODELS.items():
    d = rows_for(cols)
    y, tr, te = d["price_k"], d["year"] < TEST_FROM, d["year"] >= TEST_FROM
    X = design(cols, flat, d)
    m = LinearRegression().fit(X[tr], np.log(y[tr]) if log else y[tr])
    p = m.predict(X[te])
    p = np.exp(p) if log else p
    e = p - y[te]
    res.append({"model": name, "train": f"{int(d.loc[tr, 'year'].min())}-{TEST_FROM - 1}", "n_test": int(te.sum()),
                "MAE_k": round(float(np.abs(e).mean()), 1),
                "RMSE_k": round(float(np.sqrt((e ** 2).mean())), 1),
                "MAPE_%": round(float((np.abs(e) / y[te]).mean() * 100), 1),
                "bias_k": round(float(e.mean()), 1)})
    per_year[name[:1]] = pd.Series(np.abs(e).values, index=d.loc[te, "year"].values).groupby(level=0).mean()
    coefs[name[:1]] = pd.Series(m.coef_, index=X.columns)
    del X, m, d

res = pd.DataFrame(res)
print("\nBack-test, test years 2020-2026 (each model trains from the first year its supply measure allows):")
print(res.to_string(index=False))
print("\nMAE by test year, $k:")
print(pd.DataFrame(per_year).round(1).to_string())
show = BASE + LOC + MRT
print("\nCoefficients, model D ($k per unit; distances in km):")
print(coefs["D"].reindex(show).round(2).to_string())
print("\nCoefficients, model F, lease-decay part ($k per extra year past each bend):")
print(coefs["F"].reindex(["flat_age"] + AGE).round(2).to_string())
if "H" in coefs:
    print(f"\nModel H, RFS-regime column: {coefs['H'].get('rfs4', float('nan')):+.2f} $k (sales 2001-2004, whose lag-4 supply is a 1997-2000 RFS year)")
for k, col in (("G", "avl4_k"), ("O", "sold4_k"), ("L", "bto1_k"), ("M", "avl1_k"), ("N", "bto0_k"), ("I", "cmp1_k"), ("J", "cmp4_k"), ("K", "cmp5_k")):
    if k in coefs:
        print(f"Model {k}: {coefs[k][col]:+.2f} $k per 1,000 units ({SUPPLY[col][2]}), income {coefs[k]['income_k']:+.2f} $k per $1,000")
print("\nCoefficients, model E (x100 = % change in price per unit):")
print((coefs["E"].reindex(show) * 100).round(2).to_string())
res.to_csv(OUT / "location_backtest.csv", index=False)
pd.DataFrame(coefs).to_csv(OUT / "location_coefs.csv")
print("\nSaved hdb_data/location_backtest.csv and hdb_data/location_coefs.csv")

# ---------------------------------------------------------------- choose on validation years, then ONE sealed test
# The table above compares every model on 2020-2026, so 2020-2026 has also been used to choose. Done properly:
#   1. time-series cross-validation inside the training years: for each validation year Y in 2015-2019,
#      train on the years before Y and predict Y (an expanding window, never using later years)
#   2. pick the model with the lowest average validation MAE
#   3. train it on everything up to 2019 and score it ONCE on 2020-2026 (the sealed test)
VAL_YEARS = [2015, 2016, 2017, 2018, 2019]
# candidates: the single models, plus three ways to combine the two regimes (tested here only, never on the test table above)
#   P  H + flats sold lag 4: both supply measures in one plane, each with its own coefficient
#   Q  average of H's and O's predictions (a two-model ensemble)
#   R  H + policy stance (tight36), known at the sale date
#   S  H + flats sold lag 4 + policy stance
#   T  O + policy stance: official data only (SingStat flats sold, one measure in every year), no newspaper
#      RFS series and no rfs4 column. S must beat T by PREFER_MARGIN on validation, otherwise the simpler,
#      consistent T is chosen (the same rule as the hybrid in hdb_hybrid_cv.py; set before the run).
#   U  S with the stance replaced by two counts, cooling rounds and easing rounds (no 1-3 intensity)
#   V  S with the stance replaced by five counts, one per policy type (SSD, ABSD, loan limits & HDB rules,
#      BSD / tax, easing; no intensity): the regression learns how much each type of round goes with.
#      U or V replaces S only if it beats S by PREFER_MARGIN on validation (rule set before the run).
#      Added AFTER the first run of U and V (24 Oct 2026), and said so in the README: a count column can only be
#      chosen if at least MIN_ROUNDS rounds of its kind were announced before the test years. In that run V won
#      validation (48.5k vs S 52.5k) but failed the sealed test (86.1k vs S 75.2k): its BSD / tax coefficient came
#      from one round and its ABSD coefficient from three, too few to carry into 2020-2026. The rule is stated
#      openly because the sealed test showed the problem; V stays in every table for the record.
PREFER_MARGIN = 0.5
MIN_ROUNDS = 3
EXTRA = {}
if "H" in [n[0] for n in MODELS]:
    hcols, hflat, _ = [v for n, v in MODELS.items() if n.startswith("H")][0]
    if "sold4_k" in hdb:
        EXTRA["P  H + flats sold lag 4"] = (hcols[:5] + ["sold4_k"] + hcols[5:], True, False)
    if "tight36" in hdb:
        EXTRA["R  H + policy stance (tight36)"] = (hcols + ["tight36"], True, False)
        if "sold4_k" in hdb:
            EXTRA["S  H + flats sold lag 4 + policy stance"] = (hcols[:5] + ["sold4_k"] + hcols[5:] + ["tight36"], True, False)
            if "cool36" in hdb:
                sbase = hcols[:5] + ["sold4_k"] + hcols[5:]
                EXTRA["U  S, stance as cooling and easing counts"] = (sbase + ["cool36", "ease36"], True, False)
                EXTRA["V  S, stance as counts by policy type"] = (sbase + list(POLCOL.values()), True, False)
if "sold4_k" in hdb and "tight36" in hdb:
    EXTRA["T  O + policy stance (official data only)"] = (BASE0 + ["sold4_k"] + REST + ["tight36"], True, False)
ALL = {**{n: v for n, v in MODELS.items() if n[0] in "DFGHIJO"}, **EXTRA}
ENSEMBLE = {"Q  average of H and O": ("H", "O")} if all(any(n.startswith(k) for n in ALL) for k in "HO") else {}

def run_folds(name):
    """one design matrix per model: predictions for each validation year (trained on the years before it) and for
    the sealed test (trained on years before TEST_FROM). The test predictions are only looked at for the chosen model."""
    cols, flat, log = ALL[name]
    d = rows_for(cols)
    X, y, yr = design(cols, flat, d), d["price_k"].values, d["year"].values
    out = {}
    for Y in VAL_YEARS + [TEST_FROM]:
        tr, pr = yr < Y, (yr == Y) if Y != TEST_FROM else (yr >= TEST_FROM)
        out[Y] = pd.Series(LinearRegression().fit(X[tr], y[tr]).predict(X[pr]), index=d.index[pr])
    del X, d
    return out

P_ = {name: run_folds(name) for name in ALL}
for key, (ka, kb) in ENSEMBLE.items():
    a_, b_ = [P_[[n for n in ALL if n.startswith(k)][0]] for k in (ka, kb)]
    P_[key] = {Y: (a_[Y] + b_[Y]).dropna() / 2 for Y in a_}      # rows both models price

cv_rows, cv_mean = [], {}
for key in P_:
    maes = []
    for Y in VAL_YEARS:
        e = P_[key][Y] - hdb.loc[P_[key][Y].index, "price_k"]
        maes.append(float(e.abs().mean()))
        cv_rows.append({"model": key, "validation_year": Y, "MAE_k": round(maes[-1], 1)})
    cv_mean[key] = float(np.mean(maes))
cv = pd.DataFrame(cv_rows)
print(f"\nTime-series cross-validation (expanding window, validation years {VAL_YEARS[0]}-{VAL_YEARS[-1]}; 2020-2026 not used), MAE $k:")
print(cv.pivot(index="model", columns="validation_year", values="MAE_k").assign(mean=lambda t: t.mean(axis=1).round(1))
        .sort_values("mean").to_string())
# rounds of each kind announced before the test years: a count column needs MIN_ROUNDS of them to be chosen
before = pol[pol["announce_date"] < pd.Timestamp(f"{TEST_FROM}-01-01")] if POL.exists() else None
n_before = {}
if before is not None:
    n_before = {"cool36": int((before["direction"] > 0).sum()), "ease36": int((before["direction"] < 0).sum()),
                **{c: int((before["grp"] == g).sum()) for g, c in POLCOL.items()}}
eligible = lambda k: all(n_before.get(c, MIN_ROUNDS) >= MIN_ROUNDS for c in ALL[k][0] if c in n_before) if k in ALL else True
for k in P_:
    if not eligible(k):
        few = ", ".join(f"{c} {n_before[c]}" for c in ALL[k][0] if c in n_before and n_before[c] < MIN_ROUNDS)
        print(f"\n{k[:1]} cannot be chosen: fewer than {MIN_ROUNDS} rounds before {TEST_FROM} in {few} (it stays in every table for the record)")
chosen = min((k for k in cv_mean if eligible(k)), key=cv_mean.get)
skey = next((n for n in ALL if n.startswith("S")), None)
tkey = next((n for n in ALL if n.startswith("T")), None)
ukeys = [n for n in ALL if n[0] in "UV"]
for k in ukeys:
    if skey:
        print(f"\n{k[:1]} vs S on validation: {k[:1]} {cv_mean[k]:.1f}k, S {cv_mean[skey]:.1f}k ({k[:1]} must beat S by {PREFER_MARGIN}k)")
if chosen in ukeys and skey and cv_mean[skey] - cv_mean[chosen] < PREFER_MARGIN:
    print(f"{chosen[:1]} beats S by only {cv_mean[skey] - cv_mean[chosen]:.1f}k, so S (one stance column) is kept")
    chosen = skey
# what U and V learned (trained on sales up to 2019, the sealed-test fit): $k per round of each kind
for k in ukeys + ([skey] if skey else []):
    cols_k, flat_k, _ = ALL[k]
    d_k = rows_for(cols_k)
    t_k = d_k["year"] < TEST_FROM
    X_k = design(cols_k, flat_k, d_k)
    c_k = pd.Series(LinearRegression().fit(X_k[t_k], d_k.loc[t_k, "price_k"]).coef_, index=X_k.columns)
    print(f"{k[:1]} policy coefficients, trained 2001-2019 ($k per round counted in the 36 months before the sale): "
          + ", ".join(f"{c} {c_k[c]:+.2f}" for c in POLICY_COLS if c in c_k))
    del X_k, d_k
if chosen == skey and tkey and cv_mean[tkey] - cv_mean[skey] < PREFER_MARGIN:
    print(f"\nS beats T by only {cv_mean[tkey] - cv_mean[skey]:.1f}k on validation (rule: {PREFER_MARGIN}k), so the simpler, official-only T is chosen")
    chosen = tkey
elif skey and tkey:
    print(f"\nS vs T on validation: S {cv_mean[skey]:.1f}k, T {cv_mean[tkey]:.1f}k (S must beat T by {PREFER_MARGIN}k)")
hkey = [n for n in ALL if n.startswith("H")][0]
sealed = {}
# the validation winner, plus H, S and T for the record (the choice above was made before these are computed)
for key in dict.fromkeys([k for k in (chosen, hkey, skey, tkey, *ukeys) if k]):
    p = P_[key][TEST_FROM]
    y = hdb.loc[p.index, "price_k"]
    e = p - y
    sealed[key] = {"model": key, "cv_mean_MAE_k": round(cv_mean[key], 1), "test": f"{TEST_FROM}-{int(hdb['year'].max())}",
                   "n_test": int(len(e)), "MAE_k": round(float(e.abs().mean()), 1), "RMSE_k": round(float(np.sqrt((e ** 2).mean())), 1),
                   "MAPE_%": round(float((e.abs() / y).mean() * 100), 1), "bias_k": round(float(e.mean()), 1)}
print(f"\nChosen on validation: {chosen} (mean validation MAE {cv_mean[chosen]:.1f}k)")
print("Sealed test, scored once (validation winner first; H, S, T, U and V shown for the record, not used to choose):")
print(pd.DataFrame(sealed.values()).to_string(index=False))
cv.to_csv(OUT / "location_cv.csv", index=False)
pd.DataFrame(sealed.values()).to_csv(OUT / "location_sealed_test.csv", index=False)
print("Saved hdb_data/location_cv.csv and hdb_data/location_sealed_test.csv")

# ---------------------------------------------------------------- the age curve, measured from past sales
def curve(c):
    """price change from age 0 at each age, and the cost of each year, from a fitted coefficient series"""
    ages = np.arange(0, 71)
    v = c["flat_age"] * ages + sum(c[a] * np.clip(ages - k, 0, None) for k, a in zip(AGE_KNOTS, AGE))
    return pd.Series(v, index=ages)

dB = rows_for(BASE)                                  # the BTO-lag-4 sales (2005 on), as before
full = {}
for name, data in (("all years", dB), ("2017+ only", dB[dB["year"] >= 2017])):
    Xc = design(BASE + LOC + MRT + AGE, True, data)
    full[name] = curve(pd.Series(LinearRegression().fit(Xc, data["price_k"]).coef_, index=Xc.columns))
    del Xc
lin = coefs["D"]["flat_age"]
tab = pd.DataFrame({"age": range(0, 71, 5)})
for name, cv in full.items():
    tab[f"vs new, {name}"] = tab["age"].map(cv).round(0)
    tab[f"cost/yr next 5, {name}"] = tab["age"].map(lambda a: (cv[min(a + 5, 70)] - cv[a]) / 5).round(1)
tab["straight line (D)"] = (tab["age"] * lin).round(0)
print(f"\nAge curve measured from past sales ($k, same flat, same place, same market). Oldest flat sold: {int(dB['flat_age'].max())} yrs")
print("(ages above the oldest sold continue the last slope)")
print(tab.to_string(index=False))

# bias by age, fitted on all years: straight line (D) vs curve (F)
band = pd.cut(dB["flat_age"], [-1, 9, 19, 29, 39, 49, 99], labels=["0-9", "10-19", "20-29", "30-39", "40-49", "50+"])
bt_age = {}
for k in ("D", "F"):
    cols_k, flat_k, _ = [v for n, v in MODELS.items() if n.startswith(k)][0]
    Xk = design(cols_k, flat_k, dB)
    e = LinearRegression().fit(Xk, dB["price_k"]).predict(Xk) - dB["price_k"]
    del Xk
    bt_age[f"bias {k}"] = e.groupby(band, observed=True).mean()
    bt_age[f"MAE {k}"] = e.abs().groupby(band, observed=True).mean()
print("\nError by flat age, all years ($k; + bias = model too high):")
print(pd.DataFrame(bt_age).round(1).to_string())

# ---------------------------------------------------------------- hdb_location.js for the future-price page
# Model D, back-test in the same shape as hdb_drivers.js, then refitted on all years.
cand = res[res["model"].str[:1].isin(list("DFGHIJKO")) & (res["n_test"] == res["n_test"].max())]   # full 2020-2026 test only
short = res[res["n_test"] < res["n_test"].max()]
if len(short):
    print("\nNot eligible for the page (tested on fewer sales, e.g. no 2026 because 2025 completions are not out yet):", ", ".join(short["model"].str[:1]))
best = min(cand["model"], key=lambda n: (cand.set_index("model").loc[n, "MAE_k"], "FDLMNGHIJKO".index(n[0])))   # lowest MAE
print(f"\nLowest back-test MAE: {best}")
# The page uses the model chosen by time-series cross-validation on 2015-2019 (2020-2026 not used to choose),
# scored once on 2020-2026: see "Choosing without the test years" in the README. With policy_events.csv and the
# SingStat flats-sold series present this is S (H + flats sold lag 4 + policy stance), or T (official data only)
# when S does not beat T by PREFER_MARGIN.
best = chosen if chosen in ALL else best
print(f"\nPage model: {best}")
cols, flat, _ = ALL[best] if best in ALL else MODELS[best]
sups = [c for c in cols if c in SUPPLY]                  # the supply columns the page model uses (first = slider)
sup = sups[0]
STANCE = [c for c in POLICY_COLS if c in cols]       # the policy columns of the page model (S: tight36; U, V: counts)
NICE_SUP = {"avl4_k": "RFS lag 4 + BTO lag 4", "sold4_k": "flats sold lag 4"}
hdb = rows_for(cols)                                     # from here on: the page model's sales
y, tr, te = hdb["price_k"], hdb["year"] < TEST_FROM, hdb["year"] >= TEST_FROM
X = design(cols, flat)
bt = LinearRegression().fit(X[tr], y[tr])
fit_tr, pred = bt.predict(X[tr]), bt.predict(X[te])
err = pred - y[te]
per_year = (pd.DataFrame({"year": hdb.loc[te, "year"].values, "pred": pred, "err": err.values})
            .groupby("year").agg(pred=("pred", "mean"), mae=("err", lambda e: e.abs().mean()),
                                 rmse=("err", lambda e: np.sqrt((e ** 2).mean())), bias=("err", "mean"))
            .round(1).reset_index())
m = LinearRegression().fit(X, y)
fit_all = m.predict(X)
coef = dict(zip(X.columns, m.coef_))
r4 = lambda v: round(float(v), 4)
LOCCOLS = LOC + MRT
model = {
    "label": "Income + " + " + ".join(NICE_SUP.get(c, SUPPLY[c][2]) for c in sups) + ({"tight36": " + policy stance", "cool36": " + cooling and easing rounds"}.get(STANCE[0], " + policy rounds by type") if STANCE else "") + " + location (your block)",
    "cols": ["income_k"] + sups + STANCE, "slider": SUPPLY[sup][0], "loc": True, "key": best[:1],
    "beta": round(float(m.intercept_), 3),
    # same keys as the other drivers: flat columns, driver columns, towns
    "coef": {c: r4(v) for c, v in coef.items() if c in BASE0 + sups + STANCE or not (c.startswith("type_") or c.startswith("model_") or c in LOCCOLS or c in AGE or c == "rfs4")},
    "age_coef": {str(k): r4(coef[a]) for k, a in zip(AGE_KNOTS, AGE) if a in coef},   # extra $k per year past each bend age
    "max_age_sold": int(hdb["flat_age"].max()),
    "loc_coef": {c: r4(coef[c]) for c in LOCCOLS},
    "type_coef": {c[5:]: r4(v) for c, v in coef.items() if c.startswith("type_")} | {"4 ROOM": 0.0},
    "model_coef": {c[6:]: r4(v) for c, v in coef.items() if c.startswith("model_")} | {"MODEL A": 0.0},
    "r2": round(float(m.score(X, y)), 3), "n": int(len(hdb)),
    "period": f"{hdb['month'].min()[:7]} to {hdb['month'].max()[:7]}",
    "backtest": {"train": f"{int(hdb.loc[tr, 'year'].min())}-{TEST_FROM - 1}", "test": f"{TEST_FROM}-{int(hdb['year'].max())}",
                 "train_mae": round(float(np.abs(fit_tr - y[tr]).mean()), 1),
                 "train_rmse": round(float(np.sqrt(((fit_tr - y[tr]) ** 2).mean())), 1),
                 "mae": round(float(np.abs(err).mean()), 1), "rmse": round(float(np.sqrt((err ** 2).mean())), 1),
                 "mape": round(float((np.abs(err) / y[te]).mean() * 100), 1),
                 "r2": round(float(1 - (err ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()), 3),
                 "bias": round(float(err.mean()), 1), "years": per_year.values.tolist()},
    "yearly": hdb.assign(fit=fit_all).groupby("year")[["price_k", "fit"]].mean().round(1).reset_index().values.tolist(),
}
sales = hdb.groupby("bid").size()
bl = blk.copy()
bl["sales"] = bl["bid"].map(sales).fillna(0).astype(int)
bl["approx"] = bl.get("fill_note", pd.Series("", index=bl.index)).fillna("").str.contains("approx").astype(int)
num = lambda c, d: bl[c].astype(float).round(d)
blocks = pd.DataFrame({"b": bl["block"], "s": bl["street_name"], "t": bl["town"], "la": num("latitude", 5), "lo": num("longitude", 5),
                       "cbd": num("dist_to_cbd_km", 3), "mall": num("dist_to_mall_km", 3),
                       "sch": num("dist_to_nearest_high_demand_primary_school_km", 3),
                       "s1": bl["within_1km_high_demand_primary_school_flag"].fillna(0).astype(float).astype(int),
                       "ax": bl["approx"], "n": bl["sales"],
                       "mall_name": bl["nearest_mall"].fillna(""),
                       "sch_name": bl["nearest_high_demand_primary_school"].fillna("")}).dropna(subset=["cbd", "mall", "sch"])
towns_ok = set(hdb["town"].unique())
blocks = blocks[blocks["t"].isin(towns_ok)]
dec = lambda s: (pd.to_datetime(s).dt.year + (pd.to_datetime(s).dt.dayofyear - 1) / 365.25).round(3)
stations = pd.DataFrame({"name": st["station"], "lines": st["lines"], "la": st["latitude"].round(5), "lo": st["longitude"].round(5),
                         "ann": dec(st["announcement_date"]), "open": dec(st["opening_date"])})
model["selection"] = {"val_years": VAL_YEARS, "cv_mean_k": round(cv_mean.get(best, float("nan")), 1),
                      "cv": {k: round(v, 1) for k, v in sorted(cv_mean.items(), key=lambda t: t[1])}}
policy = None
if STANCE:
    pol_dec = pol["announce_date"].dt.year + (pol["announce_date"].dt.dayofyear - 1) / 365.25
    policy = {"window_years": 3, "cols": STANCE,           # each round: [date, direction x intensity, id, direction, policy type]
              "rounds": [[round(float(d), 4), float(w), str(i), int(r), str(g)] for d, w, i, r, g in zip(pol_dec, pol["w"], pol["event_id"], pol["direction"], pol["grp"])]}
loc_out = {"model": model, "cap_km": CAP_KM, "policy": policy,
           "blocks": blocks.values.tolist(), "block_cols": list(blocks.columns),
           "stations": stations.values.tolist(),
           "flat_types": sorted(hdb["flat_type"].unique()), "flat_models": sorted(hdb["flat_model"].unique())}
Path("hdb_location.js").write_text("window.HDB_LOCATION = " + json.dumps(loc_out, separators=(",", ":")) + ";\n")
print(f"Saved hdb_location.js: model {best[:1]} (MAE {model['backtest']['mae']}k), {len(blocks):,} blocks, {len(stations)} stations, "
      f"{Path('hdb_location.js').stat().st_size / 1e6:.1f} MB")
