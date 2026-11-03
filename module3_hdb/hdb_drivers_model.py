"""
HDB future price with one "driver" at a time, for the dropdown on hdb-future-price.html.

Every model:  price = beta + area + age + storey + town + driver(s) from LAST year
  income       median monthly household income (SingStat)
  ppi          URA private residential price index, All Residential (yearly average)
  gdp          real GDP per person, chained 2015 dollars (SingStat)
  income_rate  income + SORA interest rate (MAS, yearly average)

Each model is back-tested the same way: trained on 2001-2019 sales (interest rate: from
2006, SORA starts 2005), tested on the 2020-2026 sales it never saw. MAE (average miss per sale) is the fair comparison.

New HDB supply is measured three ways, each a separate driver, all tested on the same 2020-2026 sales:
  bto  BTO units launched (project list, 2001 on)          -> lag 0 / 1 / 4 / 5, trains from 2001 / 2002 / 2005 / 2006
  avl  RFS lag + BTO lag: RFS flats offered a year 1995-2000 (four quarterly selection batches),
       then BTO launched 2001 on                           -> lag 1 / 4 / 5, trains from 2001 (your RAG bridge)
  cmp  HDB flats completed (official, 2000 on)            -> lag 1 / 4 / 5, trains from 2001 / 2004 / 2005
Lag 0 / 1 test the ANNOUNCEMENT effect: buyers who apply for a BTO stop looking at resale flats
the moment it launches. Lag 4 / 5 test the KEY-COLLECTION effect, about 4 years after launch.
(avl lag 0 is not built: from 2001 on it is the same number as BTO lag 0.)

Also saves hdb_data/macro_annual.csv: one row per year with every driver
(the table that would go into BigQuery).

Needs hdb_data/hdb_resale_all.csv from hdb_resale_1990_2026.py.
    conda activate dsai-m3
    python hdb_drivers_model.py
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

OUT = Path("hdb_data")
API = "https://api-open.data.gov.sg/v1/public/api/datasets/{}/{}"
TEST_FROM = 2020
TRAIN_FROM = 2001       # same start for every model (income starts 2000, used from 2001 sales)
BASE_TOWN = "ANG MO KIO"
SOURCES = {   # name: (data.gov.sg dataset id, cached file)
    "income": ("d_6d878eb9c5a47f54fee7ce496f681e8d", "income_key_indicators.csv"),
    "ppi":    ("d_97f8a2e995022d311c6c68cfda6d034c", "private_ppi_quarterly.csv"),
    "gdp":    ("d_c43f61819c32009f2e86c29b0550e7fc", "gdp_per_capita_annual.csv"),
    "rate":   ("d_5fe5a4bb4a1ecc4d8a56a095832e2b24", "interest_rates_monthly.csv"),
    # SingStat TS/M400151 "Residential Units Constructed And Sold By HDB, Annual", 1974 on (official, not dark data)
    "hdbcs":  ("d_5363b6c403ae9a7d91c387eae64c4fac", "hdb_constructed_sold_annual.csv"),
}


# ------------------------------------------------------------------ 1. download (once, then cached)
def call(ds, action):
    for _ in range(6):
        r = requests.get(API.format(ds, action), timeout=60)
        if r.status_code == 429:
            time.sleep(30)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"{action} kept hitting the rate limit for {ds}")


def get(name):
    ds, file = SOURCES[name]
    path = OUT / file
    if not path.exists():
        print(f"downloading {name} ({ds}) ...")
        call(ds, "initiate-download")
        for attempt in range(12):
            time.sleep(15)
            data = call(ds, "poll-download").get("data", {})
            if data.get("url"):
                pd.read_csv(data["url"]).to_csv(path, index=False)
                break
        else:
            raise RuntimeError(f"no download link for {name}")
        time.sleep(15)                                  # stay under the free rate limit
    return pd.read_csv(path)


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def wide_row(df, starts):
    """One row of a wide SingStat table (a column per year or month) -> long table."""
    lab = df.columns[0]
    hit = df[df[lab].astype(str).str.strip().str.lower().str.startswith(starts.lower())]
    if hit.empty:
        raise SystemExit(f"row starting '{starts}' not found; rows are:\n" + "\n".join(df[lab].astype(str)))
    s = hit.iloc[0].drop(lab)
    return pd.DataFrame({"col": s.index.astype(str), "value": num(s)})


# ------------------------------------------------------------------ 2. one row per year
inc = wide_row(get("income"), "Median Monthly Household Employment Income Including Employer CPF")
inc["year"] = inc["col"].str[:4].astype(int)
income = inc.groupby("year")["value"].mean().rename("income")            # $ per month

p = get("ppi")
p = p[p["property_type"].astype(str).str.lower() == "all residential"]
p["year"] = p["quarter"].astype(str).str[:4].astype(int)
ppi = num(p["index"]).groupby(p["year"]).mean().rename("ppi")              # index, 2009-Q1 = 100

g = wide_row(get("gdp"), "Per Capita GDP In Chained")
g["year"] = g["col"].str[:4].astype(int)
gdp = g.groupby("year")["value"].mean().rename("gdp")                     # $ per person

r = wide_row(get("rate"), "Singapore Overnight Rate Average")
r["year"] = r["col"].str[:4].astype(int)
rate = r.dropna().groupby("year")["value"].mean().rename("rate")          # % a year

# ---- your supply workbooks (copied into hdb_data/) ----
BTO_XLSX = OUT / "bto-2001-2024_with_RFS_WIS_residual_2000_2004_reconstructed.xlsx"
PVT_XLSX = OUT / "Singapore_Private_Residential_Supply_Definitive_Merged_1995_2026_EC_History.xlsx"
for f in (BTO_XLSX, PVT_XLSX):
    if not f.exists():
        raise SystemExit(f"{f} not found - copy your supply workbook into hdb_data/")

# BTO: one row per project (town, name, units, launch date) -> units launched per calendar year.
# The RFS/WIS sheets for 2000-2004 are not added (your notes: queue, stock and offers must not be summed).
b = pd.read_excel(BTO_XLSX, sheet_name=0)
b.columns = [str(c).replace("\xa0", " ").strip() for c in b.columns]
b["units"] = pd.to_numeric(b["Units"], errors="coerce")
b["launch"] = pd.to_datetime(b[[c for c in b.columns if c.lower().startswith("launch")][0]], errors="coerce")
b = b.dropna(subset=["units", "launch"])
bto = b.groupby(b["launch"].dt.year)["units"].sum().rename("bto")              # units launched in the year

# ---- two more ways to measure new HDB supply, so 2001-2004 sales can be used ----
# avl: new-flat AVAILABILITY - your newspaper-RAG bridge (HDB_RAG_Reconciled_Lag4_Lag5_2000_2024.xlsx,
#      sheet Supply_Bridge_1995_2024): RFS flats offered for selection 1995-2000, then BTO units
#      offered 2001-2024 (identical to the project list), treated as ONE series of new flats on offer.
# cmp: new flats COMPLETED - official HDB flats constructed per year, 2000-2024
#      (HDB_Flats_Constructed_2000_2024.xlsx, SingStat via data.gov.sg): one consistent measure every year.
# bto (above) stays as it was, so "Income + BTO supply (lag 4)" is the same model as on the live page.
RFS_LAST = 2000                                                                  # last year avl is an RFS figure (from 2001: BTO)
# RFS years: HDB offered flats in four quarterly selection batches (new flats + earlier batches' unselected
# "remainder" flats). rfs_quarterly_1995_2001.csv holds every quarterly batch found in the newspaper archive.
#   RFS_MODE=year     (default) a year = 4 x the average of the quarters found, about 27,000-40,000 flats
#   RFS_MODE=quarter  the bridge's original single quarter per year (about 8,000-9,000), for comparison
import os
RFS_MODE = os.environ.get("RFS_MODE", "year")
RFS_Q = Path("rfs_quarterly_1995_2001.csv")
RAG_XLSX = OUT / "HDB_RAG_Reconciled_Lag4_Lag5_2000_2024.xlsx"
CMP_XLSX = OUT / "HDB_Flats_Constructed_2000_2024.xlsx"
avl = cmp = sold = None
# official annual series from SingStat (downloaded once into hdb_data/hdb_constructed_sold_annual.csv):
#   Flats Constructed                          -> cmp  (replaces the 2000-2024 workbook; same numbers, longer)
#   Flats Sold Under 'Home Ownership Scheme'   -> sold (new HDB flats sold in the year, one measure before and after BTO)
try:
    cs = get("hdbcs")
    def year_row(starts):
        w = wide_row(cs, starts)
        w = w[w["col"].str[:4].str.isdigit()]
        return pd.Series(w["value"].values, index=w["col"].str[:4].astype(int).values).dropna()
    cmp = year_row("Flats Constructed").rename("cmp")
    sold = year_row("Flats Sold").rename("sold")
    print(f"\nOfficial HDB flats constructed {cmp.index.min()}-{cmp.index.max()} and sold {sold.index.min()}-{sold.index.max()} (SingStat):")
    print("  " + ", ".join(f"{y}: built {int(cmp.get(y, 0)):,} sold {int(sold.get(y, 0)):,}" for y in range(1995, 2006)))
except Exception as e:
    print("\nSingStat constructed/sold table not available:", e)
if RAG_XLSX.exists():
    br = pd.read_excel(RAG_XLSX, sheet_name="Supply_Bridge_1995_2024", header=3)
    br = br[pd.to_numeric(br["Source Year"], errors="coerce").notna()]
    bridge = pd.Series(pd.to_numeric(br["Primary Supply Value"], errors="coerce").values,
                       index=br["Source Year"].astype(int).values).dropna()
    both = bridge.index.intersection(bto.index)
    diff = (bridge[both] - bto[both]).abs()
    print(f"\nRAG bridge {bridge.index.min()}-{bridge.index.max()}; matches the BTO project list for {int((diff < 1).sum())} of {len(both)} shared years")
    avl = bridge.combine_first(bto).rename("avl")                               # RFS 1995-2000, then BTO
    if RFS_MODE == "year" and RFS_Q.exists():
        q = pd.read_csv(RFS_Q).dropna(subset=["flats_available"])
        q = q[q["year"] <= RFS_LAST]
        per_year = q.groupby("year")["flats_available"].agg(["count", "mean"])
        yearly = (per_year["mean"] * 4).round()                                  # 4 x the average quarterly batch
        print("\nRFS years from the quarterly batches (RFS_MODE=year):")
        for y, r in per_year.iterrows():
            print(f"  {y}: {int(r['count'])} quarter(s) found, average {r['mean']:,.0f} a quarter -> {yearly[y]:,.0f} a year"
                  f"   (single-quarter bridge value: {bridge.get(y, float('nan')):,.0f})")
        avl = yearly.combine_first(avl).rename("avl")
    else:
        print(f"\nRFS years: single quarter per year from the bridge (RFS_MODE={RFS_MODE}"
              + ("" if RFS_Q.exists() else f"; {RFS_Q} not found") + ")")
else:
    print(f"\n{RAG_XLSX} not found - availability (RFS + BTO) drivers skipped")
if cmp is None and CMP_XLSX.exists():
    cm = pd.read_excel(CMP_XLSX, sheet_name=0, header=3)
    cm = cm[pd.to_numeric(cm["Year"], errors="coerce").notna()]
    cmp = pd.Series(pd.to_numeric(cm["HDB Flats Constructed"], errors="coerce").values,
                    index=cm["Year"].astype(int).values, name="cmp").dropna()
    print(f"HDB flats completed {cmp.index.min()}-{cmp.index.max()}: {', '.join(f'{y} {int(v):,}' for y, v in cmp.head(5).items())} ...")
elif cmp is None:
    print(f"{CMP_XLSX} not found - completion drivers skipped")

# Private: Annual_Model sheet, "Total Pipeline Supply" (URA pipeline at Q4, EC excluded; 2026 = 2Q)
pv = pd.read_excel(PVT_XLSX, sheet_name="Annual_Model", header=None)
hdr = pv.index[pv[0].astype(str).str.strip() == "Year"][0]
pv.columns = pv.iloc[hdr].astype(str).str.strip()
pv = pv.iloc[hdr + 1:]
pv = pv[pd.to_numeric(pv["Year"], errors="coerce").notna()]
pvt = pd.Series(pd.to_numeric(pv["Total Pipeline Supply"], errors="coerce").values,
                index=pv["Year"].astype(int).values, name="pvt").dropna()

macro = pd.concat([income, ppi, gdp, rate, bto, pvt] + [x for x in (avl, cmp, sold) if x is not None], axis=1).sort_index()
macro = macro[macro.index >= 1989].round(3)
print("\nbto = BTO units launched (project list); avl = RFS offered a year 1995-2000 + BTO from 2001; cmp = HDB flats constructed; sold = HDB flats sold (SingStat); pvt = private pipeline at Q4 (URA)")
macro.index.name = "year"
macro.to_csv(OUT / "macro_annual.csv")
print("\nOne row per year (each sale uses the PREVIOUS year's row):")
print(macro.to_string())

# ------------------------------------------------------------------ 3. sales + last year's drivers
hdb = pd.read_csv(OUT / "hdb_resale_all.csv",
                  usecols=["month", "year", "town", "storey_range", "floor_area_sqm", "flat_age", "resale_price"])
ends = hdb["storey_range"].str.split(" TO ", expand=True).astype(int)
hdb["storey"] = (ends[0] + ends[1]) / 2
hdb["price_k"] = hdb["resale_price"] / 1000
lagged = macro.copy()
lagged.index = lagged.index + 1                                            # year Y sale <- year Y-1 values
hdb = hdb.join(lagged, on="year")
hdb["income_k"] = hdb["income"] / 1000
hdb["gdp_k"] = hdb["gdp"] / 1000
# supply with an explicit lag: a sale in year Y uses supply from year Y-k
#   k = 0 / 1: launched in the sale year / the year before  (announcement effect)
#   k = 4 / 5: launched about when those flats complete     (key-collection effect)
for k in (0, 1, 4, 5):
    hdb[f"bto{k}_k"] = hdb["year"].map(lambda y: bto.get(y - k)) / 1000
    if k >= 4:
        hdb[f"pvt{k}_k"] = hdb["year"].map(lambda y: pvt.get(y - k)) / 1000
    if avl is not None and k >= 1:
        hdb[f"avl{k}_k"] = hdb["year"].map(lambda y: avl.get(y - k)) / 1000
        # 1 when the lagged figure is an RFS year (1995-2000: quarterly batches incl. remainder flats), 0 when it is BTO launches
        hdb[f"rfs{k}"] = (hdb["year"] - k <= RFS_LAST).astype(float)
for k in (1, 4, 5):
    if cmp is not None:
        hdb[f"cmp{k}_k"] = hdb["year"].map(lambda y: cmp.get(y - k)) / 1000     # flats completed k years before the sale
for k in (4, 5):
    if sold is not None:
        hdb[f"sold{k}_k"] = hdb["year"].map(lambda y: sold.get(y - k)) / 1000   # new flats sold k years before the sale

# ------------------------------------------------------------------ 3b. government policies and global events
# Each row becomes a 0/1 column: 1 for sales in the WINDOW months starting at that month.
# Its coefficient = how far prices sat above (+) or below (-) what the flat and the driver
# predict, on average, in that window. Edit policy_events.csv / global_events.csv to add or correct rows.
WINDOW = 12
# Your dbt seeds (built Sep 2026): policy_events.csv and global_events.csv, in this folder.
POLICY_CSV, GLOBAL_CSV = Path("policy_events.csv"), Path("global_events.csv")
for f in (POLICY_CSV, GLOBAL_CSV):
    if not f.exists():
        raise SystemExit(f"{f} not found - copy your dbt seed file into this folder")


def policy_group(r):
    """Group a policy round by direction and the instruments it used."""
    ins = str(r["instruments"]).upper().split(";")
    if int(r["direction"]) < 0:
        return "Easing (incl. grants)"
    if "ABSD" in ins:
        return "ABSD"
    if "SSD" in ins:
        return "SSD"
    if ins == ["BSD"]:
        return "BSD / tax"
    return "Loan limits & HDB rules"          # LTV, MSR, TDSR, tenure, stress rate, wait-out, MOP, IAS/IOL, DPS


EVENT_GROUP = {   # global_events.csv event_id -> group used for the averages on the page
    "G06": "Crisis followed by QE", "G13": "Crisis followed by QE",
    "G01": "Crisis without QE", "G02": "Crisis without QE", "G03": "Crisis without QE", "G04": "Crisis without QE",
    "G09": "Fed tightening / taper", "G16": "Fed tightening / taper",
    "G18": "Fed easing",
    "G12": "Trade / geopolitical shock", "G15": "Trade / geopolitical shock", "G19": "Trade / geopolitical shock",
    "G05": "Other financial shock", "G10": "Other financial shock",
    "G14": "Local lockdown (Circuit Breaker)",
}
pol = pd.read_csv(POLICY_CSV, dtype=str)
glo = pd.read_csv(GLOBAL_CSV, dtype=str)
lab = lambda d: pd.Timestamp(d).strftime("%b %Y")
events = pd.concat([
    pd.DataFrame({"id": pol["event_id"], "type": "policy", "month": pol["effective_date"].str[:7],
                  "label": pol["effective_date"].map(lab), "detail": pol["description"], "source": pol["source"],
                  "group": pol.apply(policy_group, axis=1), "segment": pol["segment"], "confidence": pol["confidence"]}),
    pd.DataFrame({"id": glo["event_id"], "type": "event", "month": glo["start_date"].str[:7],
                  "label": glo["start_date"].map(lab), "detail": glo["name"] + ": " + glo["description"], "source": "",
                  "group": glo["event_id"].map(EVENT_GROUP).fillna("Low relevance (not averaged)"),
                  "segment": glo["region"], "confidence": glo["sg_property_relevance"]}),
], ignore_index=True).sort_values(["type", "month"], ascending=[False, True]).reset_index(drop=True)
print("\nPolicy rounds and global events, with the group each one is averaged in:")
print(events[["id", "month", "group"]].to_string(index=False))
hdb["ym"] = hdb["month"].str[:4].astype(int) * 12 + hdb["month"].str[5:7].astype(int)
EV_COLS = []
for e in events.itertuples():
    start = int(e.month[:4]) * 12 + int(e.month[5:7])
    col = "ev_" + e.id
    hdb[col] = ((hdb["ym"] >= start) & (hdb["ym"] < start + WINDOW)).astype("float32")   # float32: half the memory
    EV_COLS.append(col)
print(f"\n{len(events)} policies/events from {POLICY_CSV} and {GLOBAL_CSV}, each a {WINDOW}-month window")

BASE_FLAT = ["floor_area_sqm", "flat_age", "storey"]
# ------------------------------------------------------------------ 3c. what actual prices did after each round
# Flat-only model on ALL sales from 1990 (no driver): log price ~ floor area + age + storey + town.
# Its average gap per month is a price index with the flat mix taken out (log points).
Xf = pd.concat([hdb[BASE_FLAT], pd.get_dummies(hdb["town"], dtype=float).drop(columns=BASE_TOWN)], axis=1)
lp = np.log(hdb["resale_price"])
fm = LinearRegression().fit(Xf, lp)
index = (lp - fm.predict(Xf)).groupby(hdb["ym"]).mean()                 # log points, by month
index = index - index[index.index >= index.index.max() - 11].mean()     # latest 12 months = 0


def lv(a, b):
    v = index.reindex(range(a, b + 1))
    return float(v.mean()) if v.notna().sum() >= 2 else None


pc = lambda x: round((np.exp(x) - 1) * 100, 1)                          # log points -> %
# Normal growth = average yearly growth of the index over the whole period (log points a year)
first, last = index.index.min(), index.index.max()
NORMAL = float((index.loc[last - 11:last].mean() - index.loc[first:first + 11].mean()) / ((last - first - 11) / 12))
print(f"\nNormal (long-run average) HDB price growth: {pc(NORMAL)}% a year, {first // 12}-{last // 12}")
study = {}
for e in events.itertuples():
    st = int(e.month[:4]) * 12 + int(e.month[5:7])
    now, yr_before = lv(st - 3, st - 1), lv(st - 15, st - 13)
    if now is None or yr_before is None:
        study[e.id] = None
        continue
    before = now - yr_before                                             # growth in the 12 months before
    # policies are aimed at the trend going in; global shocks are compared with normal growth
    # policies: the trend going in for year 1, then normal growth (no boom or slump runs at full speed for 3 years)
    # global events: normal growth throughout
    def base(h):
        return before * min(h, 12) / 12 + NORMAL * max(h - 12, 0) / 12 if e.type == "policy" else NORMAL * h / 12
    after, extra = [], []
    for h in (12, 24, 36):
        later = lv(st + h - 3, st + h - 1)
        after.append(None if later is None else pc(later - now))
        extra.append(None if later is None else pc(later - now - base(h)))
    study[e.id] = {"before": pc(before), "baseline": "trend then normal" if e.type == "policy" else "normal growth",
                   "base": [pc(base(h)) for h in (12, 24, 36)], "after": after, "extra": extra}

print("\nActual HDB prices around each round (flat mix taken out, all sales from 1990), %:")
print(f"  {'month':8} {'group':28} {'12m before':>10} | {'after 1y':>8} {'2y':>7} {'3y':>7} | {'baseline 3y':>14} | {'vs base 1y':>10} {'2y':>7} {'3y':>7}")
f = lambda v: f"{v:+.1f}" if v is not None else "n/a"
for e in events.itertuples():
    r = study[e.id]
    if r is None:
        print(f"  {e.month:8} {str(e.group)[:28]:28} {'n/a':>10}")
        continue
    print(f"  {e.month:8} {str(e.group)[:28]:28} {f(r['before']):>10} | {f(r['after'][0]):>8} {f(r['after'][1]):>7} {f(r['after'][2]):>7} | {r['baseline'][:6]:>7} {f(r['base'][2]):>6} | {f(r['extra'][0]):>10} {f(r['extra'][1]):>7} {f(r['extra'][2]):>7}")
print("  group averages (vs baseline 1y / 2y / 3y):")
for g in events["group"].unique():
    if g.startswith("Low relevance"):
        continue
    rows_g = [study[i] for i in events.loc[events["group"] == g, "id"] if study[i]]
    avg = [np.mean([r["extra"][k] for r in rows_g if r["extra"][k] is not None]) if any(r["extra"][k] is not None for r in rows_g) else None for k in range(3)]
    print(f"  {'':8} {g[:28]:28} ({len(rows_g)} rounds)   " + "  ".join(f(round(a, 1)) if a is not None else "n/a" for a in avg))
index_out = [[int(k), round(float(v), 4)] for k, v in index.items()]

BASE = ["floor_area_sqm", "flat_age", "storey"]
DRIVERS = {   # key: (label, driver columns, the series the user's slider moves)
    "income":      ("Income",                 ["income_k"],          "income"),
    "ppi":         ("Private PPI",            ["ppi"],               "ppi"),
    "gdp":         ("Real GDP per person",    ["gdp_k"],             "gdp"),
    "income_rate": ("Income + interest rate", ["income_k", "rate"],  "rate"),
    "income_bto0": ("Income + BTO supply (lag 0, same year)", ["income_k", "bto0_k"], "bto"),
    "income_bto1": ("Income + BTO supply (lag 1)", ["income_k", "bto1_k"], "bto"),
    "income_bto4": ("Income + BTO supply (lag 4)", ["income_k", "bto4_k"], "bto"),
    "income_bto5": ("Income + BTO supply (lag 5)", ["income_k", "bto5_k"], "bto"),
    "income_pvt4": ("Income + private pipeline (lag 4)", ["income_k", "pvt4_k"], "pvt"),
    "income_pvt5": ("Income + private pipeline (lag 5)", ["income_k", "pvt5_k"], "pvt"),
}
if avl is not None:
    DRIVERS["income_avl1"] = ("Income + RFS lag 1 + BTO lag 1", ["income_k", "avl1_k"], "avl")
    DRIVERS["income_avl4"] = ("Income + RFS lag 4 + BTO lag 4", ["income_k", "avl4_k"], "avl")
    DRIVERS["income_avl5"] = ("Income + RFS lag 5 + BTO lag 5", ["income_k", "avl5_k"], "avl")
if cmp is not None:
    DRIVERS["income_cmp1"] = ("Income + HDB flats completed (lag 1)", ["income_k", "cmp1_k"], "cmp")
    DRIVERS["income_cmp4"] = ("Income + HDB flats completed (lag 4)", ["income_k", "cmp4_k"], "cmp")
    DRIVERS["income_cmp5"] = ("Income + HDB flats completed (lag 5)", ["income_k", "cmp5_k"], "cmp")
if sold is not None:
    DRIVERS["income_sold4"] = ("Income + HDB flats sold (lag 4, official)", ["income_k", "sold4_k"], "sold")
    DRIVERS["income_sold5"] = ("Income + HDB flats sold (lag 5, official)", ["income_k", "sold5_k"], "sold")
# printed back-test only, not on the page: launches in the sale year / year before go with HIGHER
# prices (+ coefficient) because HDB launches more when the market is hot, so a slider would mislead
PRINT_ONLY = {"income_bto0", "income_bto1", "income_avl1"}
test_years = sorted(hdb.loc[hdb["year"] >= TEST_FROM, "year"].unique())
models, rows = {}, []
for key, (label, cols, slider) in DRIVERS.items():
    keep = list(dict.fromkeys(["month", "year", "town", "price_k"] + BASE + cols + EV_COLS + [c for c in ("rfs1", "rfs4", "rfs5") if c in hdb]))
    df = hdb.loc[hdb["year"] >= TRAIN_FROM, keep].dropna(subset=cols)   # only the columns this model needs
    X = pd.concat([df[BASE + cols], pd.get_dummies(df["town"], dtype=float).drop(columns=BASE_TOWN)], axis=1)
    y = df["price_k"]
    tr, te = df["year"] < TEST_FROM, df["year"] >= TEST_FROM
    bt = LinearRegression().fit(X[tr], y[tr])
    fit_tr, pred = bt.predict(X[tr]), bt.predict(X[te])
    err = pred - y[te]                                                     # + = model too high
    bt_m = {"train_mae": round(mean_absolute_error(y[tr], fit_tr), 1),    # on the years it learned from
            "train_rmse": round(float(np.sqrt(np.mean((fit_tr - y[tr]) ** 2))), 1),
            "mae": round(float(np.mean(np.abs(err))), 1),                 # average miss, $k
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 1),          # big misses count more, $k
            "mape": round(float(np.mean(np.abs(err) / y[te]) * 100), 1),  # average miss as % of price
            "r2": round(r2_score(y[te], pred), 3), "bias": round(float(err.mean()), 1)}
    per_year = (pd.DataFrame({"year": df.loc[te, "year"], "pred": pred, "err": err})
                .groupby("year").agg(pred=("pred", "mean"), mae=("err", lambda e: e.abs().mean()),
                                     rmse=("err", lambda e: np.sqrt((e ** 2).mean())), bias=("err", "mean"))
                .round(1).reset_index())
    # regime test (printed only): add a 0/1 column marking sales whose lagged supply is an RFS year
    rcol = {"avl1_k": "rfs1", "avl4_k": "rfs4", "avl5_k": "rfs5"}.get(cols[-1])
    bt_r = None
    if rcol:
        Xr = pd.concat([X, df[[rcol]]], axis=1)
        br_m = LinearRegression().fit(Xr[tr], y[tr]); er = br_m.predict(Xr[te]) - y[te]
        bt_r = {"mae": round(float(np.mean(np.abs(er))), 1), "rmse": round(float(np.sqrt(np.mean(er ** 2))), 1),
                "bias": round(float(er.mean()), 1), "coef": round(float(br_m.coef_[-1]), 2)}
        del Xr, br_m
    # same back-test with the policy/event windows added
    Xe = pd.concat([X, df[EV_COLS]], axis=1)
    bte = LinearRegression().fit(Xe[tr], y[tr])
    pe = bte.predict(Xe[te]); ee = pe - y[te]
    bt_e = {"mae": round(float(np.mean(np.abs(ee))), 1), "rmse": round(float(np.sqrt(np.mean(ee ** 2))), 1),
            "mape": round(float(np.mean(np.abs(ee) / y[te]) * 100), 1), "r2": round(r2_score(y[te], pe), 3),
            "bias": round(float(ee.mean()), 1)}
    del Xe, bte                                                       # the 49-column table is the biggest; free it now
    gc.collect()
    # final model: all years, flat + driver only (the windows made the back-test worse, so they stay out)
    m = LinearRegression().fit(X, y)
    fit_all = m.predict(X)
    yearly = df.assign(fit=fit_all).groupby("year")[["price_k", "fit"]].mean().round(1).reset_index()
    models[key] = {
        "label": label, "cols": cols, "slider": slider,
        "beta": round(m.intercept_, 3),
        "coef": {c: round(v, 4) for c, v in zip(X.columns, m.coef_) if not c.startswith("ev_")},
        "backtest_ev": bt_e, "backtest_rfs": bt_r,
        "r2": round(m.score(X, y), 3), "n": len(df), "period": f"{df['month'].min()[:7]} to {df['month'].max()[:7]}",
        "mae_all": round(mean_absolute_error(y, fit_all), 1), "rmse_all": round(float(np.sqrt(np.mean((fit_all - y) ** 2))), 1),
        "backtest": {"train": f"{df.loc[tr, 'year'].min()}-{TEST_FROM - 1}", "test": f"{TEST_FROM}-{df['year'].max()}",
                     **bt_m, "years": per_year.values.tolist()},          # years: [year, mean pred, MAE, RMSE, bias]
        "yearly": yearly.values.tolist()}
    rows.append((label, models[key], [float(round(m.coef_[list(X.columns).index(c)], 3)) for c in cols], per_year))
    del df, X, y, m                                          # free memory before the next driver (8 GB laptop)
    gc.collect()

print(f"\nBack-test: trained on earlier years, every model tested on the same {TEST_FROM}-{max(test_years)} sales ($k)")
print(f"  {'driver':46} {'train':9} {'trainMAE':>8} {'trainRMSE':>9} | {'MAE':>6} {'RMSE':>6} {'MAPE%':>6} {'R2':>6} {'bias':>7} | driver coef")
for label, mm, dc, _ in rows:
    b = mm["backtest"]
    print(f"  {label:46} {b['train']:9} {b['train_mae']:8} {b['train_rmse']:9} | {b['mae']:6} {b['rmse']:6} {b['mape']:6} {b['r2']:6} {b['bias']:+7} | {dc}")
print("\nFor comparison only - same back-test with a 0/1 column per policy/event window added (not used on the page) ($k):")
for label, mm, _, _ in rows:
    b = mm["backtest_ev"]
    print(f"  {label:46} MAE {b['mae']:6}  RMSE {b['rmse']:6}  MAPE {b['mape']:5}%  R2 {b['r2']:6}  bias {b['bias']:+7}   (without: MAE {mm['backtest']['mae']})")
if avl is not None:
    print("\nRFS-regime test (printed only) - same back-test with a 0/1 column for sales whose lagged supply is a 1995-2000 RFS year:")
    for label, mm, _, _ in rows:
        b = mm.get("backtest_rfs")
        if b:
            print(f"  {label:46} MAE {b['mae']:6}  RMSE {b['rmse']:6}  bias {b['bias']:+7}  RFS column {b['coef']:+.2f}k   (without: MAE {mm['backtest']['mae']})")
print("\nBack-test by year (MAE / RMSE, $k):")
yrs = rows[0][3]["year"].tolist()
print(f"  {'driver':46} " + " ".join(f"{y:>13}" for y in yrs))
for label, _, _, py in rows:
    d = {int(r.year): f"{r.mae:.0f} / {r.rmse:.0f}" for r in py.itertuples()}
    print(f"  {label:46} " + " ".join(f"{d.get(y, '-'):>13}" for y in yrs))

# ------------------------------------------------------------------ 4. growth rates for the sliders
def cagr(s, n):
    s = s.dropna()
    last = s.index.max()
    return float(round(((s[last] / s[last - n]) ** (1 / n) - 1) * 100, 2)) if (last - n) in s.index else None

series = {}
for name in [n for n in ["income", "ppi", "gdp", "rate", "bto", "pvt", "avl", "cmp", "sold"] if n in macro]:
    s = macro[name].dropna()
    series[name] = {"values": [[int(y), float(v)] for y, v in s.items()],
                    "cagr": {f"{n}y": cagr(s, n) for n in (5, 10, 20) if cagr(s, n) is not None},
                    "avg10": round(float(s[s.index > s.index.max() - 10].mean()), 3)}
    print(f"{name:7} {int(s.index.min())}-{int(s.index.max())}  last {s.iloc[-1]:,.2f}  growth/yr {series[name]['cagr']}")

out = {"events": events.to_dict("records"), "window": WINDOW, "study": study, "normal": pc(NORMAL), "index": index_out, "models": {k: v for k, v in models.items() if k not in PRINT_ONLY}, "series": series, "base_town": BASE_TOWN, "towns": sorted(hdb["town"].unique()),
       "latest_sale_year": int(hdb["year"].max())}
Path("hdb_drivers.js").write_text("window.HDB_DRIVERS = " + json.dumps(out, separators=(",", ":")) + ";\n")
print(f"saved hdb_drivers.js ({len(models) - len(PRINT_ONLY & models.keys())} drivers; printed only: {', '.join(sorted(PRINT_ONLY & models.keys()))}) and hdb_data/macro_annual.csv")
