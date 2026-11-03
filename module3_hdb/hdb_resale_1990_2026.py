"""
Download every HDB resale transaction from data.gov.sg (Jan 1990 to the latest month),
combine the 5 datasets into one table, save it, and fit the floor-area / flat-age plane.

Run inside the dsai-m3 environment:
    conda activate dsai-m3
    python hdb_resale_1990_2026.py

Free (no API key) limit on data.gov.sg is about 5 requests per minute, so the script
waits between calls. The whole download takes roughly 5-10 minutes.
"""
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

# The 5 datasets in data.gov.sg's "Resale Flat Prices" collection (collection 189).
# Together they cover Jan 1990 to the latest month. The script reads each one's dates itself.
DATASET_IDS = [
    "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",   # Jan 2017 onwards (updated monthly)
    "d_43f493c6c50d54243cc1eab0df142d6a",
    "d_2d5ff9ea31397b66239f245f57751537",
    "d_ebc5ab87086db484f88045b47411ebc5",
    "d_ea9ed51da2787afaf8e51f827c304208",
]
API = "https://api-open.data.gov.sg/v1/public/api/datasets/{}/{}"
WAIT = 15                       # seconds between API calls (stays under ~5 per minute)
OUT_DIR = Path("hdb_data")      # raw CSVs and the combined file are saved here
OUT_DIR.mkdir(exist_ok=True)


def call(dataset_id, action):
    """One API call, retrying politely if data.gov.sg says 'too many requests' (HTTP 429)."""
    for attempt in range(6):
        r = requests.get(API.format(dataset_id, action), timeout=60)
        if r.status_code == 429:
            print(f"    rate-limited, waiting {WAIT * 2}s ...")
            time.sleep(WAIT * 2)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"{action} kept hitting the rate limit for {dataset_id}")


def download(dataset_id):
    """initiate-download -> poll-download until a link appears -> read the CSV."""
    csv_path = OUT_DIR / f"{dataset_id}.csv"
    if csv_path.exists():                       # already downloaded earlier: reuse it
        print(f"  {dataset_id}: using saved {csv_path}")
        return pd.read_csv(csv_path)

    print(f"  {dataset_id}: preparing download ...")
    call(dataset_id, "initiate-download")
    for attempt in range(12):
        time.sleep(WAIT)
        data = call(dataset_id, "poll-download").get("data", {})
        print(f"    poll {attempt + 1}: {data.get('status')}")
        if data.get("url"):
            df = pd.read_csv(data["url"])
            df.to_csv(csv_path, index=False)    # keep a raw copy so reruns skip the download
            return df
    raise RuntimeError(f"No download link for {dataset_id} after 12 polls")


# ---------------------------------------------------------------- 1. download all five
frames = []
for i, ds in enumerate(DATASET_IDS):
    df = download(ds)
    df["source_dataset"] = ds
    print(f"    {len(df):,} rows, {df['month'].min()} to {df['month'].max()}")
    frames.append(df)
    if i < len(DATASET_IDS) - 1:
        time.sleep(WAIT)

# ---------------------------------------------------------------- 2. combine and tidy
# Older datasets have no 'remaining_lease' column; concat fills it with NaN for them.
hdb = pd.concat(frames, ignore_index=True)
hdb = hdb.drop_duplicates(subset=[c for c in hdb.columns if c != "source_dataset"])

hdb["month"] = pd.to_datetime(hdb["month"], format="%Y-%m")
hdb["year"] = hdb["month"].dt.year
hdb["flat_type"] = hdb["flat_type"].str.upper().str.replace("MULTI GENERATION", "MULTI-GENERATION")
hdb["flat_model"] = hdb["flat_model"].str.upper()
hdb["flat_age"] = hdb["year"] - hdb["lease_commence_date"]          # works for every year
hdb["remaining_lease_years"] = 99 - hdb["flat_age"]                 # same rule for all years
hdb = hdb.sort_values("month").reset_index(drop=True)

hdb.to_csv(OUT_DIR / "hdb_resale_all.csv", index=False)
print(f"\nAll years: {len(hdb):,} transactions, {hdb['month'].min():%Y-%m} to {hdb['month'].max():%Y-%m}")
print(hdb.groupby("year").size().to_string())

# ---------------------------------------------------------------- 3. the plane, per year
# Prices are in dollars of each year (not inflation-adjusted), so one plane for 1990-2026
# would mix very different price levels. Fitting each year separately shows how
# the value of a square metre and the cost of a year of age have changed.
rows = []
for year, g in hdb.groupby("year"):
    X = g[["floor_area_sqm", "flat_age"]]
    y = g["resale_price"] / 1000                                     # $k, like the plane page
    m = LinearRegression().fit(X, y)
    rows.append({"year": year, "n": len(g), "beta_k": m.intercept_,
                 "omega1_k_per_sqm": m.coef_[0], "omega2_k_per_year": m.coef_[1],
                 "r2": m.score(X, y)})
by_year = pd.DataFrame(rows).round(3)
by_year.to_csv(OUT_DIR / "plane_by_year.csv", index=False)
print("\nFitted plane for each year (price in $k):")
print(by_year.to_string(index=False))

# ---------------------------------------------------------------- 4. one plane for recent years
recent = hdb[hdb["year"] >= 2023]
X, y = recent[["floor_area_sqm", "flat_age"]], recent["resale_price"] / 1000
m = LinearRegression().fit(X, y)
print(f"\nRecent plane, {len(recent):,} rows ({recent['month'].min():%Y-%m} to {recent['month'].max():%Y-%m})")
print(f"beta  = {m.intercept_:.1f} $k")
print(f"omega1 = {m.coef_[0]:.2f} $k per m2")
print(f"omega2 = {m.coef_[1]:.2f} $k per year of age")
print(f"R2    = {m.score(X, y):.3f}")
