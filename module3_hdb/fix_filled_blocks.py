"""Recalculate the CBD, mall and school columns of the blocks that fill_missing_blocks.py added.

fill_missing_blocks.py copied these distances from the nearest original block. In new estates
(Punggol, Sengkang, Tengah ...) that block can be far away, so check_block_lookup.py flagged them.
Here each filled block gets its own values, measured from its own coordinates to:
  - the CBD point your original lookup uses (worked out from your original blocks)
  - every mall and popular primary school, at the positions worked out from your original blocks
    (hdb_data/mall_school_locations.csv from check_block_lookup.py)
Your original blocks are not touched.

Updates, after saving a .bak copy of each:
  hdb_data/hdb_block_lookup_filled.csv
  hdb_data/hdb_block_lookup_filled_mrt_corrected.csv
Run check_block_lookup.py first, then:
    python fix_filled_blocks.py
    python check_block_lookup.py          (the filled blocks should no longer be flagged)
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

OUT = Path("hdb_data")
FILES = ["hdb_block_lookup_filled.csv", "hdb_block_lookup_filled_mrt_corrected.csv"]
CBD, MALL, SCH, FLAG = "dist_to_cbd_km", "dist_to_mall_km", "dist_to_nearest_high_demand_primary_school_km", "within_1km_high_demand_primary_school_flag"
MALLN, SCHN = "nearest_mall", "nearest_high_demand_primary_school"

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

loc = pd.read_csv(OUT / "mall_school_locations.csv").dropna(subset=["lat", "lon"])
malls, schools = loc[loc["kind"] == "mall"], loc[loc["kind"] == "school"]
print(f"Positions from your original blocks: {len(malls)} malls, {len(schools)} popular schools")
for name in ["Hougang Mall", "Valley Point"]:
    r = malls[malls["name"] == name]
    if len(r):
        print(f"   {name}: ({r['lat'].iat[0]:.5f}, {r['lon'].iat[0]:.5f}) from {r['blocks'].iat[0]} blocks"
              + (f", OneMap matched '{r['onemap_match'].iat[0]}' {r['fit_vs_onemap_m'].iat[0]:.0f} m away" if "onemap_match" in r else ""))

for fname in FILES:
    path = OUT / fname
    if not path.exists():
        print(f"\n{fname}: not found, skipped")
        continue
    b = pd.read_csv(path, dtype={"postal_code": str}, low_memory=False)
    filled = b.get("fill_method", pd.Series("", index=b.index)).fillna("").astype(str).str.contains("OneMap") & b["latitude"].notna()
    orig = b["fill_method"].isna() if "fill_method" in b else ~filled
    orig &= b["latitude"].notna() & b[CBD].notna()
    # the CBD point your original blocks were measured from
    o = b[orig]
    cbd = least_squares(lambda p: haversine(o["latitude"].values, o["longitude"].values, p[0], p[1]) - o[CBD].values,
                        x0=[1.2840, 103.8514]).x
    f = b[filled]
    la, lo = f["latitude"].values, f["longitude"].values
    Dm = haversine(la[:, None], lo[:, None], malls["lat"].values[None, :], malls["lon"].values[None, :])
    Ds = haversine(la[:, None], lo[:, None], schools["lat"].values[None, :], schools["lon"].values[None, :])
    new = pd.DataFrame({CBD: haversine(la, lo, cbd[0], cbd[1]).round(3),
                        MALLN: malls["name"].values[Dm.argmin(axis=1)], MALL: Dm.min(axis=1).round(3),
                        SCHN: schools["name"].values[Ds.argmin(axis=1)], SCH: Ds.min(axis=1).round(3)}, index=f.index)
    new[FLAG] = (new[SCH] <= 1.0).astype(int)
    change = pd.DataFrame({"cbd_km": (new[CBD] - f[CBD]).abs(), "mall_km": (new[MALL] - f[MALL]).abs(),
                           "school_km": (new[SCH] - f[SCH]).abs(),
                           "mall_renamed": new[MALLN] != f[MALLN], "school_renamed": new[SCHN] != f[SCHN]})
    shutil.copy(path, path.with_suffix(".csv.bak"))
    for c in new.columns:
        b[c] = b[c].astype(object) if c in (MALLN, SCHN) else b[c]
        b.loc[filled, c] = new[c]
    if "fill_note" in b:
        b["fill_note"] = b["fill_note"].astype(object)
        b.loc[filled, "fill_note"] = b.loc[filled, "fill_note"].fillna("").astype(str) + "; CBD/mall/school recalculated"
    b.to_csv(path, index=False)
    print(f"\n{fname}: {filled.sum()} filled blocks recalculated (backup: {fname}.bak)")
    print(f"   CBD point used: ({cbd[0]:.4f}, {cbd[1]:.4f})")
    print(f"   changed by more than 0.3 km: CBD {(change['cbd_km'] > 0.3).sum()}, mall {(change['mall_km'] > 0.3).sum()}, "
          f"school {(change['school_km'] > 0.3).sum()}; nearest mall renamed {change['mall_renamed'].sum()}, "
          f"nearest school renamed {change['school_renamed'].sum()}")
    big = change.assign(block=f["block"], street=f["street_name"], town=f.get("town")).sort_values("cbd_km", ascending=False).head(5)
    print("   largest CBD corrections:", "; ".join(f"{r.block} {r.street} ({r.town}) {r.cbd_km:.2f} km" for r in big.itertuples()))
