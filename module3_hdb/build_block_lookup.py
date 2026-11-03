"""Make the latest block lookup: your V3F lookup + the filled blocks + MRT columns rebuilt
from the corrected station table.

Reads  hdb_data/hdb_block_lookup_filled.csv   (from fill_missing_blocks.py: 220 new + 84 demolished blocks added)
       mrt/mrt_stations_corrected.csv        (184 stations: first announcement, first opening, coordinates)
Writes hdb_data/hdb_block_lookup_filled_mrt_corrected.csv   (same columns as your V3F lookup, plus a few new ones)
       hdb_data/mrt_columns_old_vs_new.csv                  (one row per block: old and new nearest station and distance)

Every MRT column keeps its V3F name; only the values are recomputed:
  nearest_mrt_current / dist_to_mrt_current_km   nearest station OPEN on AS_OF (the day you run this)
  nearest_mrt_2030 / dist_to_mrt_2030_km         nearest station open by 31 Dec 2030 (opened or planned)
  mrt_uplift_km                                  current distance - 2030 distance
  future_mrt_benefit_flag                        1 if a 2030 station is closer than today's (uplift > 0)
  future_mrt_benefit_gt_500m_flag                1 if it is more than 500 m closer
  within_500m / within_1km _mrt_current / _2030  distance flags
  mrt_stage_pre_announcement_flag                1 if the nearest station ever planned (any date) is not yet announced on AS_OF
New columns:
  nearest_mrt_current_lines, nearest_mrt_2030_lines, nearest_mrt_2030_opening_date
  nearest_mrt_announced / dist_to_mrt_announced_km / mrt_announced_opening_date
      nearest station announced but not open on AS_OF (any planned year, e.g. Cross Island Line 2030-32)
  mrt_as_of, mrt_source
School, mall and CBD columns are your V3F values (copied from the nearest block for filled rows).
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("hdb_data")
AS_OF = pd.Timestamp(date.today())
Y2030 = pd.Timestamp("2030-12-31")

blk = pd.read_csv(OUT / "hdb_block_lookup_filled.csv", dtype=str)
st = pd.read_csv("mrt/mrt_stations_corrected.csv").dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
st["ann"], st["opn"] = pd.to_datetime(st["announcement_date"]), pd.to_datetime(st["opening_date"])

old = blk[["block", "street_name", "town", "nearest_mrt_current", "dist_to_mrt_current_km",
           "nearest_mrt_2030", "dist_to_mrt_2030_km", "within_1km_mrt_current"]].copy()

has = blk["latitude"].notna() & blk["longitude"].notna()
lat, lon = blk.loc[has, "latitude"].astype(float).values, blk.loc[has, "longitude"].astype(float).values


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


D = haversine(lat[:, None], lon[:, None], st["latitude"].values[None, :], st["longitude"].values[None, :])


def nearest(mask):
    """index and distance of the nearest station among those in mask, for every block"""
    d = np.where(mask[None, :], D, np.inf)
    j = d.argmin(axis=1)
    return j, d[np.arange(len(j)), j]


j_cur, d_cur = nearest((st["opn"] <= AS_OF).values)
j_30, d_30 = nearest((st["opn"] <= Y2030).values)
j_ann, d_ann = nearest(((st["ann"] <= AS_OF) & (st["opn"] > AS_OF)).values)
j_all, _ = nearest(np.ones(len(st), bool))

r3 = lambda x: np.round(x, 4)
new = {
    "nearest_mrt_current": st["station"].values[j_cur], "dist_to_mrt_current_km": r3(d_cur),
    "nearest_mrt_current_lines": st["lines"].values[j_cur],
    "nearest_mrt_2030": st["station"].values[j_30], "dist_to_mrt_2030_km": r3(d_30),
    "nearest_mrt_2030_lines": st["lines"].values[j_30],
    "nearest_mrt_2030_opening_date": st["opening_date"].values[j_30],
    "mrt_uplift_km": r3(d_cur - d_30),
    "future_mrt_benefit_flag": (d_cur - d_30 > 1e-9).astype(float),
    "future_mrt_benefit_gt_500m_flag": (d_cur - d_30 > 0.5).astype(float),
    "within_500m_mrt_current": (d_cur <= 0.5).astype(float), "within_1km_mrt_current": (d_cur <= 1.0).astype(float),
    "within_500m_mrt_2030": (d_30 <= 0.5).astype(float), "within_1km_mrt_2030": (d_30 <= 1.0).astype(float),
    "mrt_stage_pre_announcement_flag": (st["ann"].values[j_all] > AS_OF).astype(float),
    "nearest_mrt_announced": np.where(np.isfinite(d_ann), st["station"].values[j_ann], None),
    "dist_to_mrt_announced_km": np.where(np.isfinite(d_ann), r3(d_ann), np.nan),
    "mrt_announced_opening_date": np.where(np.isfinite(d_ann), st["opening_date"].values[j_ann], None),
}
for c, v in new.items():
    if c not in blk.columns:
        blk[c] = None
    blk[c] = blk[c].astype(object)
    blk.loc[has, c] = v
blk["mrt_as_of"] = AS_OF.date().isoformat()
blk["mrt_source"] = "mrt/mrt_stations_corrected.csv"

out = OUT / "hdb_block_lookup_filled_mrt_corrected.csv"
blk.to_csv(out, index=False)

# ---------------------------------------------------------------- what changed
cmp = old.copy()
cmp["new_nearest_mrt_current"] = blk["nearest_mrt_current"]
cmp["new_dist_to_mrt_current_km"] = blk["dist_to_mrt_current_km"]
cmp["new_nearest_mrt_2030"] = blk["nearest_mrt_2030"]
cmp["new_dist_to_mrt_2030_km"] = blk["dist_to_mrt_2030_km"]
cmp.to_csv(OUT / "mrt_columns_old_vs_new.csv", index=False)

o = pd.to_numeric(cmp["dist_to_mrt_current_km"], errors="coerce")
n = pd.to_numeric(cmp["new_dist_to_mrt_current_km"], errors="coerce")
both = o.notna() & n.notna()
changed = both & (cmp["nearest_mrt_current"].str.upper() != cmp["new_nearest_mrt_current"].str.upper())
print(f"MRT columns as of {AS_OF.date()}: {int((st['opn'] <= AS_OF).sum())} stations open, "
      f"{int((st['opn'] <= Y2030).sum())} by 2030, {int(((st['ann'] <= AS_OF) & (st['opn'] > AS_OF)).sum())} announced not open")
print(f"Blocks: {len(blk):,} ({has.sum():,} with coordinates); from: {blk['fill_method'].fillna('V3F').str.split(' ').str[0].value_counts().to_dict()}")
print(f"\nNearest CURRENT station changed for {changed.sum():,} of {both.sum():,} blocks ({changed.mean() / both.mean():.0%})")
print(f"Median distance to current MRT: old {o[both].median():.3f} km -> new {n[both].median():.3f} km")
print(f"Within 1 km of a current MRT: old {pd.to_numeric(cmp['within_1km_mrt_current'], errors='coerce')[both].mean():.1%}"
      f" -> new {(n[both] <= 1).mean():.1%}")
big = cmp[both & ((o - n) > 0.5)].assign(km_closer=(o - n).round(2)).sort_values("km_closer", ascending=False)
print(f"\n{len(big):,} blocks are now more than 500 m closer to an open MRT; biggest changes:")
print(big[["block", "street_name", "town", "nearest_mrt_current", "dist_to_mrt_current_km",
           "new_nearest_mrt_current", "new_dist_to_mrt_current_km", "km_closer"]].head(10).to_string(index=False))
print(f"\nSaved {out} and {OUT / 'mrt_columns_old_vs_new.csv'}")
