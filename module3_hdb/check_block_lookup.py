"""Check the CBD, mall and school columns of the HDB block lookup.

Reads (first one found):
  hdb_data/hdb_block_lookup_filled_mrt_corrected.csv
  hdb_data/hdb_block_lookup_filled.csv
  hdb_data/hdb_block_lookup_from_V3F.csv
Nothing is changed in the lookup. Writes:
  hdb_data/block_lookup_check.csv        every block that failed a check, and which check
  hdb_data/mall_school_locations.csv     where each mall / school must be, worked out from the blocks
                                         (and OneMap's location, with --onemap)

Checks (offline, seconds):
  1. CBD      recompute the straight-line distance from latitude/longitude to Raffles Place and compare;
              also find the CBD point that best fits your column (shows which point was used)
  2. School   within_1km flag must be 1 exactly when the school distance is 1 km or less
  3. Distance kind  straight-line or travel (road / walking) distance? (ratio to straight line)
  4. Mall / school positions: every block names its nearest mall and school with a distance, so the
              position of each mall and school can be worked out from all the blocks that name it
              (like finding a phone from several masts). Blocks whose distance does not fit are flagged,
              and so are blocks where another mall / school is clearly nearer than the one named.
  5. Neighbours  two blocks 150 m apart naming the same mall cannot differ by much more than 150 m
              in distance to it (triangle rule)
Optional, needs internet (your laptop):
  --onemap    look up every mall and school name on OneMap (free, no key) and recompute every block's
              distance from the official position

    conda activate dsai-m3
    python check_block_lookup.py            (offline checks)
    python check_block_lookup.py --onemap   (also check against OneMap, a few minutes)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

OUT = Path("hdb_data")
RAFFLES = (1.2840, 103.8514)          # Raffles Place MRT, the usual "CBD" point
TOL = 0.30                             # km: allowed miss before a block is flagged
NEAR = 0.15                            # km: neighbour radius for the triangle check
ONEMAP = "https://www.onemap.gov.sg/api/common/elastic/search"

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

path = next((OUT / f for f in ["hdb_block_lookup_filled_mrt_corrected.csv", "hdb_block_lookup_filled.csv",
                                "hdb_block_lookup_from_V3F.csv"] if (OUT / f).exists()), None)
if path is None:
    sys.exit("No block lookup found in hdb_data/")
b = pd.read_csv(path, dtype={"postal_code": str}, low_memory=False).dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
print(f"Checking {path}: {len(b):,} blocks with coordinates\n")
CBD, MALL, SCH, FLAG = "dist_to_cbd_km", "dist_to_mall_km", "dist_to_nearest_high_demand_primary_school_km", "within_1km_high_demand_primary_school_flag"
MALLN, SCHN = "nearest_mall", "nearest_high_demand_primary_school"
lat, lon = b["latitude"].values, b["longitude"].values
issues = []                                                       # (row, check, detail)
def flag(rows, check, detail):
    for r, d in zip(rows, detail):
        issues.append((r, check, d))

# ---------------------------------------------------------------- 1. CBD
straight = haversine(lat, lon, *RAFFLES)
diff = b[CBD].values - straight
fit = least_squares(lambda p: haversine(lat, lon, p[0], p[1]) - b[CBD].values, x0=RAFFLES)
best = fit.x
d_best = b[CBD].values - haversine(lat, lon, *best)
print("1. CBD distance")
print(f"   vs Raffles Place {RAFFLES}: median difference {np.median(diff):+.3f} km, 95% of blocks within ±{np.percentile(np.abs(diff), 95):.3f} km")
print(f"   the CBD point that best fits your column: ({best[0]:.4f}, {best[1]:.4f}), "
      f"{haversine(best[0], best[1], *RAFFLES) * 1000:.0f} m from Raffles Place; misses: median {np.median(np.abs(d_best)):.3f} km")
bad = np.where(np.abs(d_best) > TOL)[0]
flag(bad, "cbd", [f"column {b[CBD].iat[r]:.2f} km, from coordinates {haversine(lat[r], lon[r], *best):.2f} km" for r in bad])
print(f"   blocks off by more than {TOL} km: {len(bad)}")

# ---------------------------------------------------------------- 2. school flag
print("\n2. Popular-school flag vs distance")
f = b[FLAG].fillna(0).astype(float).values
d = b[SCH].values
bad = np.where(((d <= 1.0) & (f == 0)) | ((d > 1.0) & (f == 1)))[0]
flag(bad, "school_flag", [f"distance {d[r]:.2f} km but flag {int(f[r])}" for r in bad])
print(f"   blocks where the flag and the distance disagree: {len(bad)}")

# ---------------------------------------------------------------- 3. straight-line or travel distance?
ratio = np.median(b[CBD].values / np.maximum(straight, 0.01))
kind = "straight-line" if abs(ratio - 1) < 0.08 else "travel (road or walking)"
print(f"\n3. Distance kind: your CBD column is {ratio:.2f} x the straight line, so it looks like {kind} distance")
scale = 1.0 if kind == "straight-line" else ratio                # used below so travel distances are not all flagged

# ---------------------------------------------------------------- 4. where each mall / school must be
def locate(name_col, dist_col, label):
    rows = []
    for name, g in b.dropna(subset=[name_col, dist_col]).groupby(name_col):
        if len(g) < 3:
            rows.append({"kind": label, "name": name, "blocks": len(g), "lat": np.nan, "lon": np.nan, "fit_km": np.nan})
            continue
        la, lo, dd = g["latitude"].values, g["longitude"].values, g[dist_col].values / scale
        w = 1 / np.maximum(dd, 0.05)
        x0 = [np.average(la, weights=w), np.average(lo, weights=w)]
        p = least_squares(lambda q: haversine(la, lo, q[0], q[1]) - dd, x0=x0).x
        res = dd - haversine(la, lo, p[0], p[1])
        rows.append({"kind": label, "name": name, "blocks": len(g), "lat": round(p[0], 6), "lon": round(p[1], 6),
                     "fit_km": round(float(np.median(np.abs(res))), 3)})
        bad = g.index[np.abs(res) > TOL]
        flag(bad, f"{label}_distance", [f"{name}: column {b.at[r, dist_col]:.2f} km, fits {haversine(b.at[r, 'latitude'], b.at[r, 'longitude'], p[0], p[1]) * scale:.2f} km" for r in bad])
    return pd.DataFrame(rows)

print("\n4. Mall and school positions worked out from the blocks that name them")
loc = pd.concat([locate(MALLN, MALL, "mall"), locate(SCHN, SCH, "school")], ignore_index=True)
for label, name_col, dist_col in [("mall", MALLN, MALL), ("school", SCHN, SCH)]:
    L = loc[(loc["kind"] == label) & loc["lat"].notna()]
    print(f"   {label}s: {len(L)} located (median fit {L['fit_km'].median():.3f} km); "
          f"{(loc['kind'] == label).sum() - len(L)} named by fewer than 3 blocks (not checked)")
    worst = L.sort_values("fit_km", ascending=False).head(5)
    if len(worst):
        print("   worst fits:", "; ".join(f"{r.name} {r.fit_km} km ({r.blocks} blocks)" for r in worst.itertuples()))
    # is another one clearly nearer than the one named?
    D = haversine(lat[:, None], lon[:, None], L["lat"].values[None, :], L["lon"].values[None, :]) * scale
    named = b[name_col].map(dict(zip(L["name"], range(len(L)))))
    ok = named.notna().values
    idx = named[ok].astype(int).values
    d_named = D[ok][np.arange(ok.sum()), idx]
    j = D[ok].argmin(axis=1)
    closer = (D[ok][np.arange(ok.sum()), j] < d_named - TOL) & (j != idx)
    rows = np.where(ok)[0][closer]
    flag(rows, f"{label}_not_nearest", [f"names {b.at[r, name_col]}, but {L['name'].iat[j[k]]} is about "
                                        f"{D[r, j[k]]:.2f} km vs {D[r, int(named.iat[r])]:.2f} km"
                                        for r, k in zip(rows, np.where(closer)[0])])
    print(f"   blocks where another {label} looks clearly nearer than the one named: {len(rows)}")

# ---------------------------------------------------------------- 5. neighbours (triangle rule)
print(f"\n5. Neighbours within {NEAR * 1000:.0f} m naming the same mall / school")
cell = np.floor(lat / 0.0015).astype(int) * 100000 + np.floor(lon / 0.0015).astype(int)
for label, name_col, dist_col in [("mall", MALLN, MALL), ("school", SCHN, SCH)]:
    n_bad = 0
    for _, g in b.assign(cell=cell).groupby("cell"):
        if len(g) < 2:
            continue
        la, lo, nm, dd = g["latitude"].values, g["longitude"].values, g[name_col].to_numpy(dtype=object), g[dist_col].values / scale
        sep = haversine(la[:, None], lo[:, None], la[None, :], lo[None, :])
        same = (nm[:, None] == nm[None, :]) & (sep < NEAR) & (sep > 0)
        gap = np.abs(dd[:, None] - dd[None, :]) - sep
        ii, jj = np.where(np.triu(same & (gap > TOL)))
        for i, k in zip(ii, jj):
            n_bad += 1
            flag([g.index[i]], f"{label}_neighbour", [f"{dd[i] * scale:.2f} km vs {dd[k] * scale:.2f} km at block "
                                                      f"{g['block'].iat[k]} {g['street_name'].iat[k]}, {sep[i, k] * 1000:.0f} m away"])
    print(f"   {label}: {n_bad} neighbour pairs that break the triangle rule by more than {TOL} km")

# ---------------------------------------------------------------- optional: OneMap
if "--onemap" in sys.argv:
    import requests
    print("\nOneMap: looking up every mall and school name ...")
    def onemap(q):
        for _ in range(3):
            try:
                r = requests.get(ONEMAP, params={"searchVal": q, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1}, timeout=20)
                if r.ok:
                    return r.json().get("results", [])
            except requests.RequestException:
                pass
            time.sleep(1)
        return []
    om_lat, om_lon, om_hit = [], [], []
    for r in loc.itertuples():
        res = onemap(r.name)
        key = r.name.upper().replace("'", "")
        hit = next((x for x in res if key in (x.get("BUILDING", "") + " " + x.get("SEARCHVAL", "")).upper().replace("'", "")), res[0] if res else None)
        om_lat.append(float(hit["LATITUDE"]) if hit else np.nan)
        om_lon.append(float(hit["LONGITUDE"]) if hit else np.nan)
        om_hit.append(hit.get("SEARCHVAL", "") if hit else "not found")
        time.sleep(0.25)
    loc["onemap_lat"], loc["onemap_lon"], loc["onemap_match"] = om_lat, om_lon, om_hit
    loc["fit_vs_onemap_m"] = (haversine(loc["lat"], loc["lon"], loc["onemap_lat"], loc["onemap_lon"]) * 1000).round(0)
    print(f"   found {loc['onemap_lat'].notna().sum()} of {len(loc)} names")
    for label, name_col, dist_col in [("mall", MALLN, MALL), ("school", SCHN, SCH)]:
        L = loc[(loc["kind"] == label) & loc["onemap_lat"].notna()].set_index("name")
        m = b[name_col].isin(L.index)
        true = haversine(b.loc[m, "latitude"].values, b.loc[m, "longitude"].values,
                         L.loc[b.loc[m, name_col], "onemap_lat"].values, L.loc[b.loc[m, name_col], "onemap_lon"].values) * scale
        dif = b.loc[m, dist_col].values - true
        print(f"   {label}: your column vs OneMap position, median difference {np.median(dif):+.3f} km, "
              f"{(np.abs(dif) > TOL).sum():,} of {m.sum():,} blocks off by more than {TOL} km")
        rows = b.index[m][np.abs(dif) > TOL]
        flag(rows, f"{label}_vs_onemap", [f"{b.at[r, name_col]}: column {b.at[r, dist_col]:.2f} km, OneMap {t:.2f} km"
                                          for r, t in zip(rows, true[np.abs(dif) > TOL])])

# ---------------------------------------------------------------- report
loc.to_csv(OUT / "mall_school_locations.csv", index=False)
rep = pd.DataFrame(issues, columns=["row", "check", "detail"])
if len(rep):
    rep = b.loc[rep["row"], ["block", "street_name", "town", "latitude", "longitude"]].reset_index(drop=True).join(rep[["check", "detail"]].reset_index(drop=True))
rep.to_csv(OUT / "block_lookup_check.csv", index=False)
print("\nSummary: blocks flagged by each check")
print(rep["check"].value_counts().to_string() if len(rep) else "   none: every check passed")
n_blocks = rep[["block", "street_name"]].drop_duplicates().shape[0] if len(rep) else 0
print(f"\n{n_blocks:,} of {len(b):,} blocks ({n_blocks / len(b):.1%}) have at least one flag -> hdb_data/block_lookup_check.csv")
print("Mall and school positions -> hdb_data/mall_school_locations.csv")
