"""Fill the blocks that hdb_location_model.py could not match.

Reads  hdb_data/hdb_block_lookup_from_V3F.csv, hdb_data/unmatched_blocks.csv (written by hdb_location_model.py)
       and hdb_data/hdb_resale_all.csv (for each block's town)
Writes hdb_data/hdb_block_lookup_filled.csv  (your lookup + filled rows; the original is not changed)
       hdb_data/filled_blocks_report.csv    (one row per filled or failed block, with how it was filled)

1. Demolished / SERS blocks (in your lookup, NOT_FOUND):
   copy the location of the nearest surviving block on the SAME street (closest block number).
   If no block on that street survives, geocode the street itself with OneMap.
   fill_method = "neighbour block <n>" or "street centre (OneMap)"
2. New blocks (not in your lookup): geocode the address with OneMap (block number must match).
   CBD / mall / school distances are copied from the nearest geocoded block (usually < 200 m away).
   fill_method = "OneMap", fill_note says which block the distances came from.
MRT distances are not copied: hdb_location_model.py recomputes them from latitude/longitude.
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path("hdb_data")
ONEMAP = "https://www.onemap.gov.sg/api/common/elastic/search"
ABBR = {"AVE": "AVENUE", "BT": "BUKIT", "RD": "ROAD", "ST": "STREET", "CRES": "CRESCENT", "DR": "DRIVE",
        "NTH": "NORTH", "STH": "SOUTH", "CL": "CLOSE", "LOR": "LORONG", "JLN": "JALAN", "C'WEALTH": "COMMONWEALTH",
        "TG": "TANJONG", "KG": "KAMPONG", "UPP": "UPPER", "PL": "PLACE", "CTRL": "CENTRAL", "GDNS": "GARDENS",
        "HTS": "HEIGHTS", "TER": "TERRACE", "PK": "PARK", "MKT": "MARKET", "CTR": "CENTRE", "HG": "HOUGANG"}
norm = lambda s: s.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)


def full_street(street):
    # "ST. GEORGE'S LANE" keeps "ST."; "CANBERRA ST" -> "CANBERRA STREET"
    return " ".join(w if w.endswith(".") else ABBR.get(w, w) for w in street.split())


def onemap(query):
    for attempt in range(3):
        try:
            r = requests.get(ONEMAP, params={"searchVal": query, "returnGeom": "Y",
                                             "getAddrDetails": "Y", "pageNum": 1}, timeout=20)
            return r.json().get("results", [])
        except Exception:
            time.sleep(2)
    return []


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


blk = pd.read_csv(OUT / "hdb_block_lookup_from_V3F.csv", dtype=str)
blk["block"], blk["street_name"] = norm(blk["block"]), norm(blk["street_name"])
# unmatched pairs from hdb_location_model.py, split into demolished (in your lookup, no coordinates) and new
um = pd.read_csv(OUT / "unmatched_blocks.csv", dtype=str)
um["block"], um["street_name"] = norm(um["block"]), norm(um["street_name"])
sales = pd.read_csv(OUT / "hdb_resale_all.csv", usecols=["block", "street_name", "town"], dtype=str)
sales["block"], sales["street_name"] = norm(sales["block"]), norm(sales["street_name"])
town = sales.drop_duplicates(["block", "street_name"])
split = um[["block", "street_name"]].merge(town, on=["block", "street_name"], how="left")
in_lookup = split.set_index(["block", "street_name"]).index.isin(blk.set_index(["block", "street_name"]).index)
split["group"] = np.where(in_lookup, "in lookup, no coordinates (demolished / SERS?)", "not in lookup (new block?)")
split.to_csv(OUT / "unmatched_blocks_split.csv", index=False)

good = blk[blk["latitude"].notna()].copy()
good["lat"], good["lon"] = good["latitude"].astype(float), good["longitude"].astype(float)
COPY = [c for c in blk.columns if c not in ("block", "street_name", "town", "postal_code", "latitude", "longitude",
                                              "matched_address", "building", "status", "normalized_address")]
# columns about the MRT are recomputed by the model from lat/lon; copy only the others
STATIC = [c for c in COPY if "mrt" not in c.lower()]
blk_no = lambda b: int(re.match(r"\d+", b).group()) if re.match(r"\d+", b) else 10 ** 6

report, new_rows = [], []

# ---------------------------------------------------------------- 1. demolished / SERS
dem = split[split["group"].str.startswith("in lookup")]
for _, r in dem.iterrows():
    same = good[good["street_name"] == r["street_name"]]
    if len(same):
        nb = same.iloc[(same["block"].map(blk_no) - blk_no(r["block"])).abs().argsort().iloc[0]]
        row = {**{c: nb[c] for c in COPY}, "latitude": nb["latitude"], "longitude": nb["longitude"],
               "postal_code": None, "fill_method": f"neighbour block {nb['block']}", "fill_note": "approx_coord"}
    else:
        res = onemap(full_street(r["street_name"]))
        if not res:
            report.append({**r[["block", "street_name", "town", "group"]].to_dict(), "fill_method": "FAILED",
                           "fill_note": "no surviving block on street and OneMap found no street"})
            continue
        lat, lon = float(res[0]["LATITUDE"]), float(res[0]["LONGITUDE"])
        d = haversine(lat, lon, good["lat"].values, good["lon"].values)
        nb = good.iloc[int(d.argmin())]
        row = {**{c: nb[c] for c in STATIC}, "latitude": lat, "longitude": lon, "postal_code": None,
               "fill_method": "street centre (OneMap)",
               "fill_note": f"approx_coord; {res[0]['SEARCHVAL']}; distances from {nb['block']} {nb['street_name']} ({d.min():.2f} km)"}
    row.update({"block": r["block"], "street_name": r["street_name"], "town": r["town"], "status": "FILLED"})
    new_rows.append(row)
    report.append({**r[["block", "street_name", "town", "group"]].to_dict(),
                   "fill_method": row["fill_method"], "fill_note": row["fill_note"]})

# ---------------------------------------------------------------- 2. new blocks
new = split[split["group"].str.startswith("not")]
print(f"Geocoding {len(new)} new blocks with OneMap (about {len(new) // 4 + 1} s)...")
for i, (_, r) in enumerate(new.iterrows(), 1):
    hit = None
    for q in (f"{r['block']} {full_street(r['street_name'])}", f"{r['block']} {r['street_name']}"):
        hit = next((x for x in onemap(q) if x.get("BLK_NO", "").upper() == r["block"]), None)
        if hit:
            break
    time.sleep(0.25)                                   # stay well under OneMap's rate limit
    if not hit:
        report.append({**r[["block", "street_name", "town", "group"]].to_dict(), "fill_method": "FAILED",
                       "fill_note": "OneMap returned no result with this block number"})
        continue
    lat, lon = float(hit["LATITUDE"]), float(hit["LONGITUDE"])
    d = haversine(lat, lon, good["lat"].values, good["lon"].values)
    nb = good.iloc[int(d.argmin())]
    row = {**{c: nb[c] for c in STATIC}, "block": r["block"], "street_name": r["street_name"], "town": r["town"],
           "postal_code": hit.get("POSTAL"), "latitude": lat, "longitude": lon, "matched_address": hit.get("ADDRESS"),
           "status": "FILLED", "fill_method": "OneMap",
           "fill_note": f"distances from {nb['block']} {nb['street_name']} ({d.min():.2f} km)"}
    new_rows.append(row)
    report.append({**r[["block", "street_name", "town", "group"]].to_dict(),
                   "fill_method": "OneMap", "fill_note": row["fill_note"]})
    if i % 50 == 0:
        print(f"  {i}/{len(new)}")

# ---------------------------------------------------------------- save
filled = pd.DataFrame(new_rows)
keep = blk[~blk.set_index(["block", "street_name"]).index.isin(filled.set_index(["block", "street_name"]).index)]
outp = pd.concat([keep.assign(fill_method="V3F"), filled], ignore_index=True)[list(blk.columns) + ["fill_method", "fill_note"]]
outp.to_csv(OUT / "hdb_block_lookup_filled.csv", index=False)
rep = pd.DataFrame(report)
rep.to_csv(OUT / "filled_blocks_report.csv", index=False)
print("\n", rep.groupby(["group", rep["fill_method"].str.split(" ").str[0]]).size().to_string())
print(f"\nSaved hdb_data/hdb_block_lookup_filled.csv ({len(outp):,} blocks) and hdb_data/filled_blocks_report.csv")
if (rep["fill_method"] == "FAILED").any():
    print("\nNot filled:")
    print(rep[rep["fill_method"] == "FAILED"][["block", "street_name", "town", "fill_note"]].to_string(index=False))
