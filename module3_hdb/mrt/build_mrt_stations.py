"""Build one row per MRT station: first announcement, first opening, coordinates.

Inputs (same folder):
  mrt_station_phase_timeline_master_v4.csv   your timeline (one row per station per line)
  mrt_reference_current_plus_2030.csv        your OneMap geocodes (sheet 1 of the xlsx)
  legacy_lines_wikipedia.csv                 station opening dates, NSL/EWL/CG/NEL/CCL (Wikipedia list)
  station_coords_fetched.csv                 coordinates for stations missing or wrongly geocoded
Output:
  mrt_stations_corrected.csv
"""
import sys
from pathlib import Path
import pandas as pd

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
t = pd.read_csv(D / "mrt_station_phase_timeline_master_v4.csv")
ref = pd.read_csv(D / "mrt_reference_current_plus_2030.csv")
leg = pd.read_csv(D / "legacy_lines_wikipedia.csv")
fet = pd.read_csv(D / "station_coords_fetched.csv")

key = lambda s: s.str.upper().str.strip()
t["key"] = key(t["station_name_key"])
t["ann"] = pd.to_datetime(t["announcement_date"])
t["opn"] = pd.to_datetime(t["opening_date"])
leg["key"] = key(leg["station"]) + " MRT"
leg["opn"] = pd.to_datetime(leg["opening_date"])

# Lines whose timeline rows carry one placeholder date per line: take the station's own date from the list.
LEGACY = {"NSL", "EWL", "NEL", "CCL"}
ann_by_line = t[t["line_name"].isin(LEGACY)].groupby("line_name")["ann"].min()   # your announcement dates
LINE_OF = {"NS": "NSL", "EW": "EWL", "CG": "EWL", "NE": "NEL", "CC": "CCL"}
leg["line_name"] = leg["code"].str[:2].map(LINE_OF)
leg["ann"] = leg["line_name"].map(ann_by_line)
# a branch or infill keeps your own announcement date when your timeline has that station
own = t[t["line_name"].isin(LEGACY)].groupby("key")["ann"].min()
leg["ann"] = leg["key"].map(own).fillna(leg["ann"])
# Expo opened with the Changi Airport branch: use that branch's announcement date
leg.loc[leg["code"] == "CG1", "ann"] = own.get("CHANGI AIRPORT MRT", leg.loc[leg["code"] == "CG1", "ann"].iloc[0])

rows = pd.concat([
    t.loc[~t["line_name"].isin(LEGACY), ["key", "line_name", "ann", "opn"]].assign(open_src="your timeline"),
    leg[["key", "line_name", "ann", "opn"]].assign(open_src="Wikipedia station list"),
])
first = rows.sort_values("opn").groupby("key").first()        # the line that opened first at this station
st = pd.DataFrame({
    "station": first.index,
    "lines": rows.groupby("key")["line_name"].agg(lambda s: "/".join(sorted(set(s)))),
    "first_line": first["line_name"],
    "announcement_date": first["ann"].dt.date,
    "opening_date": first["opn"].dt.date,
    "opening_source": first["open_src"],
}).reset_index(drop=True)

# coordinates: your geocodes, except wrong matches and NOT_FOUND; then the fetched list
ref["key"] = key(ref["name"])
bad = ref["status"].ne("OK") | ref["key"].isin(
    ["ORCHARD MRT", "TAMPINES MRT", "PUNGGOL MRT", "WEST COAST MRT", "SERANGOON NORTH MRT"])
good = ref[~bad].set_index("key")[["latitude", "longitude"]].assign(coord_source="your OneMap geocode")
fet["key"] = key(fet["station_name"])
fx = fet.dropna(subset=["latitude", "longitude"]).set_index("key")
fx = fx[["latitude", "longitude", "source_url"]].rename(columns={"source_url": "coord_source"})
coords = pd.concat([fx, good[~good.index.isin(fx.index)]])
st = st.join(coords, on="station")
st["station"] = st["station"].str.title().str.replace(" Mrt", " MRT").str.replace("Harbourfront", "HarbourFront") \
    .str.replace("Macpherson", "MacPherson").str.replace("One-North", "one-north")

out = D / "mrt_stations_corrected.csv"
st.sort_values("opening_date").to_csv(out, index=False)
print(f"{len(st)} stations -> {out}")
print("no coordinates:", st.loc[st["latitude"].isna(), "station"].tolist())
