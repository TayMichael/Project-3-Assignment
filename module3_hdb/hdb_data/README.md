# Local runtime data (not committed)

The modelling scripts deliberately keep generated / large data out of GitHub.
Create or copy the required files into this folder before running the full pipeline.

## Core inputs not bundled in `team6_scripts.zip`

- `hdb_block_lookup_from_V3F.csv` — required by `hdb_location_model.py`
- `bto-2001-2024_with_RFS_WIS_residual_2000_2004_reconstructed.xlsx` — required by `hdb_drivers_model.py`
- `Singapore_Private_Residential_Supply_Definitive_Merged_1995_2026_EC_History.xlsx` — required by `hdb_drivers_model.py`

## Inputs required to reproduce the latest dated-location model exactly

- `mall_school_locations.csv`
- `malls_dated.csv`
- `school_home_sites.csv`

If these three files are absent, `hdb_location_model.py` falls back to the older present-day mall/school lookup. That fallback can run, but it reintroduces the historical look-ahead issue that the 31 Oct / 1 Nov update corrected.

## MRT station table

The repo includes `../mrt/mrt_stations_corrected.csv`. Before running `hdb_location_model.py`, copy it here as:

`hdb_data/mrt_stations_corrected.csv`

## Generated/downloaded by the scripts

Examples include `hdb_resale_all.csv`, `macro_annual.csv`, CV/back-test CSVs, charts and explanatory JSON files.
They remain local and are excluded by `.gitignore`.
