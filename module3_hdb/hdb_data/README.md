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

## Verification performed on 2 Nov 2026

Environment / preflight status:

- Python 3.11.18 in Conda environment `dsai-m3` — PASS.
- Required Python packages imported successfully, including LightGBM and XGBoost — PASS.
- `preflight.py` completed and copied `mrt/mrt_stations_corrected.csv` into `hdb_data/` — PASS.
- Full rebuild remains blocked only by the six local input files listed above.

Test 1 — `hdb_resale_1990_2026.py`:

- PASS.
- Downloaded all five official HDB resale datasets successfully.
- Combined dataset: 988,148 transactions from 1990-01 through 2026-10.
- Generated local runtime outputs including the five raw dataset CSVs, `hdb_resale_all.csv`, and `plane_by_year.csv`.
- Recent-plane fit covered 100,427 transactions from 2023-01 through 2026-10.
- Recent-plane result: beta = 236.7 $k; omega1 = 5.28 $k per m2; omega2 = -4.44 $k per year of age; R2 = 0.533.

These generated outputs are intentionally not committed to GitHub because `module3_hdb/hdb_data/*` is ignored except for this README. They can be reproduced by rerunning the script with internet access.
