# HDB Resale Price Forecaster — Technical Overview

Team 6, NTU SCTP AI Module 3 Project Assignment.

This folder contains the current Python modelling code for an explainable, time-aware HDB resale-price forecaster. The project combines public HDB resale transactions with lease, market, location and policy features, validates chronologically, and compares interpretable linear models with tree and hybrid alternatives.

## 1. Objective

The target is HDB resale price in S$000. The system is designed for decision support rather than automated valuation:

- estimate an expected resale price for a selected flat and sale year;
- explain the main price contributions;
- replay what a model trained at an earlier point in time would have known;
- compare transparent linear models with tree and hybrid models;
- test policy, location and market-driver hypotheses without presenting associations as causal effects.

## 2. Data foundation

The transaction pipeline combines five public HDB resale datasets from data.gov.sg, covering Jan 1990 to Oct 2026 (987,450 transactions in the project snapshot).

The main model uses 2001 onward because the market-driver history begins around 2000 and lagged inputs must be known before the sale date.

Additional inputs include:

- prior-year household income;
- HDB supply / RFS-BTO series and flats sold, lagged four years where selected;
- block coordinates and distances to CBD, malls and popular primary schools;
- MRT/LRT announcement and opening dates;
- housing-policy rounds and their announced timing;
- separate global-event tables for event-study / scenario analysis.

Large/local runtime inputs are intentionally not committed. See `hdb_data/README.md` and run `preflight.py` before a full rebuild.

## 3. Main interpretable model (Plane S)

Plane S is a linear regression whose explanatory features carry the domain structure:

- flat: floor area, storey, town, flat type and flat model;
- lease: flat age plus hinge terms every five years from age 10 to 55;
- market: prior-year income, supply lag 4 and flats-sold lag 4;
- location: CBD, dated mall/school features and dynamic MRT features;
- policy: a 36-month announced-round stance (`tight36`).

The lease hinges allow a transparent piecewise-linear decay curve instead of forcing one age slope over the whole lease.

### Current headline results

- Plane S validation (2015–2019 expanding-window mean MAE): about **S$52.4k**.
- Plane S 2020–2026 out-of-time benchmark after dated mall/school rerun: about **S$75.3k**.
- Tuned dollar hybrid validation MAE: **S$38.6k**.
- Tuned 127-leaf hybrid benchmark MAE: about **S$65.0k**.
- A later dated-location rerun of the default hybrid is about **S$64.7k**; the tuned model remains the honest procedure-selected result because switching after inspecting the benchmark would be test-set selection.

The 2020–2026 period is a benchmark, not a pristine sealed test, because earlier model-governance choices were influenced after looking at it. The clean prospective test is future 2027 sales with rules frozen in advance.

## 4. Time-aware validation

Forecast validation uses expanding windows:

- train through 2014 → validate 2015;
- train through 2015 → validate 2016;
- …;
- train through 2018 → validate 2019;
- train through 2019 → benchmark 2020–2026.

This protects the time arrow: a later sale is not allowed to teach an earlier prediction.

Random splits are used only for separate same-period valuation comparisons, not as the primary future-forecast validation.

## 5. Location and MRT timing

Dynamic MRT features are rebuilt for each sale month:

- `mrt_open_km`: nearest station already open, capped at 3 km;
- `mrt_new_gain`: how much closer an announced-but-not-open station will be, zero if it is not actually nearer;
- `stage_early` / `stage_late`: where the sale falls in the announcement-to-opening wait.

Current fitted location-model effects are approximately:

- open MRT distance: **−S$38k per additional km**;
- announced future MRT gain: **+S$34k per km closer**.

A station cannot affect a sale before its announcement. Malls are counted from their opening month, and schools from the month they started at their home site. One remaining caveat is that the project's high-demand-school list is based on the current compiled list rather than a fully historical annual popularity list.

## 6. Policy stance

`tight36` is a predictive market-regime signal:

- cooling rounds have direction +1;
- easing rounds have direction −1;
- hand intensity is 1–3;
- for each sale month, the feature sums direction × intensity for rounds announced in the preceding 36 months;
- a round announced in the sale month itself does not count yet.

The fitted coefficient is positive (about +S$5k per stance point). This must **not** be interpreted as cooling policy causing prices to rise. Cooling rounds are usually introduced when the market is already hot, so reverse causality / endogeneity is expected.

The alternative `S+Z` adds a continuous 48–84 month lagged stance. It improved the inspected 2020–2026 benchmark but lost on the 2015–2019 selection period (about S$55.5k vs S at S$52.4k), so it was not adopted. The intended next test is a single prospectively defined 2027 rematch.

## 7. Trees and hybrid model

Pure tree models performed well at same-period valuation but poorly at extrapolating the post-2020 market level.

The dollar hybrid therefore keeps Plane S for the market-level forecast and trains LightGBM only on the plane's residuals. This preserves an interpretable market trend while letting the tree correct flat/location detail.

The selected tuning run used 127 leaves and achieved about S$38.6k mean MAE on 2015–2019 expanding-window validation. Sensible tuning variants were within roughly S$1k of one another, so the architecture mattered more than fine tuning.

## 8. Course benchmark / ablation

`hdb_course_compare.py` tests the project's extra feature groups on the course repo's own 2017–2019 data while holding rows/models/splits constant. Feature groups are added one at a time (flat model, lease bends, location, MRT, time trend) so gains can be attributed to the added information rather than a changed test set.

The largest improvement came from CBD / mall / school location features. On the course comparison, LightGBM error fell from about S$24.3k to S$17.6k on the random split and from about S$27.6k to S$18.7k on the chronological 2019 test after the full feature set.

## 9. Known limitations

- 2020–2026 is an inspected benchmark, not a sealed test.
- Rare flat models are grouped using counts computed before the folds; fold-local grouping would be cleaner.
- The popular-school list is not a complete historical annual popularity series.
- Renovation, orientation, unit position and block upgrading are not available.
- Policy coefficients are predictive associations, not causal estimates.
- Global-event scenario effects can overlap with drivers already represented in the formula and should be treated as what-if overlays.
- Full reproduction requires local data files intentionally excluded from GitHub.

## 10. Scripts

### Core pipeline

- `hdb_resale_1990_2026.py` — download/combine HDB resale transactions.
- `hdb_pairs.py` — two-input linear planes for visualisation.
- `hdb_drivers_model.py` — macro/supply driver construction and back-tests.
- `hdb_location_model.py` — location, lease, MRT and policy features; Plane S and related candidates.
- `hdb_tree_models.py` — RF / gradient boosting / XGBoost / LightGBM comparisons.
- `hdb_hybrid_cv.py` — plane + residual LightGBM hybrid and tuning grid.
- `hdb_explain_models.py` — model explanations and error analysis.
- `hdb_replay_models.py` — historical refits / replay models.

### Tests and analyses

- `hdb_lease_rule_cv.py` — July 2013 older-flat rule candidates.
- `hdb_age_gap_check.py` — descriptive age-gap checks.
- `hdb_policy_stance_cv.py` — policy stance robustness and lag variants.
- `hdb_unsupervised_cv.py` — K-means / Isolation Forest / PCA tests inside folds.
- `hdb_course_compare.py` — course-data feature ablation comparison.

### Block / MRT utilities

- `build_block_lookup.py`
- `fill_missing_blocks.py`
- `check_block_lookup.py`
- `fix_filled_blocks.py`
- `mrt/build_mrt_stations.py`

## 11. Run locally

From the repository root:

```bash
python module3_hdb/preflight.py
```

Then follow `../RUN_IN_VSCODE.md`.

## 12. Repository scope

This public repository intentionally excludes private correspondence, superseded ZIP bundles, Power BI files, presentation drafts, temporary logs and large local/generated datasets. Those exclusions are deliberate source-control hygiene, not missing model logic.
