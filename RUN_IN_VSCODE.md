# Run Project 3 in VS Code

## 1. Clone and open the repository

```bash
git clone https://github.com/TayMichael/Project-3-Assignment.git
cd Project-3-Assignment
code .
```

## 2. Activate the course environment

If the existing course environment is available:

```bash
conda activate dsai-m3
```

Otherwise use a Python environment and install:

```bash
python -m pip install -r requirements.txt
```

## 3. Run the preflight check

From the repository root:

```bash
python module3_hdb/preflight.py
```

The preflight reports missing packages and local data files before a long model run starts.

## 4. Supply local runtime inputs

Place the non-repository source files in `module3_hdb/hdb_data/` as listed in:

`module3_hdb/hdb_data/README.md`

To reproduce the latest dated-location model exactly, the dated mall/school inputs listed there must also be present. Without them the location script can fall back to older undated features, which is not the same model.

## 5. Run the modelling scripts

```bash
cd module3_hdb
python hdb_resale_1990_2026.py
python hdb_pairs.py
python hdb_drivers_model.py
python hdb_location_model.py
python hdb_tree_models.py
python hdb_hybrid_cv.py --grid
python hdb_explain_models.py
python hdb_replay_models.py
python hdb_lease_rule_cv.py
python hdb_age_gap_check.py
python hdb_policy_stance_cv.py
python hdb_unsupervised_cv.py
```

Optional course comparison:

```bash
python hdb_course_compare.py <course hdb_resale_2017_2019.csv> <Blocks.csv> <MRT_Stations.csv>
```

## Verification status

- **SELF-PROVEN:** supplied Python files compile successfully in the staging environment.
- **PARTIALLY PROVEN:** dependencies and required filenames were derived from the supplied scripts and README.
- **HOST-DEPENDENT-NOT-REPRODUCIBLE HERE:** a full model rebuild needs network access and local data files intentionally excluded from GitHub.
