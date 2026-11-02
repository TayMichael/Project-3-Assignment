from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "hdb_data"

PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "requests": "requests",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "openpyxl": "openpyxl",
}

CORE_LOCAL = [
    "hdb_block_lookup_from_V3F.csv",
    "bto-2001-2024_with_RFS_WIS_residual_2000_2004_reconstructed.xlsx",
    "Singapore_Private_Residential_Supply_Definitive_Merged_1995_2026_EC_History.xlsx",
]
LATEST_EXACT = [
    "mall_school_locations.csv",
    "malls_dated.csv",
    "school_home_sites.csv",
]

print(f"Python: {sys.version.split()[0]}")
print(f"Project: {ROOT}")

missing_pkg = []
for module, package in PACKAGES.items():
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "installed")
        print(f"[OK] package {package}: {version}")
    except Exception:
        print(f"[MISSING] package {package}")
        missing_pkg.append(package)

DATA.mkdir(exist_ok=True)

mrt_src = ROOT / "mrt" / "mrt_stations_corrected.csv"
mrt_dst = DATA / "mrt_stations_corrected.csv"
if not mrt_dst.exists() and mrt_src.exists():
    shutil.copy2(mrt_src, mrt_dst)
    print("[FIXED] copied mrt/mrt_stations_corrected.csv -> hdb_data/")
elif mrt_dst.exists():
    print("[OK] hdb_data/mrt_stations_corrected.csv")
else:
    print("[MISSING] MRT corrected station table")

missing_core = []
for name in CORE_LOCAL:
    if (DATA / name).exists():
        print(f"[OK] core data {name}")
    else:
        print(f"[MISSING] core data {name}")
        missing_core.append(name)

missing_latest = []
for name in LATEST_EXACT:
    if (DATA / name).exists():
        print(f"[OK] latest dated-location data {name}")
    else:
        print(f"[MISSING-LATEST] dated-location data {name}")
        missing_latest.append(name)

print("\nRESULT")
if missing_pkg:
    print("BLOCKED: install missing packages with: python -m pip install -r ../requirements.txt")
elif missing_core:
    print("BLOCKED FOR FULL REBUILD: add the missing core local files listed above.")
    print("The scripts themselves are present; hdb_resale_1990_2026.py can still download the raw resale history when internet access is available.")
else:
    print("CORE PREFLIGHT PASS: required packages and core local source files are present.")
    if missing_latest:
        print("LATEST-MODEL WARNING: the scripts can fall back to undated mall/school features, but the current dated-location model cannot be reproduced exactly until the MISSING-LATEST files are added.")
    else:
        print("LATEST-MODEL PASS: dated mall/school inputs are present.")
