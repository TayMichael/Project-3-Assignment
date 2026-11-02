# Project 3 Assignment — HDB Resale Price Forecaster

NTU SCTP AI Module 3 team project: an explainable, time-aware machine-learning forecaster for Singapore HDB resale prices.

## Repository contents

- `module3_hdb/` — current Team 6 Python modelling scripts, policy/global-event CSVs, MRT source tables, preflight check and the detailed technical README.
- `RUN_IN_VSCODE.md` — local setup and execution route.
- `requirements.txt` — Python dependencies imported by the supplied scripts.

## Scope and exclusions

This repository intentionally excludes:

- WhatsApp/chat exports or other private correspondence;
- superseded/duplicate Python ZIP bundles;
- temporary logs, assembly notes and working files;
- Power BI files while the refreshed package is still pending;
- presentation slides while the final deck is still being consolidated;
- large binary source/audit PDFs in this initial code-first commit (the detailed technical record remains in `module3_hdb/README.md`);
- large/local runtime data under `module3_hdb/hdb_data/` except its README placeholder.

The latest supplied code bundle is the 2 Nov 2026 `team6_scripts.zip`; older duplicate code is not retained here.

## Notebook status

The latest supplied bundle contains Python `.py` scripts rather than Jupyter `.ipynb` notebooks. No notebook has been fabricated or converted solely for repository appearance.

## Reproducibility note

The repository is not fully self-contained because several large/local source files are deliberately excluded. Run:

```bash
python module3_hdb/preflight.py
```

before rebuilding the model. See [`module3_hdb/hdb_data/README.md`](module3_hdb/hdb_data/README.md) for required local inputs.

## Quick start

See [`RUN_IN_VSCODE.md`](RUN_IN_VSCODE.md).

For the detailed model design, assumptions, validation results, limitations and audit notes, see [`module3_hdb/README.md`](module3_hdb/README.md).
