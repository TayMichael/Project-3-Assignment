"""Lesson 3.5 (unsupervised learning) applied to model S  (an experiment; the page model is not changed)

Three lesson 3.5 tools, each tried as a way to improve the page model S, under the same rule as every candidate:

  KM8/16/32     S + k-means LOCATION SEGMENTS. Every block is clustered on its location alone (latitude, longitude,
                 km to the CBD, to a mall and to a popular primary school; standardised), into 8, 16 or 32 segments;
                 S gets one 0/1 column per segment. Asks: do natural location groups add anything beyond the 26
                 towns and the distances S already has?
  ISO01, ISO1    S trained WITHOUT the sales Isolation Forest finds most unusual (0.1% or 1% of the training sales),
                 on price per m2 vs the same year, town and flat type, area vs the flat type, age and storey.
                 Only TRAINING sales are removed: every validation and test sale is still predicted and scored.
                 Asks: does cleaning out unusual sales make the forecast better?
  PCA95          S with its numeric columns replaced by PCA components that keep 95% of their variation
                 (standardised first; the town and flat dummies stay). Asks: does reducing dimensions help?

No leakage: in every fold the k-means segments, the Isolation Forest and the PCA are fitted on that fold's
training years only, then applied to the year being predicted.

Rule, set before the run: time-series cross-validation on 2015-2019 (each year predicted from the years before it);
a candidate replaces S only if it beats S by PREFER_MARGIN ($0.5k) on the validation mean. Then the 2020-2026
out-of-time benchmark is scored once for all (the others for the record).

Runs hdb_location_model.py unchanged first, about 6-8 minutes, then about 15-25 more. Prints only; nothing is written.
    conda activate dsai-m3
    python hdb_unsupervised_cv.py
"""
import runpy

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

print("Running hdb_location_model.py (unchanged) to build the sales and features ...")
g = runpy.run_path("hdb_location_model.py", run_name="unsupervised")
hdb, design, ALL = g["hdb"], g["design"], g["ALL"]
VAL_YEARS, TEST_FROM, MARGIN = g["VAL_YEARS"], g["TEST_FROM"], g["PREFER_MARGIN"]
skey = next(n for n in ALL if n.startswith("S"))
scols, sflat, _ = ALL[skey]
hdb = hdb.reset_index(drop=True)

LOC_K = ["latitude", "longitude", "dist_to_cbd_km", "dist_to_mall_km", "dist_to_nearest_high_demand_primary_school_km"]
IF_F = ["price_vs_peers", "area_vs_type", "flat_age", "storey"]
y, yr = hdb["price_k"].values, hdb["year"].values
XS = design(scols, sflat, hdb)                        # model S's design matrix: numeric columns + town/flat dummies
assert list(XS.columns[:len(scols)]) == list(scols)    # design() puts S's numeric columns first, then the dummies
XS = XS.to_numpy(dtype=float)
inum = list(range(len(scols)))
idum = list(range(len(scols), XS.shape[1]))
miss = hdb[LOC_K].isna().any(axis=1)
if miss.any():
    print(f"{int(miss.sum()):,} sales lack a location value for k-means; they get the median (segments only)")
    hdb[LOC_K] = hdb[LOC_K].fillna(hdb[LOC_K].median())
blk = hdb["latitude"].round(5).astype(str) + "," + hdb["longitude"].round(5).astype(str)   # one key per block location

# Isolation Forest features: price per m2 vs peers (same year, town, flat type), area vs flat type
psm = hdb["price_k"] / hdb["floor_area_sqm"]
hdb["price_vs_peers"] = np.log(psm / psm.groupby([hdb["year"], hdb["town"], hdb["flat_type"]]).transform("median"))
hdb["area_vs_type"] = np.log(hdb["floor_area_sqm"] / hdb.groupby("flat_type")["floor_area_sqm"].transform("median"))
# (peer medians come from each sale's own year: a training sale is compared only with sales of its own year)


def fit_predict(X, tr, pr, keep=None):
    t = tr if keep is None else (tr & keep)
    return LinearRegression().fit(X[t], y[t]).predict(X[pr])


def folds():
    for Y in VAL_YEARS + [TEST_FROM]:
        yield Y, yr < Y, (yr == Y) if Y != TEST_FROM else (yr >= TEST_FROM)


pred, notes = {}, {}
# ---------------------------------------------------------------- S
pred["S  page model"] = {Y: (fit_predict(XS, tr, pr), pr) for Y, tr, pr in folds()}
print("S done")

# ---------------------------------------------------------------- K: k-means location segments
for k in (8, 16, 32):
    name = f"KM{k} S + {k} k-means location segments"
    pred[name], notes[name] = {}, []
    for Y, tr, pr in folds():
        b_tr = hdb.loc[tr, LOC_K].groupby(blk[tr]).first()                 # blocks sold in the training years
        sc = StandardScaler().fit(b_tr)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(sc.transform(b_tr))
        seg = km.predict(sc.transform(hdb[LOC_K]))                         # every sale -> nearest segment centre
        D = np.eye(k)[seg][:, 1:]                                          # one 0/1 column per segment, first dropped
        pred[name][Y] = (fit_predict(np.hstack([XS, D]), tr, pr), pr)
        notes[name].append(f"{Y}: {len(b_tr):,} blocks, sizes {np.bincount(km.labels_).min()}-{np.bincount(km.labels_).max()}")
        del D
    print(f"KM{k} done")

# ---------------------------------------------------------------- IF: train without the most unusual training sales
for share in (0.001, 0.01):
    name = ("ISO01" if share == 0.001 else "ISO1") + f" S trained without the {share:.1%} most unusual sales"
    pred[name], notes[name] = {}, []
    for Y, tr, pr in folds():
        F = hdb.loc[tr, IF_F].to_numpy(float)
        sc = StandardScaler().fit(F)
        iso = IsolationForest(n_estimators=200, contamination=share, random_state=0, n_jobs=-1).fit(sc.transform(F))
        keep = np.ones(len(hdb), bool)
        keep[np.where(tr)[0][iso.predict(sc.transform(F)) == -1]] = False
        pred[name][Y] = (fit_predict(XS, tr, pr, keep), pr)
        notes[name].append(f"{Y}: removed {int((~keep).sum()):,} of {int(tr.sum()):,} training sales")
    print(f"{name.split()[0]} done")

# ---------------------------------------------------------------- PCA95: PCA on S's numeric columns
name = "PCA95 S with numeric columns as PCA components (95%)"
pred[name], notes[name] = {}, []
for Y, tr, pr in folds():
    sc = StandardScaler().fit(XS[tr][:, inum])
    pca = PCA(n_components=0.95, random_state=0).fit(sc.transform(XS[tr][:, inum]))
    Z = np.hstack([pca.transform(sc.transform(XS[:, inum])), XS[:, idum]])
    pred[name][Y] = (fit_predict(Z, tr, pr), pr)
    notes[name].append(f"{Y}: {len(inum)} numeric columns -> {pca.n_components_} components")
    del Z
print("PCA95 done")

# ---------------------------------------------------------------- choose on validation, then the 2020-2026 benchmark once
cv = {n: [float(np.abs(p[Y][0] - y[p[Y][1]]).mean()) for Y in VAL_YEARS] for n, p in pred.items()}
tab = pd.DataFrame(cv, index=VAL_YEARS).T.round(1)
tab["mean"] = tab.mean(axis=1).round(1)
print(f"\nValidation MAE ($k), each year predicted from the years before it ({TEST_FROM} on not used):")
print(tab.to_string())
means = tab["mean"]
best = means.idxmin()
chosen = best if best != "S  page model" and means["S  page model"] - means[best] >= MARGIN else "S  page model"
print(f"\nBest validation mean: {best.split()[0]} ${means[best]:.1f}k; S ${means['S  page model']:.1f}k "
      f"(a new model must beat S by ${MARGIN}k). Chosen: {chosen}")

rows = []
for n, p in pred.items():
    pt, m = p[TEST_FROM]
    e = pt - y[m]
    rows.append({"model": n, "chosen": n == chosen, "cv_mean_k": means[n],
                 "test_mae_k": round(float(np.abs(e).mean()), 1), "test_rmse_k": round(float(np.sqrt((e ** 2).mean())), 1),
                 "test_mape_pct": round(float((np.abs(e) / y[m]).mean() * 100), 1), "test_bias_k": round(float(e.mean()), 1)})
pd.set_option("display.width", 220)
print(f"\nOut-of-time benchmark {TEST_FROM}-{int(yr.max())}, scored once (the choice above was made before this):")
print(pd.DataFrame(rows).to_string(index=False))

print("\nNotes per fold:")
for n, lines in notes.items():
    print(f"  {n}")
    for l in lines:
        print(f"    {l}")
