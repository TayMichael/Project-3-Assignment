"""Does the 2013 CPF / HDB-loan rule for older flats improve model S?  (an experiment; the page model is not changed)

From 1 July 2013 the CPF limits for property with less than 60 years of lease were extended to HDB flats,
and HDB loans for such flats were restricted (MAS 2013, Annex IV "Use of CPF Funds and Provision of HDB Loans"):
  remaining lease 30-59 years: CPF only if youngest buyer's age + lease >= 80, pro-rated; HDB loan likewise
  remaining lease 20-29 years: no CPF;  under 20 years: no HDB loan
(From 10 May 2019 the test became "lease covers the buyer to age 95", pro-rated; still a restriction on older
flats, so the sales from 2019 on stay in the same restricted era here.)

The rule only touches OLDER flats, so it is NOT a policy round: the market-wide stance (round P15, 28 Jun 2013,
intensity 3) is unchanged. Instead two candidates add columns that switch on only for sales from 1 Jul 2013:
  W  S + lease_lt60_13: 0/1, remaining lease (99 - flat age) under 60 years (the 30-year and 20-year limits
     never bind here: the oldest flat sold, about 60 years old, still has about 39 years left)
  X  S + age13_over_35 ... age13_over_55: extra lease-decay bends that apply only from 1 Jul 2013,
     so the curve for old flats can be steeper after the rule than before it

Same test as every other candidate: time-series cross-validation on 2015-2019 (each year predicted from the years
before it), and W or X replaces S only if it beats S by PREFER_MARGIN ($0.5k) on the validation mean. Then the
2020-2026 sealed test is scored once for S, W and X (W and X for the record if not chosen).

Runs hdb_location_model.py unchanged first (same sales, same features), about 6-8 minutes, then about 5 more.
    conda activate dsai-m3
    python hdb_lease_rule_cv.py
"""
import runpy

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

print("Running hdb_location_model.py (unchanged) to build the sales and features ...")
g = runpy.run_path("hdb_location_model.py", run_name="lease_rule")
hdb, design, ALL = g["hdb"], g["design"], g["ALL"]                       # hdb = the page model's sales
VAL_YEARS, TEST_FROM, MARGIN = g["VAL_YEARS"], g["TEST_FROM"], g["PREFER_MARGIN"]
skey = next(n for n in ALL if n.startswith("S"))
scols, sflat, _ = ALL[skey]

# ---------------------------------------------------------------- the rule, known at the sale date
RULE = pd.Timestamp("2013-07-01")
post = (hdb["date"] >= RULE).astype(float)
remaining = 99 - hdb["flat_age"]
hdb["lease_lt60_13"] = post * (remaining < 60)
KNOTS13 = [35, 40, 45, 50, 55]
for k in KNOTS13:
    hdb[f"age13_over_{k}"] = post * (hdb["flat_age"] - k).clip(lower=0)
W_COLS = ["lease_lt60_13"]
X_COLS = [f"age13_over_{k}" for k in KNOTS13]

print(f"\nSales with remaining lease under 60 years, by year (the rule applies from {RULE:%d %b %Y}):")
print(hdb.assign(lt60=remaining < 60).groupby("year")["lt60"].agg(["sum", "mean"])
      .rename(columns={"sum": "sales", "mean": "share"}).round(3).to_string())

CANDS = {"S  page model": scols,
         "W  S + lease under 60 years from Jul 2013": scols + W_COLS,
         "X  S + extra lease-decay bends from Jul 2013": scols + X_COLS}

# ---------------------------------------------------------------- time-series CV, then the sealed test
y, yr = hdb["price_k"].values, hdb["year"].values
pred, cv = {}, {}
for name, cols in CANDS.items():
    X = design(cols, sflat, hdb)
    pred[name] = {}
    for Y in VAL_YEARS + [TEST_FROM]:
        tr, pr = yr < Y, (yr == Y) if Y != TEST_FROM else (yr >= TEST_FROM)
        pred[name][Y] = (LinearRegression().fit(X[tr], y[tr]).predict(X[pr]), pr)
    cv[name] = [float(np.abs(pred[name][Y][0] - y[pred[name][Y][1]]).mean()) for Y in VAL_YEARS]
    del X
tab = pd.DataFrame(cv, index=VAL_YEARS).T.round(1)
tab["mean"] = tab.mean(axis=1).round(1)
print(f"\nValidation MAE ($k), each year predicted from the years before it ({TEST_FROM} on not used):")
print(tab.to_string())

means = tab["mean"]
best = means.idxmin()
chosen = best if best != "S  page model" and means["S  page model"] - means[best] >= MARGIN else "S  page model"
print(f"\nBest validation mean: {best[:1]} ${means[best]:.1f}k; S ${means['S  page model']:.1f}k "
      f"(a new model must beat S by ${MARGIN}k). Chosen: {chosen}")

rows = []
for name in CANDS:
    p, m = pred[name][TEST_FROM]
    e = p - y[m]
    rows.append({"model": name, "chosen": name == chosen, "cv_mean_k": means[name],
                 "test_mae_k": round(float(np.abs(e).mean()), 1), "test_rmse_k": round(float(np.sqrt((e ** 2).mean())), 1),
                 "test_mape_pct": round(float((np.abs(e) / y[m]).mean() * 100), 1), "test_bias_k": round(float(e.mean()), 1)})
print(f"\nSealed test {TEST_FROM}-{int(yr.max())}, scored once (the choice above was made before this):")
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- what W and X learned (all sales)
for name, extra in (("W", W_COLS), ("X", X_COLS)):
    cols = CANDS[next(n for n in CANDS if n.startswith(name))]
    Xa = design(cols, sflat, hdb)
    c = pd.Series(LinearRegression().fit(Xa, y).coef_, index=Xa.columns)
    print(f"\n{name} coefficients for the rule columns ($k), trained on all sales:")
    print(c[extra].round(2).to_string())
    if name == "X":
        ages = np.arange(0, 71, 5)
        base = c["flat_age"] * ages + sum(c[f"age_over_{k}"] * np.clip(ages - k, 0, None) for k in range(10, 60, 5))
        after = base + sum(c[f"age13_over_{k}"] * np.clip(ages - k, 0, None) for k in KNOTS13)
        print("\nX: age effect vs a new flat ($k), before and after 1 Jul 2013:")
        print(pd.DataFrame({"age": ages, "before Jul 2013": base.round(1), "from Jul 2013": after.round(1)}).to_string(index=False))
    del Xa
