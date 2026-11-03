"""Two questions about the policy stance in model S  (an experiment; the page model is not changed)

Model S uses tight36 = the sum of direction x intensity of the policy rounds announced in the 36 months before the
sale month (policy_events.csv: direction +1 cooling, -1 easing; intensity 1-3, a judgement of each package).

Question 1: does the 1-3 judgement matter?  Replace it with a rule nobody chooses:
  Y   S with tight36_cnt: intensity = number of instruments in the round's `instruments` column, capped at 3
      (e.g. "SSD;LTV" = 2, "ABSD;LTV;min_cash;MSR" = 3). Matches the hand score for 17 of the 30 rounds.

Question 2: like BTO supply at lag 4, is there a later, lasting policy effect?  The current stance counts a round
for the 36 months after its announcement (lags 0-3 years), when the market that triggered it is still hot
(reverse causality: cooling comes in hot markets, so the stance goes with HIGHER prices). Counting rounds
announced 48-84 months before the sale (lags 4-7 years) asks whether anything is left once that market has moved on:
  Z   S with tight36 replaced by lag4_stance (rounds announced 48-84 months before the sale month)
  S+Z S + lag4_stance: both clocks, each with its own coefficient
  (policy_events.csv starts in 1996, so for sales before about 2003 the lag-4 window is only partly covered.)

Same rule as every other candidate, set before the run: time-series cross-validation on 2015-2019 (each year
predicted from the years before it); a candidate replaces S only if it beats S by PREFER_MARGIN ($0.5k) on the
validation mean. Then the 2020-2026 out-of-time benchmark is scored once for all (the others for the record).

Runs hdb_location_model.py unchanged first (same sales, same features), about 6-8 minutes, then about 5 more.
Prints only; nothing is written.
    conda activate dsai-m3
    python hdb_policy_stance_cv.py
"""
import runpy

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

print("Running hdb_location_model.py (unchanged) to build the sales and features ...")
g = runpy.run_path("hdb_location_model.py", run_name="policy_stance")
hdb, design, ALL = g["hdb"], g["design"], g["ALL"]
VAL_YEARS, TEST_FROM, MARGIN = g["VAL_YEARS"], g["TEST_FROM"], g["PREFER_MARGIN"]
skey = next(n for n in ALL if n.startswith("S"))
scols, sflat, _ = ALL[skey]
assert "tight36" in scols, "model S has no tight36 column"

# ---------------------------------------------------------------- the rounds, two scores each
pol = pd.read_csv("policy_events.csv", parse_dates=["announce_date"])
pol["n_instr"] = pol["instruments"].astype(str).str.count(";") + 1
pol["w_hand"] = pol["direction"] * pol["intensity"]
pol["w_cnt"] = pol["direction"] * pol["n_instr"].clip(upper=3)
print("\nEach round: hand intensity vs instrument count (capped at 3):")
print(pol[["event_id", "announce_date", "direction", "intensity", "instruments", "n_instr"]]
      .assign(count_capped=pol["n_instr"].clip(upper=3)).to_string(index=False))
print(f"Same score for {(pol['intensity'] == pol['n_instr'].clip(upper=3)).sum()} of {len(pol)} rounds")


def stance(weight, m_from, m_to):
    """for every sale month: sum of `weight` over rounds announced from m_to months before up to m_from months before
    (m_from = 0 means 'before this month', exactly as tight36 in hdb_location_model.py)"""
    out = {}
    for m in pd.to_datetime(np.sort(hdb["date"].unique())):
        lo, hi = m - pd.DateOffset(months=m_to), m - pd.DateOffset(months=m_from)
        out[m] = float(pol.loc[(pol["announce_date"] < hi) & (pol["announce_date"] >= lo), weight].sum())
    return hdb["date"].map(out)


chk = stance("w_hand", 0, 36)
print(f"\nCheck: rebuilt tight36 equals the model's for {(chk.values == hdb['tight36'].values).mean():.1%} of sales")
hdb["tight36_cnt"] = stance("w_cnt", 0, 36)
hdb["lag4_stance"] = stance("w_hand", 48, 84)
print("\nAt the start of each year: tight36 (hand), tight36_cnt (count), lag4_stance (rounds 4-7 years before):")
yr0 = hdb[hdb["date"].dt.month == 1].groupby("year")[["tight36", "tight36_cnt", "lag4_stance"]].first()
print(yr0.loc[yr0.index % 2 == 1].to_string())

base = [c for c in scols if c != "tight36"]
CANDS = {"S  page model": scols,
         "Y  stance with instrument-count intensity": base + ["tight36_cnt"],
         "Z  stance from rounds 4-7 years before": base + ["lag4_stance"],
         "S+Z S + stance from rounds 4-7 years before": scols + ["lag4_stance"]}

# ---------------------------------------------------------------- time-series CV, then the 2020-2026 benchmark
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
print(f"\nBest validation mean: {best.split()[0]} ${means[best]:.1f}k; S ${means['S  page model']:.1f}k "
      f"(a new model must beat S by ${MARGIN}k). Chosen: {chosen}")

rows = []
for name in CANDS:
    p, m = pred[name][TEST_FROM]
    e = p - y[m]
    rows.append({"model": name, "chosen": name == chosen, "cv_mean_k": means[name],
                 "test_mae_k": round(float(np.abs(e).mean()), 1), "test_rmse_k": round(float(np.sqrt((e ** 2).mean())), 1),
                 "test_mape_pct": round(float((np.abs(e) / y[m]).mean() * 100), 1), "test_bias_k": round(float(e.mean()), 1)})
print(f"\nOut-of-time benchmark {TEST_FROM}-{int(yr.max())}, scored once (the choice above was made before this):")
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- what each learned (all sales)
print("\nPolicy coefficients ($k per stance point), trained on all sales:")
for name, cols in CANDS.items():
    Xa = design(cols, sflat, hdb)
    c = pd.Series(LinearRegression().fit(Xa, y).coef_, index=Xa.columns)
    pc = [k for k in ("tight36", "tight36_cnt", "lag4_stance") if k in c.index]
    print(f"  {name:<45} " + ", ".join(f"{k} {c[k]:+.2f}" for k in pc))
    del Xa
print("\nReading: a NEGATIVE lag4_stance coefficient means past rounds go with lower prices once the hot market has "
      "passed (an association, not a proven effect); positive or near zero means the later rounds add nothing.")
