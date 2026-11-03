# Team 6: HDB Resale Forecaster

A regression model that explains and forecasts HDB resale prices, built on real resale data from data.gov.sg (**987,450 sales, Jan 1990 – Oct 2026**), and presented as a Power BI report (in the Power BI branch). It starts from what the intercept (β) and the coefficients (ω) of a linear regression mean, and builds up to a forecaster that prices a flat up to 10 years ahead from the flat, its location, its lease, outside economic drivers, government policy rounds and global events.

## 1. One plane: floor area and flat age
- **Plane for 2023 – Oct 2026** (99,729 sales): `price = $236.4k + $5.27k × floor area (m²) − $4.42k × flat age (years)`, R² = 0.533.
- **One plane for all years 1990–2026**: the age coefficient turns *positive* (+$4.40k), R² = 0.325. Older flats were mostly sold in later, pricier years, so age picks up the price rise over time — an example of omitted-variable bias.
- **Full model** (floor area, flat age, storey, town; 2023 – Oct 2026): R² = 0.823. Floor area +$6.12k per m², age −$6.03k per year, storey +$5.97k per floor. Town effects are measured against Ang Mo Kio (e.g. Bukit Timah +$221k, Choa Chu Kang −$218k). Adding town makes the age penalty larger, because older flats sit more often in pricier mature towns.

## 2. Two-input planes
Fitted with scikit-learn `LinearRegression`; prices in $k.

| Inputs | Sales used | β | ω₁ | ω₂ | R² |
|---|---|---|---|---|---|
| Floor area × flat age | 2023–2026 | 236.39 | +5.271 per m² | −4.425 per year | 0.533 |
| Floor area × storey | 2023–2026 | −7.19 | +5.451 per m² | +12.532 per floor | 0.560 |
| Flat age × storey | 2023–2026 | 637.46 | −3.761 per year | +9.510 per floor | 0.209 |
| Floor area × sale year | 1990–2026 | (shown from 1990) | +3.532 per m² | +12.907 per year | 0.732 |
| Flat age × sale year | 1990–2026 | (shown from 1990) | −4.952 per year of age | +15.849 per year | 0.574 |

With sale year in the model, flat age is negative again: the positive age slope in the all-years plane was the price rise over time, not age.

## 3. The forecaster
### The model
`price = β + area + age + storey + town + driver (previous year's value)` — fitted on sales from 2001 (income data starts in 2000; each sale uses the previous year's figure, so the model only uses information published before the sale). Policy rounds and global events are **not** regression columns: tried as 49 one-year 0/1 windows, they made the back-test worse (overfitting), so they are measured separately with an event study (below).

### Drivers and back-test
Every model is trained on 2001–2019 only (interest rate and BTO lag 5: 2006–2019; BTO lag 4: 2005–2019) and tested on the **same 2020–2026 sales it never saw**.

| Driver | Test MAE | Test RMSE | Test MAPE | Test R² | Too high / low | For comparison: MAE with 49 policy/event 0/1 columns |
|---|---|---|---|---|---|---|
| Median household income | $77.5k | $110.5k | 13.4% | 0.677 | −$23.4k | $79.7k |
| **Private property price index (URA)** | **$76.0k** | $104.9k | 13.7% | 0.709 | −$4.1k | $79.9k |
| Real GDP per person | $83.3k | $117.7k | 14.1% | 0.633 | −$32.7k | $93.7k |
| Income + interest rate (SORA) | $155.0k | $199.6k | 24.8% | −0.054 | −$147.4k | $113.3k |
| **Income + BTO supply (lag 4)** | **$74.7k** | **$102.0k** | 13.9% | **0.725** | **+$0.4k** | $91.4k |
| Income + BTO supply (lag 5) | $106.2k | $128.1k | 22.0% | 0.566 | +$73.2k | $72.1k |
| Income + private pipeline (lag 4) | $77.4k | $108.6k | 13.9% | 0.688 | −$13.3k | $82.8k |
| Income + private pipeline (lag 5) | $79.1k | $105.7k | 15.1% | 0.704 | +$7.8k | $79.3k |

- **MAE** = average miss per sale; **RMSE** weights big misses more; **MAPE** = average miss as % of price.
- Errors grow the further ahead the model predicts (Income: $62k in 2021 → $96k in 2026).
- **Income + BTO supply (lag 4) is the best driver**: lowest test MAE and RMSE, and almost no bias (+$0.4k). Each extra 1,000 BTO units launched four years earlier — roughly when they complete — goes with **−$4.19k** on resale prices. It did best in 2021–2024, the years of the post-COVID BTO delays ($61–76k a year vs $63–82k for the private PPI).
- **BTO lag 5 and the private pipeline did not help**: lag 5 was about $73k too high on average; the private pipeline (lag 4 or 5) scored about the same as income alone.
- BTO launches start in 2001, and before about 2005 most new flats came through RFS/WIS, which the series does not count — so the BTO models train from 2005/2006.
- **Tested: four ways to measure new HDB supply.** BTO launches only exist from 2001, so with a 4-year lag the BTO models start in 2005, even though plenty of new flats were supplied in 2000–2004. Three longer series were therefore tested as separate drivers, all scored on the same 2020–2026 sales:

  | Supply measure | Trains from | Driver MAE | Location model MAE |
  |---|---|---|---|
  | **BTO units launched, lag 4** (best single driver; model F) | 2005 | **$74.7k** | **$64.2k** |
  | **RFS lag 4 + BTO lag 4**: RFS flats offered a year 1995–2000 (4 × the average quarterly batch, from HDB's notices in the newspaper archive, `rfs_quarterly_1995_2001.csv`), then BTO launched 2001+ (**the base of the report's location model S**) | **2001** | $76.4k | $69.7k (G); **$66.9k with a 0/1 column for the RFS years (H)** |
  | RFS lag 5 + BTO lag 5 | 2001 | $77.3k | – |
  | HDB flats constructed (official SingStat, 1977 on), lag 1 | 2001 | $78.3k | $69.7k (I) |
  | HDB flats constructed, lag 4 | 2001 | $80.2k | $75.1k (J) |
  | HDB flats sold (official SingStat, 1977 on), lag 4 | 2001 | $83.8k | $79.5k (O) |
  | HDB flats constructed, lag 5 | 2001 | $92.2k | $88.0k (K) |
  | HDB flats sold, lag 5 | 2001 | $97.4k | – |

  As a single driver, BTO launches at lag 4 are the best predictor. A launch four years before the sale is roughly when those flats **complete and owners collect keys**: real extra supply, decided in an earlier market, so not a reaction to today's prices (coefficient −$3.5k per extra 1,000 units). Completions at lag 1 come closest, showing 2001–2004 do carry information; completions and flats sold lagged 4–5 years point at the wrong period and are badly biased (about −$50k to −$80k). Joining the RFS years to the BTO launches costs about $5.5k on the 2020–2026 test (G $69.7k against F $64.2k), and $2.7k once the RFS years get their own 0/1 column (H $66.9k); the location model nevertheless uses RFS lag 4 + BTO lag 4, so that it can learn from 2001–2004 (see **"Bringing back 2001–2004"** below). A lagged figure also misses the 2001–03 glut of ready unsold flats (about 17,500 in early 2002). All seven alternatives appear in the report's driver dropdown.

- **Tested and rejected: the announcement effect (lag 0 and lag 1).** Buyers who book a BTO leave the resale market at launch, not at key collection, so launches in the sale year or the year before were tested too:

  | Supply measure | Trains from | Driver MAE | Location model MAE | Effect per extra 1,000 units |
  |---|---|---|---|---|
  | BTO launched, same year (lag 0) | 2001 | $85.8k* | $82.3k* | **+$7.1k** |
  | RFS lag 1 + BTO lag 1 (RFS 2000, then BTO) | 2001 | $78.0k | $71.1k | **+$1.7k** |
  | BTO launched, year before (lag 1) | 2002 | $98.1k | $95.6k | **+$7.0k** |
  | *BTO launched, lag 4 (used), for comparison* | 2005 | $74.7k | $64.2k | −$3.5k |

  \*tested on 2020–2025 only (2026 launches not yet published).

  The sign is the wrong way round: more launches go with **higher** prices. This is **reverse causality**: HDB launches more *because* prices are hot (launches rose from 1,243 in 2005 to over 25,000 a year in 2011–13 after the 2007–10 surge, and to over 23,000 in 2022–23 after the COVID surge). In the same year the two effects cannot be separated, and the government's response dominates. The announcement effect is probably real, but yearly launch totals cannot measure it. Four years later the launch decision is old news and only the supply effect is left, which is why lag 4 works. These three run in `hdb_drivers_model.py` and `hdb_location_model.py` (models L, M, N) and are printed in the back-test, but are kept off the report, where a slider showing "more BTO, higher price" would mislead.
- **Income + interest rate failed**: trained when rates were near zero, it learned "higher rate → lower price"; when SORA rose to 3.4–3.7% in 2023–24, prices kept rising, and it predicted about $147k too low.

### Government policy and global events
**Sources:** `policy_events.csv` (30 policy rounds, 1996–2026, mostly MAS-cited) and `global_events.csv` (19 global and local shocks, 1997–2025) — the dbt seeds behind the `int_policy_month` model.

Each round or event is measured on **actual resale prices** (flat mix removed with a flat-only model on all sales from 1990 — no driver), 1, 2 and 3 years later, against a baseline:
- **Policy rounds** — the trend in the 12 months before for year 1 (a round is aimed at that trend), then normal growth for years 2 and 3 (no boom or slump runs at full speed for three years).
- **Global events** — **normal growth**, the long-run average of **4.9% a year** (1990–2026).

| Group | Rounds | vs baseline after 1 yr | 2 yrs | 3 yrs |
|---|---|---|---|---|
| Loan limits & HDB rules (LTV, MSR, TDSR, tenure, stress rate, wait-out, MOP) | 9 | −4.3% | −10.0% | −11.9% |
| SSD | 5 | −3.4% | −5.8% | −10.8% |
| ABSD | 6 | −2.2% | −5.1% | −7.2% |
| BSD / tax | 2 | −1.4% | −1.0% | −3.4% |
| Easing (incl. grants) | 8 | −0.7% | −1.0% | −1.7% |
| **Crisis followed by QE** (2008 GFC, 2020 COVID) | 2 | −0.8% | +6.8% | **+12.0%** |
| **Crisis without QE** (1997 Asian crisis, 2000 dot-com, 2001 9/11, 2003 SARS) | 4 | −10.6% | −18.0% | **−21.8%** |
| Fed tightening / taper (2013, 2022) | 2 | −4.0% | −9.8% | −11.2% |
| Trade / geopolitical shock (2018, 2022, 2025) | 3 | −2.2% | −1.6% | +2.5% |
| Other financial shock (2007 subprime onset, 2015 China) | 2 | +5.6% | +2.2% | +2.7% |
| Local lockdown (2020 Circuit Breaker) | 1 | +1.8% | +6.2% | +10.6% |
| Fed easing (Sep 2024) | 1 | −0.8% | −6.8% | n/a |

Every tightening group comes out negative. Easing rounds come out close to zero: they were usually made in weak markets, and did not lift HDB prices above the baseline. Four low-relevance events (2010 euro crisis, 2011 US downgrade, 2016 Brexit, 2023 US bank stress) are shown in the table but not averaged.

In the report, a **global event** placed "in year T" multiplies the model's price in T+1, T+2 and T+3 onwards by its measured 1/2/3-year figure. A **policy round** adds its weight to the policy stance (the report model uses the stance as a column; see "Using the report" below) for sales from the month after it was announced, for 36 months, exactly as in training. A real round keeps its own month (moved to the year you pick); a "like <group>" choice counts as announced on 1 July of its year. Any number of rounds and events can be added, each in its own year; "+ the rounds that really happened" fills in the real ones up to the sale year.

### Why QE matters: crises with and without quantitative easing
**What QE is.** Quantitative easing is when a central bank creates new money and uses it to buy government bonds and other assets in large amounts — used when interest rates are already near zero and cannot be cut further. It works in three steps:
1. **Long-term interest rates fall**, including mortgage rates.
2. **More money looks for returns**: investors pushed out of low-yielding bonds move into stocks, property and other markets, including Singapore.
3. **Asset prices rise first**, before wages or consumer prices.

**How US QE reaches HDB prices.** Singapore's central bank (MAS) manages the exchange rate, not interest rates, so Singapore interest rates follow US rates. When the US Federal Reserve holds rates at zero and runs QE, Singapore mortgage rates fall too. In this data, SORA averaged **2.3–3.3%** in 2005–2007, then **0.05–0.27%** in 2009–2013 and **0.20–0.45%** in 2020–2021.

| | Sep 2008 — Global Financial Crisis | Jan 2020 — COVID-19 | Jul 1997 — Asian Financial Crisis | Mar 2000 — Dot-com bust | Sep 2001 — 9/11 | Mar 2003 — SARS |
|---|---|---|---|---|---|---|
| Central bank response | Fed cut to near zero and started QE (from Nov 2008), more rounds in 2010 and 2012 | Fed cut to zero and started bond buying (Mar 2020), running until Mar 2022 | No QE | Rate cuts only, no QE | Rate cuts only, no QE | No QE (local health shock) |
| Singapore rates afterwards | SORA about 0.1% in 2010–2013 | SORA about 0.2% in 2021 | — | — | — | — |
| HDB prices 3 years later | **+31.4%** | **+26.8%** | **−20.9%** | **−15.3%** | **−0.7%** | **−2.3%** |
| vs normal growth (+15.3% over 3 years) | **+14.0%** | **+10.0%** | **−31.4%** | **−26.5%** | **−13.9%** | **−15.3%** |
| Pattern | Small dip, then a steep climb 2009–2012 | Flat during lockdown, then a steep climb 2021–2023 | Steep fall | Falling | Flat to falling | Flat to falling |
| Government response | Cooling rounds Sep 2009 – Dec 2013 (loan limits, SSD, ABSD, TDSR) | Cooling rounds Dec 2021, May 2022, Sep 2022, Apr 2023 | SSD suspended (Nov 1997) | — | Easing (Oct 2001) | — |

**With QE**, prices dipped or stalled at first, then rose well above normal once cheap money arrived — and both times the government had to cool the market repeatedly, which fits QE-driven demand. **Without QE**, a recession or health scare simply reduced demand (job worries, fewer buyers), and nothing pushed prices back up within three years.

**What this does not prove.** Two other things differed between the pairs:
- **HDB supply** — in the early 2000s HDB had surplus flats (it moved to Build-To-Order in 2001), which held resale prices down in 2001–2003; in 2020–2022, COVID halted BTO construction and delayed keys, pushing buyers to resale.
- **Population** — 2008–2012 also had fast population growth, adding demand.

With two QE examples and four without, the fair summary is: *after crises followed by QE, HDB prices rose about 12% above normal within 3 years; after crises without QE, they fell about 22% below normal; housing supply moved the same way each time, so both likely contributed.*

### Location model: your block, the MRT in the sale year, and lease decay
The report opens on driver **"Income + RFS lag 4 + BTO lag 4 + flats sold lag 4 + policy stance + location (your block)"**, model S (H plus flats sold and the policy stance, chosen without the test years; see "Choosing without the test years" below). Type a block and street (e.g. `82B CIRCUIT RD`) and pick the flat type: the report fills in that block's own location, sets the town, recomputes the MRT network for the sale year you choose, and **sets the floor area, storey, flat age and flat model to the block's typical flat of that type** (medians of its resales since 2010, age as of the latest month, from `hdb_trees.js`). For Blk 267 Tampines St 21, 4 room, that is 114 m², floor 5, 43 years old in 2026, Model A. Before this, the sliders stayed at 93 m², floor 8 and 20 years for every block, so a block built in 1983 was priced as a 20-year-old flat. The sliders can still be moved afterwards; if the block has no resales of that flat type since 2010, they keep their values. The other 15 drivers are still in the dropdown.

#### Back-test: how much each step added
Trained on 2005–2019 (G and H: 2001–2019), tested on 2020–2026 sales it never saw (178,801 test sales; same sales for every row):

| Model | Test MAE | Test RMSE | MAPE | Bias | What the step fixed |
|---|---|---|---|---|---|
| Income alone | $77.5k | $110.5k | 13.4% | −$23.4k | (starting point) |
| A. Income + BTO lag 4 | $74.7k | $102.0k | 13.9% | +$0.4k | price **level** over time |
| B. + flat type & model | $72.4k | $98.6k | 13.5% | +$0.3k | 3-room vs Executive, Model A vs DBSS etc. at the same size |
| C. + CBD, mall, school | $71.1k | $97.1k | 13.3% | +$0.8k | where the block is |
| D. + MRT at the sale date | $70.5k | $96.2k | 13.1% | −$2.2k | MRT open / announced at the time of sale |
| E. D on log price | $94.1k | $117.7k | 17.7% | +$84.8k | worse: a log model compounds income growth into runaway prices |
| **F. D + lease-decay curve** | **$64.2k** | **$90.8k** | **11.3%** | −$12.3k | **old flats** (see below); lowest error |
| G. F, supply = RFS lag 4 (1997–2000) + BTO lag 4 (2001 on), trained from 2001 | $69.7k | $101.2k | 11.6% | −$32.7k | adds the 2001–2004 sales |
| **H. G + RFS-years column** | **$66.9k** | **$95.1k** | **11.7%** | −$14.3k | the base of page model **S** (see "Bringing back 2001–2004" and "Choosing without the test years") |

From income alone to F, the typical miss fell by **$13.3k (17%)** and the large misses (RMSE) by $19.7k (18%). Every test year improved:

| Test year | A | D | **F** |
|---|---|---|---|
| 2020 | $90.4k | $81.6k | **$76.2k** |
| 2021 | $60.6k | $54.1k | **$50.1k** |
| 2022 | $61.4k | $59.0k | **$55.4k** |
| 2023 | $64.2k | $61.9k | **$56.4k** |
| 2024 | $76.1k | $73.6k | **$66.1k** |
| 2025 | $82.8k | $80.3k | **$72.2k** |
| 2026 | $94.5k | $89.9k | **$79.1k** |

#### Bringing back 2001–2004: why H (the base of report model S) uses all years, not the lowest-error F
This is the most important modelling decision in the project, so it is set out in full.

**1. How the period was first dropped.** The first location models (A–F) measured new HDB supply by **BTO units launched**, taken 4 years before the sale (about when the flats complete). BTO launches only begin in **2001**, so a 2001–2004 sale had no lag-4 figure, and those sales were simply left out: models A–F learn from **2005–2019** only. Nothing was set to zero; four years of the market were just not seen. That left out **132,016 sales**, the market after the dotcom bust, 9/11, SARS and the Iraq war.

**2. Rebuilding the missing supply.** Before BTO, HDB sold new flats through **Registration for Flats (RFS)** and walk-in sales, and published the flats **available for selection** in each quarter's batch in notices that appeared in the press ("Flats available in the 3rd quarter"). Each batch is a **flow**: the flats offered that quarter, new flats plus flats left over from earlier batches; the press described the first batch of a year as part of "the 30,000 or so new flats that the HDB releases to applicants each year". The batches were recovered from the newspaper archive (a RAG search over scanned pages, then a search of every notice in the archive's SQLite copy) and saved quarter by quarter in `rfs_quarterly_1995_2001.csv` (year, quarter, flats, 4-room / 5-room / executive split, source page). A year's figure is **4 × the average of the quarters found**, then joined to the BTO launches into one series, **RFS lag 4 + BTO lag 4**:

| Supply year | Quarters found | Flats offered in the year | What the figure is | Used for sales in |
|---|---|---|---|---|
| 1997 | 4 (Q1–Q4) | 29,555 | RFS batches, 4 × quarterly average | 2001 |
| 1998 | 3 (Q1–Q3) | 35,683 | RFS batches, 4 × quarterly average | 2002 |
| 1999 | 3 (Q1, Q3, Q4) | 39,649 | RFS batches, 4 × quarterly average | 2003 |
| 2000 | 3 (Q1, Q2, Q4) | 32,137 | RFS batches, 4 × quarterly average | 2004 |
| 2001 | – | 2,594 | BTO units launched | 2005 |
| 2002 | – | 1,466 | BTO units launched | 2006 |
| 2003 | – | 3,092 | BTO units launched | 2007 |
| 2004 | – | 4,535 | BTO units launched | 2008 |
| … | – | BTO launched each year | BTO units launched | … to 2026 and every future year |

1995 (33,156, one quarter found) and 1996 (27,021, three quarters) are used by the lag-5 series. The yearly totals sit close to HDB's official **flats sold** (1997 28,220; 1998 36,966; 2000 26,329) and **flats constructed** (1997 31,312; 1998 36,609), which supports the 4 × quarter reading. 1999 is the highest because Q1 1999 alone offered 11,585 flats; Q2 1999 (at least 9,590) was only partly legible and is left out.

**Single quarter vs whole year.** The first version of this series used **one quarter per year** (1997 8,160; 1998 9,057; 1999 8,037; 2000 8,092), as if it were a stock of flats on hand. That understated each year about four times. `RFS_MODE=year` (the default) uses the yearly totals; `RFS_MODE=quarter` brings back the single-quarter values for comparison. Every script was rerun with the yearly totals:

| | Single quarter | **Yearly total (now)** |
|---|---|---|
| Plane S, validation 2015–2019 (mean MAE) | $53.5k | **$52.4k** |
| Plane S, 2020–2026 benchmark | $73.9k | $75.2k |
| Hybrid (127 leaves), validation 2015–2019 | $41.2k | **$38.6k** |
| Hybrid (127 leaves), 2020–2026 benchmark | $64.2k | $65.0k |
| H, benchmark | $66.9k | $66.9k |
| G (no RFS column), benchmark | $67.3k | $69.7k |
| `rfs4` column in H | −$8.0k | **+$63.2k** |

The yearly totals win on validation, the years used to choose, and lose about $1k on the 2020–2026 benchmark, which is within noise (see the 127-leaf tuning below). They are kept because they are the right unit, not because of the scores: a year of RFS batches and a year of BTO launches are both **flats offered in a year**. The same choices still come out of cross-validation: model S, and the dollar hybrid with 127 leaves.

**3. Why it is named "RFS lag 4 + BTO lag 4", not just "BTO lag 4".** The series is BTO **only from 2001**; its first four values are RFS. Calling the whole series "BTO lag 4" would suggest that 2001–2004 sales were matched to BTO launches, which did not exist then. So both parts are named, as in the original reconciliation workbook (`HDB_RAG_Reconciled_Lag4_Lag5_2000_2024.xlsx`): **RFS lag 4** for the 2001–2004 sales, **BTO lag 4** from 2005 and in every future year. On its own, "BTO lag 4" means the BTO-launches series of models A–F.

**4. Two kinds of flow, and the RFS-years column.** Both parts now count flats offered in a year, but not the same flats: an RFS year counts **every flat in the year's batches**, including unsold flats offered again; a BTO year counts **new flats launched to order**, while the unsold flats of 2001–2004 were sold separately by walk-in selection and are not counted. So the fall from 32,137 in 2000 to 2,594 in 2001 is partly real (HDB stopped building ahead of demand, with about 17,500 new flats unsold in early 2002) and partly a change of measure. One coefficient for both would let the RFS years distort the BTO effect. Model H therefore adds one 0/1 column, **`rfs4`** (the RFS lag 4 column):
- **1** for 2001–2004 sales (their figure 4 years earlier is RFS)
- **0** for sales from 2005 onwards, **and in every future year**

Its coefficient, **+$63.2k**, shifts the 2001–2004 price level so that the change of measure does not bend the supply slope: with 30,000–40,000 RFS flats a year and a supply slope of about −$1.3k per 1,000 flats (model G), the RFS years would pull 2001–2004 prices down by $40–50k, and the column puts most of that back. (With the single-quarter figures it was −$8.0k.) The supply slope itself is still one coefficient learned from both periods; a separate RFS slope would need an interaction column (`avl4_k × rfs4`), which has not been tested. Because `rfs4` is 0 in every future year, a forecast never uses the RFS adjustment: future supply is BTO, exactly as in model F.

**5. What it costs on the back-test.** Same 2020–2026 test sales for all three:

| | F (BTO lag 4 only) | G (RFS lag 4 + BTO lag 4) | **H (RFS lag 4 + BTO lag 4 + `rfs4` column)** |
|---|---|---|---|
| Learns from | 2005–2019 | 2001–2019 | **2001–2019** |
| Training sales | 355,435 | 487,451 | **487,451 (+37%)** |
| Test MAE | **$64.2k** | $69.7k | $66.9k |
| Test RMSE | **$90.8k** | $101.2k | $95.1k |
| MAPE | **11.3%** | 11.6% | 11.7% |
| Bias | **−$12.3k** | −$32.7k | −$14.3k |

| Test year | F | H | Difference |
|---|---|---|---|
| 2020 | $76.2k | $79.3k | +$3.1k |
| 2021 | $50.1k | $52.4k | +$2.3k |
| 2022 | $55.4k | $57.3k | +$1.9k |
| 2023 | $56.4k | $58.9k | +$2.5k |
| 2024 | $66.1k | $68.7k | +$2.6k |
| 2025 | $72.2k | $75.6k | +$3.4k |
| 2026 | $79.1k | $82.1k | +$3.0k |

H misses by about **$2.7k (4%) more** per sale, evenly across the test years.

**6. Why H is used anyway.**
- **The test period is a boom.** 2020–2026 prices rose strongly, so the back-test rewards a model that learned only from 2005–2019, a mostly rising market. It cannot show which model would do better in a downturn.
- **2001–2004 was a different market, and a real one.** HDB prices fell through it with no QE to support them: the policy and event table above measures prices about **14% below normal three years after 9/11 (Sept 2001)** and about **15% below normal three years after the SARS / Iraq-war shock (March 2003)**, with about 17,500 new flats unsold in early 2002. Learning from those sales grounds the income, age, location and supply coefficients in a buyers' market as well as booms.
- **It uses every sale the data allows** (2001 is the earliest, as each sale uses the previous year's income, which starts in 2000) and the full RFS lag 4 + BTO lag 4 series that was rebuilt for this purpose.

**7. Was the newspaper archive necessary? Tested against the official series.** Annual HDB **flats constructed** and **flats sold** (Home Ownership Scheme) are published by SingStat back to 1977 (table TS/M400151, *Residential Units Constructed And Sold By Housing And Development Board, Annual*, on data.gov.sg), so they are not dark data, and they are one consistent measure before and after BTO, with no change of measure in 2001 and no `rfs4` column needed. What is **not** published as open data is the quarterly **RFS flats available for selection**: it survives in HDB's notices printed in the newspapers (for example the 1997 Q3 table, 4-room 3,329 + 5-room 3,827 + executive 1,004 = **8,160**, and Lianhe Zaobao of 14 Oct 1999, **8,037** flats for Q4 1999, down from **10,115** in Q3). Added up over the batches, these give the yearly totals in item 2.

Each series must be compared at the lag that matches **when its flats reach owners**:
- **Launches** (RFS availability, then BTO): flats offered in year Y are handed over about 4 years later, so **lag 4** lines up with key collection.
- **Completions**: flats completed in year Y are handed over in year Y, so the matching lag is **0**. But a sale in, say, March cannot know that year's total (published after the year ends): using it would be **data leakage**. **Lag 1** (last year's completions) is the latest figure a buyer really had, so it is the fair official benchmark.
- **Flats sold**: counted when bought, partly reflecting demand (the reverse-causality problem found with BTO lag 0).

| Supply series | Source | Lag | Learns from | Test MAE | Bias |
|---|---|---|---|---|---|
| **RFS lag 4 + BTO lag 4, with `rfs4` (H, base of report model S)** | newspaper archive + BTO project list | 4 | 2001 | **$66.9k** | −$14.3k |
| **HDB flats constructed (I)** | SingStat, official | **1** | 2001 | **$69.7k** | −$16.1k |
| HDB flats constructed (J) | SingStat, official | 4 (wrong timing: points ~8 years after launch) | 2001 | $75.1k | −$53.6k |
| HDB flats sold (O) | SingStat, official | 4 | 2001 | $79.5k | −$59.8k |

At the fair timing, the official completions (I) come within **$2.8k** of H. Completions lagged 4 years, and flats sold, are badly biased: both peaked in 2014–2017 (e.g. 35,210 built and 27,708 sold in 2017), so 4 years later they predict a glut just as prices rose.

**Why H (launches) is kept over I (completions + 1):**
- **Accuracy:** $2.8k better on the same test, and better in all 7 test years (2022 and 2026 by only $0.2k). Small, and on one test period, so this alone is not decisive.
- **Forecast horizon, the stronger reason.** The report forecasts up to 10 years ahead. With launches at lag 4, the supply for 2027–2030 sales is **already known** (BTO launched 2023–2026); with completions at lag 1, only 2027 is known and every later year must be guessed on the slider. Launches lagged 4 years are, in effect, a published forecast of completions.
- **Where the newspaper data matters:** only for the 2001–2004 sales (RFS lag 4). From 2005 on, H uses annual BTO launches. So the archive's value is not that a quarterly figure outweighs an annual one; it is that, before 2001, the **flats-offered** measure exists only in those notices, batch by batch.

**Trade-off stated plainly:** I is simpler and entirely official; H is slightly more accurate and knows future supply further ahead, at the cost of a reconstructed 2001–2004 segment and the `rfs4` column. Both are in the back-test table so the choice can be checked.

**8. What H does *not* do.** It does not predict a crash. A linear plane has no "slump" feature: the 2001–2004 price level is absorbed partly by the `rfs4` term (0 in the future) and partly by the other coefficients. Crash scenarios are in the report as a separate choice, the **global event** selector (e.g. *crisis without QE*), measured from actual prices. And the gain from 2001–2004 is a judgement, not a measured improvement: on the only test available, F is slightly more accurate. Both are shown so the choice can be checked.

#### Choosing without the test years: time-series cross-validation and a 2020–2026 benchmark
**Procedure** (`hdb_location_model.py`, saved to `hdb_data/location_cv.csv` and `location_sealed_test.csv`):
1. **Time-series cross-validation inside the training years.** For each validation year 2015, 2016, 2017, 2018 and 2019, each candidate is trained on all earlier years and predicts that year (an **expanding window**; later years are never used, and 2020–2026 is not even loaded). Ordinary k-fold would let 2018 sales help predict 2016, which is leakage for a forecast.
2. **Choose** the candidate with the lowest average validation MAE.
3. **Score it once** on 2020–2026, trained on everything up to 2019. The report model H is scored alongside, since it is the model actually deployed.

**Validation MAE by year ($k; 2020–2026 not used):**

| Model | 2015 | 2016 | 2017 | 2018 | 2019 | Mean |
|---|---|---|---|---|---|---|
| **S. H + flats sold lag 4 + policy stance** | **42.9** | **55.2** | **51.4** | 49.9 | 62.8 | **52.4** |
| T. O + policy stance (official data only) | 45.8 | 59.5 | 57.0 | **49.7** | **55.1** | 53.4 |
| R. H + policy stance | 47.0 | 60.3 | 54.1 | 59.7 | 78.2 | 59.9 |
| P. H + flats sold lag 4 | 70.7 | 71.4 | 71.0 | 51.2 | 72.1 | 67.3 |
| O. HDB flats sold, lag 4 (official) | 89.5 | 84.1 | 76.5 | 54.4 | 60.2 | 72.9 |
| J. Flats completed, lag 4 | 95.1 | 93.9 | 79.6 | 62.9 | 69.4 | 80.2 |
| Q. Average of H and O | 87.9 | 79.9 | 76.8 | 73.2 | 92.7 | 82.1 |
| F. BTO lag 4 | 81.8 | 66.5 | 70.9 | 98.2 | 141.7 | 91.8 |
| D. Location + MRT, straight-line age | 82.2 | 67.1 | 71.9 | 99.1 | 145.7 | 93.2 |
| H. G + RFS lag 4 column | 86.4 | 75.8 | 77.1 | 98.9 | 133.3 | 94.3 |
| G. RFS lag 4 + BTO lag 4 | 105.3 | 96.7 | 86.3 | 89.6 | 94.8 | 94.5 |
| I. Flats completed, lag 1 | 113.2 | 103.2 | 89.5 | 79.4 | 102.9 | 97.6 |

**2020–2026 benchmark, scored once:**

| Model | Mean validation MAE | Test MAE | Test RMSE | MAPE | Bias |
|---|---|---|---|---|---|
| **S (chosen on validation; page model)** | **$52.4k** | **$75.2k** | $107.4k | 12.1% | −$50.1k |
| T (official data only; for the record) | $53.4k | $78.0k | $111.3k | 12.3% | −$57.7k |
| O (chosen in the first round, before P–S were added) | $72.9k | $79.5k | $112.3k | 12.6% | −$59.8k |
| H (previous page model) | $94.3k | $66.9k | $95.1k | 11.7% | −$14.3k |

**What it shows: the best model depends on the market regime.**
- **2015–2019 was a falling market** (after the 2013 loan-limit rounds, resale prices drifted down while income kept rising). Models led by income and BTO launches kept predicting rises and missed by up to $146k in 2019. **Flats sold** had peaked in 2014–2017, so lagged 4 years it predicted the glut and the soft prices of 2018–2019: O wins validation.
- **2020–2026 was a boom.** The same peak in flats sold now predicts low prices just as they rose, and O is $60k too low on average; the launch-based models (F, G, H) do best.
- So a properly sealed procedure would have deployed **O**, scoring **$79.5k**, not the $64–67k of the models picked by looking at the test. **H's $66.9k is a real test score, but not an untouched one**: it is the best of many models compared on the same years.
- No single supply measure wins in both regimes. The largest remaining error is not which supply series is used but **market regime and policy**: the cooling rounds of 2013 and the post-COVID run-up, which one income figure and one supply figure cannot follow.

**Combining the regimes.** Four candidates were added to the same procedure (and kept off the test table above, so their test scores were not seen before choosing):
- **P**: H + flats sold lag 4 (both supply measures in one plane, each with its own coefficient)
- **Q**: the average of H's and O's predictions (a two-model ensemble)
- **R**: H + **policy stance** `tight36`: the sum of direction × intensity (cooling +, easing −; intensity 1–3, from `policy_events.csv`) of the policy rounds **announced in the 36 months before the sale month**. Only rounds announced in earlier months count, so a sale never knows a later round. It is about +11 at the start of 2013 (the 2010–2012 SSD, ABSD and loan rounds) and −1 to −2 in the easing years of 2001–2007. 36 months matches the event study above, where cooling rounds kept working for 2–3 years.
- **S**: H + flats sold lag 4 + policy stance
- **T**: O + policy stance, official data only (added later to test whether the newspaper series is needed; see "Is the newspaper series needed?" below)

**S won validation by a wide margin ($52.4k, against $72.9k for O and $94.3k for H)** and scored **$75.2k** on the 2020–2026 benchmark, better than the $79.5k of O, the earlier honest pick. Over all 12 years (5 validation, 7 test) S averages about **$66k per sale**, against about **$78k for H**. The policy stance alone cut H's validation error from $94k to $60k (R). **But its coefficient is positive** (+$5.5k per point, trained on 2001–2019): more cooling rounds in the last 3 years go with *higher* prices. It helps as a **sign of a hot market**, not as a brake: rounds are announced when prices run hot, the stance peaked at about +11 in 2013 with prices, then faded towards 0 in 2014–2017 as no new rounds came while prices drifted down. See "Reading the policy stance" below.

**What S still misses:** it is $50k too low on average in 2020–2026. The cooling rounds of 2021–2023 raised the stance to about +9 by 2025, which pushed S's prices up, but not by enough: the post-COVID boom, driven by pent-up demand and construction delays, ran hotter than any market in the training years. The intensity scores (1–3) are judgements recorded in the policy file, not official measures.

**Decision: the report uses S**, the model a disciplined process selects, the best over a full cycle, with an honest benchmark score. H's lower 2020–2026 error came from being chosen on those years. In the report, S's formula has two more terms than H: flats sold 4 years before (2027–2029 already published; later years at the 10-year average) and the policy stance. For future years the stance counts only rounds already announced, which fade out 3 years after each; the **policy selector now feeds the stance** (its direction × intensity is added for the 3 sale years after the chosen year) instead of applying a separate percentage. **Expected error**: about $75k a sale on the 2020–2026 boom years, about $52k on the falling years of validation.

**Is the newspaper series needed? Model T, official data only.** S mixes two supply series: RFS + BTO (with the `rfs4` column for the change of measure in 2001) and HDB flats sold. Flats sold is one consistent official measure in every year, so the simpler question is whether it is enough on its own. **T = O + policy stance**: location, lease decay, income, flats sold lag 4 and the policy stance, from 2001, with no newspaper data and no `rfs4` column. The rule was written into the script before the run: S must beat T by at least **$0.5k** on validation, otherwise the simpler, consistent T goes in the report.

| | Validation 2015–2019 | Benchmark 2020–2026 | Bias |
|---|---|---|---|
| **S** (both supply series + stance) | **$52.4k** | **$75.2k** | −$50.1k |
| T (flats sold + stance, official only) | $53.4k | $78.0k | −$57.7k |

- **S passes the rule by $1.0k** and stays in the report; it is also $2.8k better on the 2020–2026 benchmark (scored for the record, not used to choose).
- **The gain is small, and T wins the two latest validation years** (2018 and 2019, the soft market that flats sold predicts well); S wins 2015–2017. So the extra supply series earns its place, but only just.
- **Where S's extra information comes from.** S's RFS + BTO column is BTO launches for every sale from 2005, and RFS only for the 2001–2004 sales. The newspaper RFS years let S learn from 2001–2004 with the same column; their level is absorbed by `rfs4`, which is 0 in every future year, so a forecast never uses an RFS figure. The inconsistency between RFS and BTO is therefore confined to four training years, and the column that handles it is reported, not hidden.
- **T is the fallback** if the newspaper series is ever questioned: about $1k worse on validation and $3k on the test, entirely reproducible from data.gov.sg.

**Letting the regression weigh the policy rounds itself: models U and V.** The stance adds up direction × an **intensity of 1–3** written by hand in `policy_events.csv`, so the regression only learns one slope for a score a person weighted. Two candidates count the same rounds with **no intensity**, so the regression learns each weight itself:
- **U**: S with the stance replaced by two counts, cooling rounds (`cool36`) and easing rounds (`ease36`) announced in the 36 months before the sale.
- **V**: S with the stance replaced by one count per policy type (`pol_ssd`, `pol_absd`, `pol_loan`, `pol_bsd`, `pol_ease`), the types of the event study (`policy_group`, by instrument).

Rule set before the run: U or V replaces S only if it beats S by $0.5k on validation.

| | Policy columns | Validation 2015–2019 | Benchmark 2020–2026 | Bias |
|---|---|---|---|---|
| **V** (counts by type) | 5 counts | **$48.5k** | $86.1k | −$70.4k |
| **S** (stance, 1–3 weights) | 1 score | $52.5k | **$75.2k** | −$50.1k |
| U (cooling / easing counts) | 2 counts | $55.4k | $77.5k | −$53.7k |
| T (official data only) | 1 score | $53.4k | $78.0k | −$57.7k |

What each learned, trained 2001–2019 ($k per round counted in the 36 months before the sale): **S** +5.50 per stance point; **U** cooling +12.90, easing −6.35; **V** SSD +28.00, ABSD +4.68, loan limits & HDB rules +6.76, BSD / tax +20.19, easing −7.30.

- **V won validation by $4.0k and failed the 2020–2026 benchmark by $10.9k.** Before 2020 there were only 1 BSD / tax round, 3 ABSD rounds and 4 SSD rounds, so its per-type coefficients came from very few events: they fitted 2015–2019 (SSD rounds fading) and did not carry into 2020–2026 (many ABSD and loan rounds, priced at only +$4.7k each). Its 20 replays agree: median miss about $70k against S's $68k, better on training cuts 2010–2015, worse on every cut from 2016. This is **overfitting**: more freedom than the data can support.
- **S stays in the report, and this choice used the 2020–2026 benchmark.** The protocol says the test is not used to choose; here it exposed a model the validation could not. So a rule was added **after** that run and is stated as such: a count column can only be chosen if at least **3 rounds** of its kind were announced before 2020 (`MIN_ROUNDS` in `hdb_location_model.py`). V fails it (BSD / tax: 1 round); U passes but loses to S on validation. V and U stay in every table for the record.
- **None of these is a causal effect of a policy.** Every cooling coefficient is positive because the government cools when prices are already rising (reverse causality, as with BTO lag 0). They measure what kind of market comes with each kind of round, not what the round does to prices. Removing the hand-written weights removed a judgment, but cannot turn this data into causal effects.

**Reading the policy stance: a sign of a hot market, not a brake.** The stance answers a **forecasting** question: "if a cooling round is announced, what will prices be in the next 3 years?" History says still high, because rounds come when the market is hot and a hot market stays hot for a while: prices rose 14.4% in the year after the February 2010 SSD round, 11.7% after August 2010 and 8.1% after January 2011. A one-year lag does not change this (the stance already uses only rounds announced before the sale): the same hot market causes both the round and the prices after it, which is **reverse causality**. So in the report, choosing a cooling round **raises** the forecast for the 3 years after it, as the stance coefficient says. The stance cannot answer the **effect** question, "how much does the round itself hold prices back, compared with the same market without it?", because the training years have no hot market without a round. Only the event study above answers that, by comparing growth after each round with the trend before it: for SSD rounds about −3.4% after 1 year and −10.8% after 3 years. The policy note in the report shows that event-study figure next to the stance's effect, for reference.

The largest remaining error is **market timing**: every model misses most in 2020 (the COVID dip broke the trend) and 2025–26 (the run-up), because one income figure cannot follow a boom or a dip.

#### Why R² here is lower than a "valuation" R²
R² depends on how the test is set up, not only on the model:

| Test | R² | MAE | Question it answers |
|---|---|---|---|
| Fit on all years, scored on the same years | 0.853 (model D) | – | How well does the line fit the data it learned from? |
| Random 80/20 split (test sales from the same years as training) | **0.854** (model D) | **$54.9k** | **Valuation**: what is this flat worth in a year the model already knows? |
| Train to 2019, test 2020–26 (the report) | **0.755** (model D) | **$70.5k** | **Forecasting**: what will it be worth in years the model has never seen? |

A random split lets the model see other sales from the same year, so it knows that year's price level and only has to rank flats within it. The time-based back-test must also guess the price level of years it never saw, which is the harder and more honest test for a future-price page. Models with year or month dummies reach R² of about 0.90–0.95 on a random split, but cannot forecast at all: there is no dummy for 2031.

#### Location features
| Feature | From | Effect in model D (same town, same flat) |
|---|---|---|
| Km to the nearest MRT **already open** in the sale month | block coordinates + station opening dates | **−$38k per km** |
| Km **closer** an announced-but-not-open station is | announcement and opening dates | **+$34k per km**: buyers price a new station in before it opens |
| Announced station in the first / second half of its wait | midpoint between announcement and opening | first half −$46k (LTA announces stations for under-served, cheaper estates, so the flag mostly marks those areas); second half near 0 (the gain is already in the price) |
| Km to the CBD | block lookup | −$8.4k per km |
| Km to the nearest mall | block lookup | −$7.5k per km |
| Within 1 km of a high-demand primary school | block lookup | +$5.8k (the distance itself comes out positive because it overlaps with the 1 km flag and the town dummies, so read the two together) |
| Flat type and flat model | resale data | dummies (base: 4 Room, Model A); models with fewer than 2,000 sales grouped as Other |

**No look-ahead.** A station counts only from its announcement date; before that, a sale knows nothing about it. Future years use the planned opening dates, so a 2031 price already includes, for example, the Cross Island Line if the block is near it. In the report, the panel under the block box shows, for the chosen sale year: the nearest open MRT, any closer announced station (with its opening year and which half of the wait it is in), the MRT effect in that year and today, the CBD, mall and school distances with their dollar effects, and the flat type and model effects.

#### Lease decay, measured from past sales
**The problem.** Models A–D gave every year of age the same cost (−$3.1k a year in D). Splitting D's errors by flat age showed the straight line was wrong at both ends:

| Flat age | Sales | Bias, straight line (D) | Bias, curve (F) | MAE D | MAE F |
|---|---|---|---|---|---|
| 0–9 | 109,944 | +$5.8k | +$0.6k | $53.7k | $51.9k |
| 10–19 | 111,450 | +$5.2k | −$0.4k | $54.1k | $52.9k |
| 20–29 | 147,425 | −$7.1k | −$0.8k | $51.9k | $51.5k |
| 30–39 | 112,756 | −$16.3k | +$0.1k | $55.2k | $54.5k |
| 40–49 | 45,710 | **+$23.2k** | +$1.8k | $62.0k | $53.5k |
| 50+ | 6,938 | **+$86.8k** (24% of the price) | +$1.8k | $101.7k | **$52.4k** |

(Fitted on all years; + = model too high.) The straight line undervalued flats aged 20–39 and badly overvalued flats past 40, because real prices hold up through the middle years and then fall faster as the remaining lease drops below about 60 years: buyers can use less CPF, banks lend less, and the next buyer pool shrinks.

**The fix.** The age slope may bend every 5 years from 10 to 55. Columns `age_over_10` … `age_over_55` count the years past each age (0 before it), so each 5-year band gets its own yearly loss, **learned from every past sale at that age** with size, storey, town, location, flat type and the market level held constant. No decline rate is assumed; history sets it. With the curve, every age band's bias is within ±$2k, and the error for 50+ flats halved ($102k → $52k).

**What history says** (model F, all years; $k, same flat in the same place and market):

| Age | Value vs a new flat | Yearly loss over the next 5 years | Same, sales since 2017 only | Straight line (D) |
|---|---|---|---|---|
| 0 | 0 | −2.3 | −4.3 | −3.1 |
| 10 | −23 | −5.2 | −10.9 | −3.1 |
| 20 | −86 | +3.1 | −1.1 | −3.1 |
| 30 | −88 | −6.6 | −2.7 | −3.1 |
| 35 | −121 | −8.4 | −7.7 | −3.1 |
| **40** | −163 | **−12.8** | −10.8 | −3.1 |
| **45** | −227 | **−13.4** | −12.8 | −3.1 |
| **50** | −294 | **−13.9** | −12.7 | −3.1 |
| **55** | −364 | **−16.6** | **−22.3** | −3.1 |
| 60 (oldest flat sold) | −447 | −16.6 (extrapolated) | −22.3 (extrapolated) | −3.1 |

- Past 40, each year costs **$13–17k**: about 5× a new flat's yearly loss ($2.3k) and about twice the rate at 10–20 years — the sharp decline is measured, not assumed.
- Since 2017 the curve is steeper at almost every age, consistent with buyers becoming more lease-conscious.
- **The hump at 20–25 years** (a small rise with age) is not a flat getting younger. It most likely reflects block upgrading (HIP / MUP, done around the 20–30-year mark) and the larger 1980s–90s flat designs that sell at that age.
- **Beyond 60 years the curve is extrapolated**: the oldest flat ever sold was 60, so a 2036 forecast for a flat that is 55 today (65 then) continues the last slope. Plausible as loan and CPF limits tighten, but unconfirmed by any sale; the uncertainty there is wider than the average ±$64k.

**In the report**, the flat ages one year per sale year (`age = age now + (sale year − 2026)`, i.e. the remaining lease falls by the same amount), and each year costs what history says it costs at that age. From 2026 to 2036, a flat aged 10 today loses about $63k to ageing (10 → 20); a flat aged 38 today loses about $121k (38 → 48), against $31k for both on the straight line. The calculation line shows the age term as "age N, lease-decay curve". The 15 other drivers keep their straight age line. `hdb_location_model.py` uses the curve in the report only if its back-test beats model D.

#### Tested and not adopted: the July 2013 CPF / HDB-loan rule for older flats (`hdb_lease_rule_cv.py`)
**The question.** From **1 July 2013** the CPF limits that already applied to private property with less than 60 years of lease were extended to HDB flats, and HDB loans for such flats were restricted (MAS 2013, Annex IV "Use of CPF Funds and Provision of HDB Loans"). From **10 May 2019** the test became "the lease must cover the youngest buyer to age 95", pro-rated. Does the model need to know about this?

| Remaining lease at purchase | CPF from 1 Jul 2013 | HDB loan from 1 Jul 2013 | From 10 May 2019 |
|---|---|---|---|
| 60 years or more | full, up to the valuation limit | normal | full if the lease covers the youngest buyer to 95 |
| 30–59 years | only if youngest buyer's age + lease ≥ 80; pro-rated cap | only if the lease covers the buyer to 80 | pro-rated towards 95 |
| 20–29 years | not allowed | only if the lease covers the buyer to 80 | CPF allowed down to 20 years, pro-rated |
| under 20 years | not allowed | no HDB loan | not allowed |

In this data only the 60-year line matters: the oldest flat sold was about 60, with about 39 years left.

**Why it looked worth testing.** Each replay version learns its own lease-decay curve, and the old-flat end steepened version after version while the young-flat end barely moved (age effect vs a new flat, $k):

| Trained to | Age 10 | Age 20 | Age 30 | Age 40 | Age 50 |
|---|---|---|---|---|---|
| 2006 | −48.5 | −71.3 | −71.5 | −66.8 | −43.8 |
| 2010 | −47.9 | −81.6 | −88.7 | −108.7 | −170.5 |
| 2015 | −45.7 | −80.6 | −100.7 | −124.5 | −209.5 |
| 2019 | −45.3 | −84.3 | −108.5 | −142.2 | −231.9 |
| 2026 (report model S) | −49.5 | −93.4 | −116.4 | −178.4 | −291.0 |

**Not a policy round.** The rule only touches older flats, so it was tested as a **lease-decay effect**. Round P15 (TDSR, 28 June 2013, +3 points) stays in the market-wide stance unchanged. Two candidates, both known at the sale date:
- **W** = S + `lease_lt60_13`: 1 for sales from 1 Jul 2013 with remaining lease (99 − flat age) under 60 years.
- **X** = S + `age13_over_35` … `age13_over_55`: extra lease-decay bends counted only for sales from 1 Jul 2013.

Same rule as every candidate, fixed before the run: time-series cross-validation on 2015–2019, W or X replaces S only if it beats S by $0.5k; then the 2020–2026 benchmark, scored once.

| Model | 2015 | 2016 | 2017 | 2018 | 2019 | **Validation mean** | Sealed MAE | RMSE | MAPE | Bias |
|---|---|---|---|---|---|---|---|---|---|---|
| **S (page model)** | 42.9 | 55.2 | 51.4 | 49.9 | 62.8 | **$52.4k** | **$75.2k** | $107.4k | 12.1% | −$50.1k |
| W: + lease under 60 from Jul 2013 | 43.8 | 55.3 | 51.6 | 50.0 | 62.7 | $52.7k | $75.1k | $107.3k | 12.1% | −$50.1k |
| X: + extra bends from Jul 2013 | 44.3 | 55.5 | 51.6 | 49.9 | 62.5 | $52.8k | $75.1k | $107.3k | 12.1% | −$50.5k |

**Result: S kept.** W and X are $0.3k and $0.4k *worse* on validation ($0.5k better was required); their $0.1k gain on the 2020–2026 benchmark is noise.

**What they learned.** W: an affected flat sold from July 2013 sells for **−$23.0k** on top of the curve. X: from July 2013 a 40-year-old flat is valued $43k lower than before (−$185k vs −$142k vs a new flat) and a 45-year-old $41k lower; by 50 the curves meet again.

**Why it adds nothing.** Flats with under 60 years left were 0–5% of sales before 2013 and 28% by 2026, so there are almost no pre-rule old-flat sales to contrast with, and S's single curve is already shaped mainly by the restricted era; W's step also overlaps with S's bends at 40 and 45. The effect is real and explains *why* lease decay steepened, but the forecast already carries it. The run also reproduced every earlier figure exactly (666,252 sales, S validation $52.4k, benchmark $75.2k, bias −$50.1k).

#### What buyers actually paid: old vs new flats, 2001–2026 (`hdb_age_gap_check.py`)
The model holds town, location, size and the market constant. This simpler check shows what buyers actually paid: the **median resale price per m²** of each flat-age band (age at the sale), year by year, straight from `hdb_resale_all.csv`. Price per m² removes the size difference between old and new flats; cells with fewer than 100 sales are blank.

**Sales behind each band** (flats aged 40+ were almost never sold before 2007, 50+ only from 2017, which is why the early replay versions had to guess the old end of the curve):

| Year | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 | 50+ |
|---|---|---|---|---|---|---|
| 2006 | 7,576 | 6,609 | 10,032 | 3,170 | 2 | 0 |
| 2007 | 7,404 | 6,634 | 9,458 | 3,256 | 210 | 0 |
| 2010 | 5,351 | 9,522 | 12,305 | 6,565 | 1,073 | 0 |
| 2013 | 1,299 | 4,775 | 5,579 | 3,557 | 876 | 0 |
| 2017 | 2,781 | 5,078 | 4,377 | 6,150 | 1,823 | 128 |
| 2020 | 6,436 | 2,796 | 4,250 | 6,263 | 3,132 | 436 |
| 2026 | 4,071 | 3,203 | 4,133 | 3,693 | 4,660 | 1,341 |

**Old flats compared with new ones** (median price per m²):

| Year | New 0–9 | 40–49 | 40–49 as % of new | 50+ as % of new | Gap, 90 m² flat |
|---|---|---|---|---|---|
| 2007 | $2,695 | $2,768 | 102.7% | | +$7k |
| 2008 | $3,303 | $3,542 | 107.2% | | +$22k |
| 2009 | $3,527 | $3,839 | 108.8% | | +$28k |
| **2010** | $4,050 | $4,500 | **111.1%** | | **+$40k** |
| 2011 | $4,659 | $5,125 | 110.0% | | +$42k |
| 2012 | $5,118 | $5,523 | 107.9% | | +$36k |
| **2013** | $5,981 | $5,641 | **94.3%** | | **−$31k** |
| 2014 | $5,860 | $5,231 | 89.3% | | −$57k |
| 2015 | $5,870 | $5,108 | 87.0% | | −$69k |
| 2016 | $5,237 | $4,933 | 94.2% | | −$27k |
| 2017 | $5,160 | $4,694 | 91.0% | 90.4% | −$42k |
| 2018 | $5,000 | $4,481 | 89.6% | 88.1% | −$47k |
| 2019 | $4,896 | $4,224 | 86.3% | 82.4% | −$60k |
| 2020 | $5,090 | $4,239 | 83.3% | 79.2% | −$77k |
| 2021 | $5,598 | $4,763 | 85.1% | 82.0% | −$75k |
| 2022 | $6,202 | $5,221 | 84.2% | 82.2% | −$88k |
| 2023 | $6,618 | $5,470 | 82.7% | 80.7% | −$103k |
| 2024 | $7,087 | $5,769 | 81.4% | 80.8% | −$119k |
| 2025 | $7,706 | $6,119 | 79.4% | 78.1% | −$143k |
| **2026** | $8,011 | $6,022 | **75.2%** | **72.4%** | **−$179k** |

**Growth since 2007** (price per m² as % of its 2007 level):

| Year | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 |
|---|---|---|---|---|---|
| 2009 | 131 | 131 | 132 | 134 | **139** |
| 2011 | 173 | 168 | 168 | 167 | **185** |
| 2013 | 222 | 190 | 187 | 182 | 204 |
| 2015 | 218 | 170 | 160 | 157 | 185 |
| 2019 | 182 | 171 | 150 | 138 | 153 |
| 2023 | 246 | 276 | 208 | 186 | 198 |
| 2026 | 297 | 323 | 239 | 207 | 218 |

**Median 4-room price ($k)**, like for like: 0–9 years $250k → $729k (2007 → 2026); 40–49 years $463k (2015) → $550k (2026); 50+ years $508k (2024) → $530k (2026).

**What it shows.**
- **Before 2013, old flats sold at a premium and rose fastest.** 2007–2011: 40–49-year-old flats +85% per m², new flats +73%. In 2010 a 40-year-old flat cost 11% more per m² than a new one (+$40k on 90 m²).
- **In 2013 the premium became a discount, and it has widened since:** 94% of a new flat's price per m² in 2013, 83% in 2020, 75% in 2026 (−$179k on 90 m²). Since 2019: new flats +63%, 40–49-year-old flats +43%.
- **The turn is the year of the July 2013 CPF and HDB-loan limits** for flats with under 60 years left, i.e. every flat aged 40+. Other 2013 measures (TDSR, the MSR cut) applied to all flats.

**Why the model's age curve was negative even in 2010.** These medians do not hold location constant. In 2007–2012 almost every 40–49-year-old flat stood in the oldest, most central estates (Queenstown, Toa Payoh, Bukit Merah …), near the CBD, while most new flats were in outer towns; much of the "premium" was location. At the same town and distance to the CBD, the model found age costing money even in 2010. After 2013 the raw discount and the model's steeper curve agree, which is also why W and X added nothing: S's curve is learned mostly from post-2013 sales.

**Limits.** 2013 had few new-flat sales (1,299 vs 5,351 in 2010), so that year's "new" median is noisy; the 2013–2026 trend is solid. Timing points to the rule but does not prove cause: these are raw medians.

#### Policy stance: does the 1–3 judgement matter, and is there a later effect? (`hdb_policy_stance_cv.py`)
Two questions about the policy stance in model S, tested with the same rule as every candidate, fixed before the run: time-series cross-validation on 2015–2019, a candidate replaces S only if it beats S by $0.5k, then the 2020–2026 benchmark is scored once for all. The script first checks it rebuilds `tight36` exactly (100.0% of sales).
- **Y** = S with the 1–3 intensity replaced by a rule nobody chooses: the number of instruments in the round, capped at 3 (matches the hand score for 17 of the 30 rounds).
- **Z** = S with `tight36` replaced by the stance from rounds announced 48–84 months (4–7 years) before the sale; **S+Z** = S plus that column.

| Model | Validation mean 2015–2019 | Benchmark MAE 2020–2026 | Bias | Stance coefficient ($k per point) |
|---|---|---|---|---|
| **S** (1–3 judgement) | **$52.4k** | $75.2k | −$50.1k | `tight36` +4.99 |
| Y (instrument count) | $54.3k | $74.2k | −$48.9k | `tight36_cnt` +5.45 |
| Z (rounds 4–7 years before) | $65.1k | $69.7k | +$6.0k | `lag4_stance` −4.84 |
| S+Z | $55.5k | $68.3k | −$28.6k | `tight36` +4.59, `lag4_stance` −4.47 |

**S is kept**, by the rule. The results are **broadly robust to replacing the judgement**: Y learns almost the same coefficient and is $1.9k worse on validation, $1.0k better on the benchmark (no significance test was run). The 4–7-year stance has a **negative** coefficient, the opposite of the 0–3-year stance, and S+Z is $6.9k better on the benchmark, but it is **$3.1k worse on validation**, so it is a promising association that did not replicate on validation; choosing it for its 2020–2026 score would mean choosing by the benchmark. It is to be re-tested once on 2027 sales with the rule written down in advance. (In earlier drafts these candidates were called C, L and SL; those letters already meant other models in the comparison above, so they were renamed Y, Z and S+Z.)

#### Lesson 3.5 tools on the model (`hdb_unsupervised_cv.py`)
The three unsupervised tools from lesson 3.5, each fitted **inside every fold** on that fold's training years only, under the same rule (beat S by $0.5k on validation):
- **KM8, KM16, KM32**: k-means location segments (8, 16 or 32) added as 0/1 columns.
- **ISO01, ISO1**: Isolation Forest removes the 0.1% or 1% most unusual **training** sales.
- **PCA95**: the 26 numeric columns replaced by the components that keep 95% of their variation (13–14 components).

| Model | Validation mean | Benchmark MAE | Bias |
|---|---|---|---|
| **S** | **$52.4k** | $75.2k | −$50.1k |
| KM8 | $52.3k | $75.1k | −$49.9k |
| KM16 | $52.2k | $74.8k | −$49.5k |
| KM32 | $52.0k | $74.6k | −$49.2k |
| ISO01 | $52.6k | $75.4k | −$50.8k |
| ISO1 | $53.3k | $76.3k | −$52.8k |
| PCA95 | $54.9k | $91.3k | −$78.1k |

**S is kept.** K-means helps a little but stays short of the $0.5k bar ($0.4k): towns and distances already describe location. Removing unusual sales makes the forecast worse, because they are real sales (big, central, high-floor or rare flats). PCA is much worse ($16.1k on the benchmark): it keeps what varies most among the inputs, not what predicts price.

#### The course's own 2017–2019 data (`hdb_course_compare.py`)
The course notebook predicts 2017–2019 resale prices from town, flat type, floor area, storey and lease start, with one random 80/20 split (`random_state=42`). The script keeps the course's data, split and models and only adds this project's flat and location features (flat model, flat age with the 10 lease-decay bends, km to the CBD, a mall and a popular school, the school-within-1 km flag, km to the nearest MRT station open at the sale month, a time trend).

| Test | Model | Course features | With this project's features |
|---|---|---|---|
| Random 80/20, `random_state=42` | Linear regression | $46.7k | $36.6k |
| Random 80/20, `random_state=42` | Random forest | $25.1k | $18.3k |
| Random 80/20, `random_state=42` | LightGBM | $24.3k | $17.6k |
| Train 2017–2018 → test 2019 | LightGBM | $27.6k | $18.8k |

Run of 1 November 2026, with malls and schools dated as in model S (the undated columns give $17.5k and $19.0k for LightGBM). **One feature group at a time** (test 4 of the script): each row is a separate model, trained and scored on the same rows, so comparing a row with the one above shows what that group adds (MAE in $):

| Features | Linear, random split | Linear, 2019 test | LightGBM, random split | LightGBM, 2019 test |
|---|---|---|---|---|
| course features | 46,726 | 48,434 | 24,329 | 27,623 |
| + flat model | 44,182 | 45,752 | 23,674 | 26,135 |
| + lease-decay bends | 42,692 | 43,041 | 23,127 | 24,896 |
| + CBD, mall, school | 40,306 | 40,794 | 17,765 | 19,408 |
| + MRT | 36,627 | 37,555 | 17,701 | 19,248 |
| + time trend (all of ours) | 36,572 | 37,581 | 17,566 | 18,730 |
| course + CBD, mall, school only | 44,355 | 45,906 | 18,941 | 22,159 |
| course + MRT only | 41,718 | 43,706 | 19,935 | 22,922 |

Every group lowers the error in all four columns, except the time trend for linear regression on 2019 ($26 worse, i.e. flat).

The added features win on every one of five random seeds and on the next-year test; location (CBD, mall, school, MRT) gives the largest single gain. This comparison does not use the 2020–2026 benchmark at all. It also shows why this project's errors look larger than the course's: the course scores a random split inside a calm three-year window, while this project forecasts years ahead through the 2020–2026 boom.

#### What the model cannot see
- **Renovation and unit condition.** Two 40-year-old flats in the same block can sell $50–100k apart if one is fully renovated. HDB's resale data records no interior condition, so this stays in the error: even after the age curve, the miss is 11% of the price for new flats and 13–15% for flats aged 30–49.
- **Unit position**: corner unit, facing, view, distance to the lift or chute (the data only gives a storey range).
- **Block upgrading** (HIP / MUP / LUP): HDB publishes the upgraded blocks, so this could be added as a future feature.

#### MRT station table (`mrt/mrt_stations_corrected.csv`, 184 stations)
Built from a station timeline (`mrt_station_phase_timeline_master_v4.csv`) and OneMap geocodes, with two kinds of fixes. These errors also affect the MRT-at-sale-date columns of the geocoded master table (V3F): in 2010 it put Pasir Ris flats 3.4 km from the nearest open MRT (Simei), when Pasir Ris station had been open since 1989. With the corrected table, the 2010 medians are 0.7–1.2 km for Clementi, Jurong East, Pasir Ris, Tampines and Woodlands.
- *Opening dates.* The timeline listed many interchanges only under their newer line (Jurong East 2028, Clementi and Pasir Ris 2032, Tampines 2017, Woodlands 2020, Orchard 2022) and gave whole lines one placeholder date (Admiralty–Yew Tee shown as 1987, opened 1996; Choa Chu Kang and Bukit Batok 1990; Boon Lay 1990; Buangkok 2006; Woodleigh 2011; most Circle Line stations 2010–11, not 2009). Every NSL, EWL, Changi branch, NEL and Circle Line station now takes its own first opening date from Wikipedia's station list (`mrt/legacy_lines_wikipedia.csv`; 5 dates blank on that list are filled from their section's opening and marked). DTL, TEL, JRL and CRL keep the timeline's dates. Announcement dates are the timeline's (Expo takes the Changi Airport branch's). "Jurong Spur" in the timeline is a placeholder, not a station, and is left out.
- *Coordinates.* Orchard, Tampines and Punggol were geocoded to Orchard Boulevard, Tampines East and Punggol Coast (0.9–1.4 km away); West Coast and Serangoon North to a church and an animal clinic; 20 more (8 OneMap NOT_FOUND, 12 missing from the reference) had none. All now come from each station's Wikipedia page (`mrt/station_coords_fetched.csv`, with source links). Riviera's point is probably the existing LRT station; Elias and Jurong Town Hall are only accurate to about 100 m.

#### Blocks
`hdb_block_lookup_from_V3F.csv` (9,748 blocks, from the geocoded master table) is joined to every sale from 2001 by block + street. The first join matched 98.9%; the misses were two groups:
- **90 demolished / SERS blocks** (no sale after 2012) — e.g. Rochor Centre (Rochor Rd 1–4), Tanglin Halt, Ghim Moh Rd, Clementi Ave 1 blocks 401–409, Redhill Cl, Boon Lay Dr 167–172. OneMap only returns buildings that still exist, so they had no coordinates. `fill_missing_blocks.py` gives each the location of the surviving block on the same street with the closest block number (flagged approximate, about 100 m); 84 filled this way.
- **220 new blocks** first resold in 2024–26 after their 5-year minimum occupation period (e.g. 82B Circuit Rd, St. George's Lane, Canberra St), built after the master table. All 220 geocoded with OneMap (block number must match exactly). Their CBD, mall and school distances were first copied from the nearest existing block; the check below showed that this was too rough in new estates, so they are now calculated from each block's own coordinates. MRT distances are always recomputed from the coordinates.

6 long-demolished blocks (Kg Bahru Hill, Jln Pasar Baru; 39 sales) stay out. **Match rate: 666,252 of 666,291 sales (100.0%).**

#### How the master table was geocoded
The block lookup comes from an earlier stage of this project (the HDB master table, versions V3B to V3F, built in Google Colab):
- **Block coordinates:** every HDB address was geocoded with the **OneMap Search API** (Singapore Land Authority), merged by postal code first and through a saved geocode cache, so each address is looked up only once. The first pass matched about 91%; staged recovery of the cache matches and a re-run of the true leftovers brought it to full coverage.
- **Areas:** URA subzone, planning area, region and district come from **point-in-polygon** matching of each block's coordinates against the official boundary files.
- **Distances:** straight-line (haversine) distance from each block to the nearest point in three reference layers: MRT stations (`mrt_reference_current_plus_2030.xlsx`, now replaced by the corrected table above), **shopping malls**, and **primary schools**. The CBD distance is to a point beside Raffles Place.
- **"High-demand" primary schools** are the **Phase 2B** schools of the P1 registration exercise (the `…_SCHOOL_POINT_2B_ONLY` version of the master table), with 1 km and 2 km flags; the report uses the 1 km flag.
- **Known simplification:** malls and schools are placed where they are today; relocated schools and malls opened after a sale are not time-adjusted (see Limitations).

#### How the distances are measured (MRT, CBD, malls, primary schools)
Every distance in the model is a **straight-line ("as the crow flies") distance** between two points given by latitude and longitude, computed with the **haversine formula** ("half versed sine", hav θ = sin²(θ/2)): the shortest path over the Earth's curved surface. It measures closeness, not travel time; a block across an expressway can be near in a straight line but slower to reach by road. Over Singapore's short distances that is a fair proxy, it is exact and free (no paid map service), and the coefficients stay in plain units ($k per km).

Two different geometry tools are used, for two different questions:

| Question | Tool | Gives | Used for |
|---|---|---|---|
| **Which area** is the block **in**? | **Point-in-polygon**: is the block's point inside an area's outline? | a name | URA subzone, planning area, region and district |
| **How far** is the block **from** something? | **Haversine distance** between two points | a number of km | MRT, CBD, mall, primary school |

| Feature | Measured to | How | Known at the sale date? |
|---|---|---|---|
| `mrt_open_km` | the nearest MRT/LRT station **already open in the sale month** | haversine from the block to every station in `mrt/mrt_stations_corrected.csv` (184 stations, coordinates checked against each station's Wikipedia page), keep the nearest open one; capped at 3 km (beyond that, another km changes little) | yes: rebuilt for every sale month from the opening dates |
| `mrt_new_gain` | the nearest station **announced but not yet open** | how many km closer it is than the nearest open station (0 if it is not closer) | yes: counts only from the announcement date |
| `stage_early` / `stage_late` | that announced station | 1 if the sale is in the first / second half of the station's announcement-to-opening wait | yes |
| `dist_to_cbd_km` | **one fixed point in Raffles Place**, about 61 m from Raffles Place MRT (1.2840, 103.8514) | haversine; the same point for every block, so each block has its own distance (Tanjong Pagar about 1 km, Tampines about 14 km). The column came with the block master table; `check_block_lookup.py` worked backwards from it to the point it was measured to, with a typical miss of 1 m, and confirmed it is straight-line, not road, distance (column ÷ straight line = 1.00) | the CBD does not move |
| `dist_to_mall_km` | the nearest of 41 shopping malls (point) | haversine | **yes** (since 31 Oct 2026): a mall counts from its opening month, closed months excluded |
| `dist_to_nearest_high_demand_primary_school_km` | the nearest of 75 popular ("Phase 2B" high-demand) primary schools (point) | haversine | **yes** for the school (from the month it started at its home site); the list of popular schools is today's |
| `within_1km_high_demand_primary_school_flag` | the same school | 1 if that distance is 1 km or less (a 1 km circle around the school point) | yes, as the distance; see below |

The positions of the 41 malls and 75 schools were worked out from all the blocks that name them (like locating a phone from several masts) and cross-checked against OneMap; see the next section.

**The 1 km school flag and MOE's rule.** The flag mirrors MOE's Primary 1 registration priority for homes within 1 km of a school. MOE measured that distance from the **school's centre point** up to the 2021 exercise. From the **2022 Primary 1 Registration Exercise** (announced 9 September 2021), it is measured from the **School Land Boundary**, "from any point on the boundary around the school to the registrant's home" ([MOE press release](https://www.moe.gov.sg/news/press-releases/20210909-changes-to-the-primary-one-registration-framework-to-ensure-our-schools-remain-open-to-all)). Because the boundary is closer than the centre, more homes now count as within 1 km. The flag here uses the centre-point rule, which matches what buyers knew for sales before September 2021; for later sales, blocks near the 1 km edge may be misclassified. The fix is a time-gated flag, like the MRT columns: keep the centre-point circle before September 2021, and from then on use **point-in-polygon** against a 1 km buffer drawn around each school's land parcel (the "Educational Institution" parcel of URA's Master Plan land-use map that contains the school's point). Planned, to be tested under the same rule as every candidate.

#### Checking the school boundary and dating the malls and schools
How the popular schools were chosen, how to date the malls and schools like the MRT stations, and how to check the school
land boundary used for the 1 km rule since the 2022 Primary 1 exercise, when no boundary download exists. Planned for after the capstone.

##### Popular primary schools: how the 75 were chosen
- The 75 "high-demand" schools are the **Phase 2B schools** of the Primary 1 registration exercise, from the `…_SCHOOL_POINT_2B_ONLY` version of the block master table, with 1 km and 2 km flags; the model uses the 1 km flag.
- **What "Phase 2B" means:** P1 registration runs in phases (1, 2A, 2B, 2C, 2C Supplementary). Balloting happens in a phase **when applicants exceed the vacancies left**. A school that is already full, or balloting, by **Phase 2B** has run out of places before the general public (Phase 2C) can apply: a common working definition of a popular school.
- **What is not recorded:** the master table does not say **which year's** registration results the list came from, or the exact rule (for example "balloted at Phase 2B or earlier" vs "full after Phase 2B"). So the list is **today's view of popularity**, applied to every year.
- **Why 1 km matters:** when a phase is balloted, MOE gives priority in this order: Singapore Citizens within 1 km, then 1–2 km, then beyond 2 km (then Permanent Residents in the same order). A home within 1 km of a popular school gets the best chance, which buyers pay for.

##### How to check which schools were popular, year by year, and when they opened
| Question | Source | How to use it |
|---|---|---|
| Was school X popular in year Y? | **sgschooling.com**, *P1 Ballot History*, **2009–2026**, by area and school: for each phase, the vacancies, applicants and places taken | rule fixed in advance: school X is popular in year Y if its **applicants exceeded vacancies at Phase 2B or earlier**. This gives a popular list for **every** year instead of today's |
| Does it match the 75? | the same tables | compare the 75 names with each year's list: the year that matches best shows which year the master table used |
| When does popularity become known? | MOE's timetable: registration runs in **July–August** | a year Y list applies to sales **from September of year Y**, so a sale never knows a result published after it |
| When did the school open, move or merge? | MOE press releases on new schools, relocations and mergers (e.g. the mergers announced in 2018–2019); each school's own history page; the newspaper archive | a school counts only from its opening (or move-in) date at that site; a merged school stops counting at the old site from the merger date |

The result is `schools_dated.csv` (school, site, latitude, longitude, opened, closed, popular_from, popular_to, source), and the school flag becomes "within 1 km of a school **that was open and popular at the sale date**".

##### Bands and announced moves: the MRT idea for popular schools
The MRT features carry over, with one difference in **shape**: P1 admission works in **bands**, not metres.

- For MRT every 100 m closer helps, so `mrt_open_km` is a **continuous** distance, capped at 3 km.
- For admission, MOE's ballot priority goes by **bands**: within 1 km, then 1–2 km, then beyond 2 km. Inside a band **everyone has the same chance**: 300 m from the school is no better than 900 m. So the features are **0/1 bands**, not a distance capped at 1 km.

| Feature | 1 if the block is… | Why |
|---|---|---|
| `pop_1km` | within 1 km of a school **popular that year** | top priority band |
| `pop_2km` | 1–2 km from one (and not within 1 km) | second band: still a better chance than beyond 2 km |
| (neither) | beyond 2 km of every popular school | no admission advantage: the base |

**Only popular schools count.** At a school with spare places everyone who applies gets in, so distance gives no advantage; that is why the list must be **popular that year**, from the ballot history.

**The "announced" part, as with MRT:**

| MRT | Popular school |
|---|---|
| station announced, not yet open | a **popular school's move to a new site is announced** (MOE announces relocations years ahead) |
| `mrt_new_gain`: km closer than the nearest open station | `pop_1km_announced`: 1 if the **new site** puts the block within 1 km and the current site does not |
| counts from announcement to opening | counts from the **announcement** to the **move-in year**; after that the block simply gets `pop_1km` |
| `stage_early` / `stage_late` | the same split of the wait, if the data supports it |

**A brand-new school is not "popular" when it is announced:** nobody knows yet whether it will be oversubscribed, and many new schools have vacancies in their first years. Rule, fixed in advance: a new school counts as popular only **after** a registration exercise shows it balloting by Phase 2B. Only an **already-popular school moving site** gets the "announced" treatment.

**All the rules together**, each known at the sale date:

| Rule | Detail |
|---|---|
| Popular that year | the latest registration exercise finished before the sale (registration ends in August) |
| Open at that site | from the opening or move-in date |
| 1 km by the rule in force | the school's point before September 2021; its land boundary from September 2021 |
| Announced moves | counted from the announcement to the move-in |

**Judged like every candidate:** `pop_1km`, `pop_2km` and `pop_1km_announced` replace today's flag and school distance only if they beat the current model on the 2015–2019 validation by the set margin.

##### How the mall opening dates can be checked
The fix is the same as for MRT: a mall counts only from the month it **opened** (and stops counting if it **closed** or was redeveloped).

| Source | What it gives | How to use it |
|---|---|---|
| Wikipedia, *List of shopping malls in Singapore*, and each mall's own page | opening year (often the month), renovations, closures | first pass: an opening year for each of the 41 malls |
| Newspaper archive (ST, BT, Zaobao, Shin Min) | the opening announcement and date, e.g. "opens on 12 June" | search each mall's name; the earliest "opens" or "opening" article gives the month |
| Owner or developer releases (e.g. CapitaLand, Frasers) | official opening date, major refurbishments | to settle conflicts between the first two |
| URA / HDB news releases | town-centre malls built with the estate | for older malls, the town-centre completion year |

Rules, written before testing: the **official opening month** counts; a **major redevelopment** that closed the mall counts as closed from its closing month to its reopening; a **renaming** keeps the same mall. The result is a small table, `malls_dated.csv` (name, latitude, longitude, opened, closed, source), and the mall distance becomes "km to the nearest mall **open in the sale month**", exactly like `mrt_open_km`.

##### How to check point-in-polygon for the school boundary when no boundary download exists
MOE's school land boundaries are **not published as a download**. The official answer for one address is **OneMap SchoolQuery**, which shows the home's distance **category** (within 1 km, 1–2 km, beyond 2 km) for each school; MOE calls it the official measurement. It works one address at a time, so it is the **check**, not the source. The source is built from two open datasets:

| Step | Data (data.gov.sg) | What to do |
|---|---|---|
| 1. School land | **Master Plan 2019 Land Use layer** (URA, GeoJSON, about 166 MB): every land parcel with its land-use type | for each school, take the land-use parcel **that contains the school's point** (point-in-polygon), of the educational-institution type: that is the school's land |
| 2. Home outline | **HDB Existing Building** (HDB, GeoJSON, about 54 MB): the outline of every HDB block, with the block number (`BLK_NO`) and a street code (`ST_COD`) | match each block by **point-in-polygon**: the block's geocoded point falls inside its own outline |
| 3. Distance | both outlines, converted to metres (Singapore's SVY21 grid, EPSG:3414) | the **shortest distance between the two outlines**, as MOE measures it |
| 4. Flag | | within 1 km if that distance ≤ 1,000 m. Equivalently, draw a 1 km buffer around the school's land and test whether the block's outline touches it (point-in-polygon on the buffer) |
| 5. Check | **OneMap SchoolQuery**, by hand | pick a sample of blocks **near the edge** (for example 0.9–1.2 km by the old point distance) for several schools; record whether the built flag agrees with OneMap's category; report the agreement rate and every mismatch |

**Known gaps, said in advance:** a Master Plan parcel may include a neighbouring field or a second school on the same site; the Master Plan is the 2019 plan, not each year's boundary; MOE uses the building plan's outline, which can differ slightly from HDB's mapped outline. The OneMap sample measures how much these gaps matter.

**The time-gated flag:** sales **before September 2021** keep the point-based 1 km circle (what buyers knew then); sales **from September 2021** use the boundary-based flag. Combined with the yearly popular lists and the opening dates, the flag becomes: within 1 km, **by the rule in force**, of a school **open and popular at the sale date**.

**How it will be judged:** the dated mall distance and the time-gated school flag are new candidates, scored on the 2015–2019 validation years with the same rule as every other (a candidate replaces the current model only if it beats it by the set margin), and the affected models rerun and rescored, as promised in the audit response.

#### How the geocoding was checked (`check_block_lookup.py`)
The CBD, mall and popular-school columns come from the geocoded master table and are used as they are (MRT is rebuilt for every sale month, see above). To make sure they can be trusted, every block was tested in five ways, with no paid map service:

| # | Check | How | Result (9,957 blocks) |
|---|---|---|---|
| 1 | **CBD distance** | recalculated from each block's latitude/longitude, and the CBD point that best fits the column was found | the column is straight-line distance to a point **61 m from Raffles Place**; typical miss **1 m** |
| 2 | **School flag** | "within 1 km of a popular school" must be 1 exactly when the school distance is 1 km or less | **0** disagreements |
| 3 | **Distance kind** | column ÷ straight-line distance | **1.00**: straight-line, not road distance |
| 4 | **Mall and school positions** | each block names its nearest mall and school with a distance, so the position of each of the **41 malls and 75 schools** can be worked out from all the blocks that name it (like locating a phone from several masts); blocks that do not fit, or where another mall/school is clearly nearer, are flagged | typical miss **0.000 km**; **0** blocks with a nearer mall or school than the one named |
| 5 | **Neighbours** | two blocks under 150 m apart naming the same mall cannot differ by much more than 150 m in distance to it (triangle rule) | **0** pairs break it |

**What the check found, and the fix.** On the first run, **52 blocks (0.5%)** failed checks 1, 4 or 5, and **every one was a block added by `fill_missing_blocks.py`**, not an original block: mostly new BTO blocks in Punggol, Sengkang and Toa Payoh, whose distances had been copied from an existing block too far away (e.g. 422B Northshore Dr, 0.9 km off on CBD distance). `fix_filled_blocks.py` recalculated the CBD, mall and school columns of all 222 filled blocks from their own coordinates, using the CBD point and the 41 mall / 75 school positions worked out from the original blocks. The original blocks were not changed (a `.bak` copy is kept). **After the fix: 0 of 9,957 blocks flagged.** The back-test stayed at $64.2k, as these blocks have few resales so far, but the report now shows the right distances for them.

**Independent cross-check with OneMap** (`check_block_lookup.py --onemap`). Every mall and school name was looked up on OneMap and the distances recalculated from OneMap's positions. 116 of 118 names were found and almost all agree; the differences come from OneMap's search, not the table: "Valley Point" matched *RV Point* (1.7 km away), "JCube" matched *Bugis Cube*, and several schools matched a tuition centre "@" the school (same site, 60–90 m). One label is under review: the 489 blocks naming *Hougang Mall* are all consistent with one mall position, about 1 km from OneMap's Hougang Mall, so the distances are sound but the name may refer to a neighbouring mall.

### Lesson 3.4: tree models compared with the plane (`hdb_tree_models.py`)
The same sales (2001–2026) and features as report model S, given to five tree models from lesson 3.4, plus two hybrids. The tree models get plain flat age instead of the age bands, so they find the lease-decay curve by themselves. Two tests, both on sales the models never saw:
- **Forecast:** trained 2001–2019, tested 2020–2026 (the report's benchmark)
- **Valuation:** a random 80% / 20% split of all sales, so the test sales come from years the model has seen

| Model | Forecast: train MAE | **Forecast: test MAE** | Forecast bias | **Valuation: test MAE** |
|---|---|---|---|---|
| **Linear plane S (page model)** | $33.6k | $75.2k | −$50.1k | $44.3k |
| Random forest (bagging, 150 trees on 150,000 sales) | $15.7k | $133.4k | −$130.3k | $19.9k |
| Gradient boosting | $12.2k | $135.6k | −$133.5k | $15.4k |
| XGBoost | $11.5k | $132.8k | −$130.6k | **$14.9k** |
| LightGBM | $12.2k | $134.0k | −$131.9k | $15.2k |
| Hybrid $: plane S + LightGBM correction in dollars (flat and location only) | $15.6k | **$64.8k** | −$52.9k | $21.3k |
| Hybrid %: plane S × LightGBM correction in percent (flat and location only) | $15.9k | $68.2k | −$57.3k | $22.2k |

Forecast test, MAE by year:

| Year | Plane S | LightGBM | Hybrid $ |
|---|---|---|---|
| 2020 | $51.8k | **$21.3k** | $25.4k |
| 2021 | $80.5k | **$60.9k** | $75.8k |
| 2022 | **$84.8k** | $109.3k | $85.8k |
| 2023 | $59.6k | $136.2k | **$49.1k** |
| 2024 | $67.8k | $178.2k | **$50.4k** |
| 2025 | $87.4k | $219.3k | **$80.6k** |
| 2026 | $96.1k | $227.9k | **$85.8k** |

**What it shows:**
- **Valuation: the trees are 3× more accurate** ($15k against $44k). Within years they have seen, they learn curves and combinations the plane cannot (a big flat in a mature town near an MRT is worth more than the parts added up).
- **Forecasting: the trees alone fail, and the failure grows every year.** A tree's price is an average of prices it has seen, so it cannot go above the 2001–2019 price levels. In 2020, when prices were still near 2019 levels, the trees beat the plane by about $30k; by 2025–26 they are $220k too low. The plane carries the market forward through income, which is why a plane stays at the core of the forecast.
- **Overfitting is not the cause:** the valuation test shows the trees generalise well within their period; what they cannot do is **extrapolate**.
- **With S as the plane, the dollar hybrid helps: $64.8k against $75.2k.** The plane S carries the market level and policy; LightGBM, given only the flat and its location, corrects where the plane misprices particular flats and places. It is the lowest forecast error in the project. With the earlier plane H the same hybrid made things worse ($78.5k), so the gain depends on the plane underneath.
- **The $64.8k alone was not enough to adopt it.** It was seen on 2020–2026, the test years, so the hybrid had to win the 2015–2019 time-series cross-validation first, as S did (next section).

#### Validating the hybrid without the test years (`hdb_hybrid_cv.py`)
The hybrid got the same procedure that chose S: for each validation year 2015–2019, train on all earlier years (2001 onward) and predict that year; choose on the average; only then score 2020–2026 once. The plane inside each fold is refitted on that fold's training years, and LightGBM learns the errors of that fold's plane. **The rule was set before the run:** a hybrid must beat S by at least **$0.5k** on the validation average, otherwise the simpler S stays. Saved to `hdb_data/hybrid_cv.csv` and `hdb_data/hybrid_sealed.csv`.

**Validation MAE by year ($k; 2020–2026 not used):**

| Model | 2015 | 2016 | 2017 | 2018 | 2019 | Mean |
|---|---|---|---|---|---|---|
| **Hybrid $: plane S + LightGBM dollars** | 31.5 | **52.6** | **39.0** | **30.3** | 41.6 | **39.0** |
| Hybrid %: plane S × LightGBM percent | **31.1** | 52.8 | 40.9 | 32.1 | **40.7** | 39.5 |
| Plane S (page model) | 42.9 | 55.2 | 51.4 | 49.9 | 62.8 | 52.4 |

Plane S scores the same $52.4k here as in `hdb_location_model.py`: the two scripts use the same sales and the same folds.

**2020–2026 benchmark.** The run with `--grid` scores only the model chosen after tuning (next section). With these default settings, the same hybrid in `hdb_tree_models.py` scores **$64.8k** on 2020–2026 (RMSE $86.2k, bias −$52.9k), against $75.2k for S. Test MAE by year: hybrid $25.4k / $75.8k / $85.8k / $49.1k / $50.4k / $80.6k / $85.8k (2020–2026), against S $51.8k / $80.5k / $84.8k / $59.6k / $67.8k / $87.4k / $96.1k. The hybrid is better in 6 of 7 years; in 2022 it is $1.0k worse.

**What it shows:**
- **The hybrid passes honestly.** It beat S by $13.4k on validation (needed $0.5k), in all 5 validation years, and then held its lead on the 2020–2026 benchmark. This is a choice made without the test years, unlike the first $64.8k.
- **It fixes the flat, not the market.** For the tuned hybrid (next section), RMSE falls from $107.4k to $86.2k and R² rises from 0.695 to 0.803: the big misses on particular flats shrink, because the trees learn the curves and combinations of size, age, town and location that the plane's straight lines miss. The **bias does not improve** (about −$50k to −$54k): both are too low on average in the boom, because the market level comes only from the plane's income, supply and policy terms. The trees were deliberately given no market features, since a tree cannot extrapolate above prices it has seen.
- **In one line: the plane forecasts the market; the trees correct the individual flat.**
- With the single-quarter RFS figures the hybrid lost 2015 by $2.2k; with the yearly totals it wins every validation year.

#### Tuning the hybrid's LightGBM settings with the same cross-validation (`hdb_hybrid_cv.py --grid`)
Four other settings for the dollar hybrid's trees were tried alongside the default, each chosen on the 2015–2019 validation years only:

| LightGBM setting (dollar hybrid) | What it changes | 2015 | 2016 | 2017 | 2018 | 2019 | **Mean** |
|---|---|---|---|---|---|---|---|
| **127 leaves per tree** | bigger trees, finer combinations | 32.3 | **52.1** | **37.6** | **29.3** | 41.6 | **38.6** |
| Slow: learning rate 0.02, up to 8,000 trees | smaller steps, more trees | 31.4 | 52.4 | 38.8 | 30.0 | 41.4 | 38.8 |
| Default: 63 leaves, learning rate 0.05, at least 40 sales per leaf | the settings used above | 31.5 | 52.6 | 39.0 | 30.3 | 41.6 | 39.0 |
| 31 leaves per tree | smaller trees | **30.3** | 52.6 | 40.6 | 31.4 | 41.9 | 39.4 |
| At least 400 sales per leaf | smoother, less detail | 30.8 | 54.1 | 40.7 | 30.9 | 41.6 | 39.6 |
| (Hybrid % with default settings) | | 31.1 | 52.8 | 40.9 | 32.1 | **40.7** | 39.5 |
| (Plane S) | | 42.9 | 55.2 | 51.4 | 49.9 | 62.8 | 52.4 |

Every model uses early stopping: 10% of the training rows tell LightGBM when more trees stop helping, so the number of trees is tuned inside each fold too.

**Chosen: 127 leaves (validation mean $38.6k). Sealed test, scored once:**

| Model | Mean validation MAE | Test MAE | Test RMSE | MAPE | Test R² | Bias |
|---|---|---|---|---|---|---|
| **Hybrid $, 127 leaves (chosen after tuning)** | **$38.6k** | **$65.0k** | **$86.2k** | **11.0%** | **0.803** | −$53.9k |
| Plane S (page model) | $52.4k | $75.2k | $107.4k | 12.1% | 0.695 | −$50.1k |

Test MAE by year for the tuned hybrid: $24.7k / $76.4k / $86.6k / $49.3k / $50.1k / $81.0k / $86.2k (2020–2026); again better than S in 6 of 7 years, all but 2022.

**What tuning shows:**
- **The settings matter little; the method matters a lot.** All five settings land within $1.0k of each other on validation ($38.6k–$39.6k), while every one of them beats plane S by more than $12k. The big gain comes from combining the plane with trees, not from fine-tuning the trees.
- **A small validation edge does not always survive.** 127 leaves beat the default by $0.4k on validation, but scored $65.0k on the 2020–2026 benchmark against the default's $64.8k (`hdb_tree_models.py`). A $0.4k difference is noise at this scale.
- **The tuned model is still the one reported.** Switching back to the default now, because its test score is $0.2k lower, would be choosing on the test years again. The honest result of the procedure is 127 leaves at **$65.0k**.

#### Where the models miss: error by segment (`hdb_explain_models.py`)
The 2020–2026 benchmark sales split into groups, for plane S and the tuned hybrid, both trained on 2001–2019 only (groups with at least 200 sales; MAE in $k, bias = average signed miss, negative = too low):

| Segment | Plane S MAE | Plane S bias | Hybrid MAE | Hybrid bias |
|---|---|---|---|---|
| 3 room | 44.6 | +10.7 | 41.6 | −22.6 |
| 4 room | 68.5 | −53.1 | 64.7 | −57.7 |
| 5 room | 97.6 | −87.3 | 80.9 | −74.7 |
| Executive | 146.8 | −142.7 | **97.6** | −89.9 |
| Actual price under $400k | 47.2 | **+23.1** | 39.4 | −11.8 |
| Actual price $400–500k | **38.0** | −1.9 | 43.5 | −30.3 |
| Actual price $800k–1m | 186.4 | −186.3 | 119.0 | −116.7 |
| Actual price $1m and over | 343.9 | −343.9 | **227.8** | −227.5 |
| Floors 21 and up | 168.8 | −153.0 | 119.5 | −110.9 |
| Bukit Timah | 193.9 | −172.4 | **95.5** | −72.6 |
| Bishan | 145.5 | −134.3 | **80.3** | −68.8 |
| Woodlands | **57.1** | −42.3 | 67.9 | −61.8 |
| Yishun | **45.5** | −14.0 | 57.5 | −49.0 |

- **Both models squeeze prices towards the middle:** cheap flats are priced too high, and $800k+ flats far too low. The boom of 2020–2026 was strongest in the dearest flats, which the training years had few of.
- **Bigger, higher and more central flats are missed most**; that is where the hybrid's location corrections help most (Bukit Timah $193.9k → $95.5k, executive flats $146.8k → $97.6k).
- **The hybrid is worse in cheaper non-central towns** (Woodlands, Yishun, Sembawang, Choa Chu Kang) and for $400–600k flats: in 2001–2019 the plane overpriced these towns, so the trees learned to lower them, and in the boom they rose too. Its corrections age, so the trees should be retrained on recent years (the drift check in the deploy recommendation).
- Full tables (flat type, age, price band, storey, year, all 26 towns) are printed by the script.

**One real flat through every model.** The same script follows the latest 4-room sale of Blk 267 Tampines St 21 (September 2026, $618k; 114 m², floors 7–9, 43 years) through every model trained on 2001–2019: plane S $565.7k; random forest $456.1k, gradient boosting $443.9k, LightGBM $435.1k, XGBoost $450.6k (all far too low: a tree cannot price above the 2001–2019 prices in its leaves); hybrid $565.7k − $21.2k = $544.5k. It records the questions each tree asked about the flat, the running total as trees are added, and how many dollars each feature contributed (saved to `hdb_data/explain_models.json`).

#### Deploy recommendation
**Recommended model: the dollar hybrid (plane S + LightGBM correction, 127 leaves)**, chosen by time-series cross-validation including tuning, and scored once on the 2020–2026 benchmark: expect about **$65k (11%) per sale** in a boom like 2020–2026 and about **$39k** in a flatter market like 2015–2019. Use it for valuing a particular flat in the next year or two, where the flat-level correction matters most.
- **The report still runs plane S**, the plane-only part of the hybrid. Running the hybrid in a browser means exporting thousands of LightGBM trees (several MB) into the report, or serving the model from a server. Until then, the plane-only numbers ($75.2k) describe the report.
- **What to watch after deployment:** the bias. Both models were about $50–54k too low in the boom; if new sales show the gap growing, the plane's market terms (income, supply, policy stance) need refitting before the trees can help.
- **Further ahead than 2–3 years**, the correction stays roughly fixed for a given flat (it depends on the flat and location, which do not change, plus age), while the plane carries the market. The uncertainty is set by the plane's market error and grows with the horizon.

**Feature importance** (LightGBM, share of total gain): income 45%, floor area 18%, town 9%, flat type 8%, **policy stance 5%**, flat age 4%, flat model 3%, distance to CBD 3%, storey 2%, flats sold lag 4 2%, open MRT 1%, RFS lag 4 + BTO lag 4 under 1% (random forest gives supply more: 8%). Mall and school distances add little once town and CBD distance are known. The report shows these bars, the age curve each model learned against the plane's age bands, and every model's price today for the typical flat of the block and flat type you pick (`hdb_data/tree_importance.png` and `tree_age_curve.png` hold the same charts for slides).

### Using the forecaster: sliders, replay from a past year, and scenarios
**All drivers slide together.** Every driver series of the chosen model has its own slider (for model S: household income growth, BTO launched, HDB flats sold). Each slider sets its series for every year after its last published figure; published years always use the actual figures.
- **Income** is a **yearly compounded growth rate**: 2% means each year's income is 2% above the year before. The slider shows the actual growth over the last 5, 10 and 20 years as reference points.
- **BTO launched** and **flats sold** are set as flats per year. Both enter the model **4 years later** (lag 4), so a slider changed after year E moves only sale years from E + 5 onwards.

**Replay: train on sales up to any year, then forecast from there** (`hdb_replay_models.py` → `hdb_replay.js`). The report model's plane is refitted on sales up to each year E from 2006 to 2025 (the features and model are those of `hdb_location_model.py`, run unchanged). Choosing "Model trained on sales up to E" turns every year after E into a forecast. Two modes for the drivers after E:
- **"only what was known then"**: drivers stop at the last figure published by E and the sliders set what follows; only policy rounds and MRT stations **announced by the end of E** count. This is the honest replay of a forecaster in year E.
- **"hindsight"**: the drivers that really happened after E are used, so only the coefficients come from the training years.

For each E the report shows how that refit did on the next 5 years over **all blocks**, with the real drivers (hindsight drivers, so this measures the coefficients, not the forecaster's guess of the drivers). All 20 refits, from `hdb_replay_models.py`:

| Trained to | Sales | Scored on | MAE | Bias (model − actual) |
|---|---|---|---|---|
| 2006 | 189,408 | 2007–2011 | $62.7k | −$58.0k |
| 2007 | 216,370 | 2008–2012 | **$38.8k** | −$26.3k |
| 2008 | 243,609 | 2009–2013 | $41.6k | +$20.1k |
| 2009 | 274,045 | 2010–2014 | $41.1k | −$5.6k |
| 2010 | 308,861 | 2011–2015 | $54.5k | +$23.9k |
| 2011 | 331,123 | 2012–2016 | $69.9k | +$48.7k |
| 2012 | 354,309 | 2013–2017 | $67.5k | +$35.1k |
| 2013 | 370,395 | 2014–2018 | $62.4k | +$7.6k |
| 2014 | 386,265 | 2015–2019 | $73.2k | −$62.2k |
| 2015 | 404,031 | 2016–2020 | $76.6k | −$72.8k |
| 2016 | 423,394 | 2017–2021 | $67.7k | −$53.5k |
| 2017 | 443,731 | 2018–2022 | $69.4k | −$40.0k |
| 2018 | 465,283 | 2019–2023 | $64.9k | −$18.8k |
| 2019 | 487,451 | 2020–2024 | $69.6k | −$42.1k |
| 2020 | 510,764 | 2021–2025 | $77.7k | −$62.1k |
| 2021 | 539,821 | 2022–2026 | $69.2k | −$37.5k |
| 2022 | 566,523 | 2023–2026 | $67.4k | −$11.4k |
| 2023 | 592,263 | 2024–2026 | $69.3k | −$15.2k |
| 2024 | 620,080 | 2025–2026 | $69.5k | −$33.3k |
| 2025 | 645,151 | 2026 | $71.4k | −$28.0k |

- **Typical miss about $68k a sale** (median of the 20), from $38.8k to $77.7k. The 2007–2009 refits did best; the 2007 refit is the best of all, so it should not be quoted alone as typical. Refits from 2022 on are scored on fewer than 5 years.
- **The bias changes sign with the market.** Too low after 2006 (it missed the 2007–2012 boom), too high after 2010–2012 (prices fell in 2013–2019 after the cooling rounds, while income kept rising and the stance read the rounds as a hot market), too low again after 2014–2016 (it missed the post-2020 surge). A straight-line plane follows the trend but misses the turning points.
- The 2019 refit is the benchmark model scored on 2020–2024 ($69.6k); on the full 2020–2026 benchmark it is $75.2k, because the 2025–2026 errors are larger.

**Sale year and flat age.** The sale year runs from 2001 to 10 years ahead, so past years can be checked against real sales. The flat's age is set as its age in the latest data year and follows it back and forward in time (a sale year before the block was built shows "not built yet").

**Actual prices.** `hdb_replay.js` holds the median resale price of every block and flat type in every year 1990–2026, from all sales. The price path draws them as dots next to the model, and the tile **"Actual sale price in Y"** shows the block's median for the chosen flat type, with **model − actual** in $ and %. For a past sale year this is a direct check; for a future year there is no actual.

**With and without the events.** Event effects are multiplied on top of the formula, so the report always also shows the **formula alone** ("without the N chosen events") and its gap to the actual price. The event figures are measured on actual prices, which already include part of what income, supply and the policy stance explain, and windows of events close together overlap, so **events can count the same market move twice**. In "only what was known then" mode an event after E also uses information a forecaster at E did not have. Read the formula-alone price as the forecast and the events as a what-if. Example (Blk 469 Segar Rd, 4 room, 2011, trained to 2007): formula alone $325.5k–$341.4k (depending on where the Jan 2011 round is placed) against $394.0k actual; the Sep 2008 crisis event (×1.140) brings it to about $389k.

**A cooling round raises the model price.** The stance coefficient is positive (+$5.29k per point in the 2007 refit): rounds are announced when the market is hot, so the plane reads them as a hot-market signal, not as a brake.

### Data leakage: how it was avoided, and where the project is still exposed
Data leakage means the model sees, during training or testing, something it could not know at the moment of a real prediction, so its score looks better than it will be in use. For a price **forecast**, the rule is: **every input must be known on the day of the sale, and the test years must stay unseen.**

**How it was avoided**

| Rule | How it is applied | Example of the leak it prevents |
|---|---|---|
| **Split by time, not at random** | Forecast models train on sales up to 2019 and are tested on 2020–2026. The random 80/20 split is used only for the separate "valuation" test of the tree models, and labelled as such. | A random split would put a 2023 sale in training and its neighbour in the same block and month in the test. |
| **Every driver from before the sale** | Income: the **previous year's** median (a 2024 sale uses 2023; 2024 is published only in 2025). Supply: BTO launched **4 years before** (known at launch), completions **1 year before** (the latest published). | Using a sale year's own income or completions, published after the year ends. |
| **Same-year supply rejected** | BTO lag 0 and lag 1 were tested and kept off the report: besides reverse causality, a March sale cannot know November's launches. Completions lag 0 is not used for the same reason. | "Predicting today's price with today's data". |
| **MRT as it was on the sale date** | `mrt_open_km` counts a station only from its **opening date**; an announced station adds value only from its **announcement date** (`stage_early` / `stage_late`), using the corrected station table. | A 2010 sale "knowing" about a line that opened in 2020. |
| **Fit on training rows only** | All coefficients, the town and flat-type effects, and the lease-decay bands of the back-tested models are fitted on ≤2019 sales. The tree models' early-stopping hold-out is 10% of the **training** rows; the hybrids' plane is fitted on training rows before the trees learn its errors. No scaler is needed for the linear plane; the trees do not use one. | Test years shaping the coefficients they are scored on. |
| **Location data has no prices in it** | Block coordinates and distances (and the filled new blocks) come from addresses and maps, not from resale prices, so a block's location cannot carry its future price. | — |
| **Policy stance only from rounds already announced** | `tight36` counts only rounds announced in months **before** the sale month, within 36 months. For future years the report counts rounds already announced (fading out after 3 years) plus any rounds you add; in a replay trained to E, "only what was known then" counts only rounds and MRT stations announced by the end of E. | A March sale "knowing" a round announced in June. |
| **Policy and event effects kept out of the scored model** | The measured 1/2/3-year price changes after each round are computed on actual prices (including the years after each event) and are shown for reference only; global events are applied as a user-chosen scenario. One 0/1 column per window made the back-test worse, so those columns are not used; the report model uses only the announced-before policy stance. | Event windows "remembering" what prices did afterwards. |
| **Refit only after testing** | The report uses the model refitted on all sales 2001–2026; the back-test numbers describe the version trained on ≤2019 only. | Reporting an error measured on data the model was trained on. |

**Where the project is still exposed (honest list)**
1. **The 2020–2026 test was used to choose between models**, so it was not sealed: the comparisons above all look at it. To check what an honest choice would have been, `hdb_location_model.py` now also selects **without** 2020–2026, then scores once (see **"Choosing without the test years"** below). The result changes the picture and is reported as found. The hybrid went through the same check before it was recommended (`hdb_hybrid_cv.py`).
2. **Malls and schools: fixed on 31 October 2026.** They used to be measured to present-day locations for every year (a 2005 sale was credited with a mall that opened in 2015). They are now dated like the MRT: a mall counts from its opening month and a popular school from the month it started at its home site (see "Malls and popular schools are now dated"). What remains: the list of *popular* schools is today's list (S2 below).
3. **The age-curve table** in "Lease decay" is measured on all years to describe history; the back-tested model F/H fits its own bands on ≤2019 only.
4. **The valuation test is optimistic by design**: train and test share blocks and months, which is why trees score about $15k there but about $134k on the forecast test.

5. **Rare flat models are grouped using all years.** `hdb_location_model.py` puts flat models with fewer than 2,000 sales into OTHER using counts from every year, before the folds. The effect is likely tiny, but the counts should come from each fold's training years; to fix and report.
6. **An outside audit (October 2026)** confirmed the time-ordered folds, the policy rounds counted only after announcement, the MRT dating, the unsupervised tools fitted inside each fold and the lease-bend arithmetic, and asked for these corrections, now made: (a) 2020–2026 is called the **2020–2026 benchmark**, not a sealed test, because the first driver comparison and the minimum-rounds rule added after model V's result both used it (by the original rule V won validation, $48.5k against S's $52.4k); (b) the policy stance is described as a **predictive market-regime signal, not the causal effect of policy**: the coefficient is positive because cooling rounds come in hot markets, and about 30 rounds are not 666,252 independent cases; (c) the 1–3 intensity, the 36-month window and the mix of private and HDB rounds are assumptions (a 24/36/48-month check is planned); (d) the $0.5k threshold is a judgement, about 1% of the validation error, set before each run. A truly untouched test needs future sales: S, S+Z, V and the hybrid are to be scored once on 2027 sales, with the rule written down beforehand.

### Questions on the drivers: town, policy stance, supply and the 2027 rematch
Questions a reader may ask about the drivers, with the numbers from the runs above.

#### How town is used: one-hot, one effect per town
Town is **one-hot encoded**: each town gets its own 0/1 column, so the model learns a **separate price effect for every town**. It does not give every town the same value.

| Sale in | ANG MO KIO | BEDOK | BUKIT TIMAH | CHOA CHU KANG | … |
|---|---|---|---|---|---|
| Bedok | (dropped) | **1** | 0 | 0 | 0 |
| Bukit Timah | (dropped) | 0 | **1** | 0 | 0 |
| Ang Mo Kio | (dropped) | 0 | 0 | 0 | 0 |

- **Ang Mo Kio is the base town** (`BASE_TOWN`). Its column is dropped, so each town's coefficient reads as "more or less than an otherwise identical flat in Ang Mo Kio". If every town kept a column, they would always add up to 1 and duplicate the intercept (the "dummy variable trap").
- In the full model (2023 – Oct 2026), for example, Bukit Timah is **+$221k** and Choa Chu Kang **−$218k**.
- One-hot assumes **no order and no spacing** between towns. Label encoding (Ang Mo Kio = 1, Bedok = 2 …) would force equal price steps between alphabetical neighbours.
- Flat type (base 4 ROOM) and flat model (base Model A) are encoded the same way.

| Model | How town goes in |
|---|---|
| Linear plane | one-hot, Ang Mo Kio dropped: one coefficient per town |
| LightGBM / XGBoost | as a **category** (`pd.Categorical`): the trees split towns into groups directly |
| Random forest | one-hot, all towns kept (trees do not suffer the dummy trap) |

#### Policy rounds and global events are not known in advance: how they are used
The model never uses anything it could not have known at the sale date.

| | Policy stance (`tight36`) | Global events |
|---|---|---|
| In the regression? | **Yes**, one column | **No**: tried as 49 one-year 0/1 columns, they made the back-test worse (overfitting) |
| In training | counts only rounds announced in the **36 months before the sale month** | measured separately, in the event study (prices 1, 2 and 3 years after each type of event) |
| In a forecast | rounds already announced keep counting until 36 months old (**already known**); new rounds are **not predicted**: they are a what-if added by the user | a **scenario** the user chooses; the forecast is shown **with and without** it |
| Honest check | the replay **"only what was known then"** retrains on sales up to a past year and uses only rounds, drivers and MRT stations announced by then: over 20 replays the typical miss was about **$68k a sale** | |

The stance cut the validation error from $94k (H) to $60k (R). Its coefficient is positive: cooling rounds come in hot markets, so it works as a **sign of a hot market**, a predictive market-regime signal, not the causal effect of policy.

#### A round announced today: which sales it affects
A round announced in **October 2026** counts from the **month after** the announcement, for 36 months:

| Sale month | Counts in `tight36`? |
|---|---|
| October 2026 | no |
| **November 2026 to October 2029** | **yes** |
| November 2029 onwards | no |

So it enters the forecasts for **2027 and 2028 in full, and 2029 up to October**. Because the coefficient is positive (about **+$5.0k to +$5.5k per stance point**), a cooling round of intensity 2 adds +2 to the stance and raises the model's price by about $10–11k in those years. That is the hot-market reading, not what the policy does to prices; that effect is measured separately in the event study.

#### Could a lagged stance make policy known further ahead? Candidate Z
Like BTO launches lag 4 and flats sold lag 4, a stance from rounds announced **4 to 7 years before** the sale (candidate **Z**, formerly L) would make policy known up to 7 years ahead. A round announced today would count twice:

| Window | Sales it affects | Coefficient |
|---|---|---|
| `tight36` (in S) | Nov 2026 – Oct 2029 | **+$5.0k to +$5.5k per point**: a hot-market signal |
| Z (rounds 4–7 years before) | about late 2030 – late 2033 | **−$4.5k to −$4.8k per point**: prices lower years later, like a delayed brake |

| Model | Validation 2015–2019 | 2020–2026 benchmark |
|---|---|---|
| **S** | **$52.4k** | $75.2k |
| S + Z | $55.5k | **$68.3k** |

S + Z was **not adopted**: it lost on validation, the only score used to choose. Choosing it for its 2020–2026 score would be choosing by years already looked at, the model V problem in reverse. Unlike BTO lag 4, which matches a physical event (flats completing about 4 years after launch), Z's lag has no such mechanism, so it needs a stronger test.

#### Why the minimum-rounds rule does not apply to Z
`MIN_ROUNDS = 3` in `hdb_location_model.py` applies to **a count column for one type of policy**: it can be chosen only if at least 3 rounds of that type were announced before 2020. It was written for models U and V. Z is not split by type: like `tight36`, it adds up **all** rounds (direction × intensity), dozens since 2001.

| Candidate | What it is | Why not adopted |
|---|---|---|
| V | one count per policy type | fails **minimum rounds** (BSD / tax: 1 round); the rule was added **after** seeing its 2020–2026 result, and is stated as such |
| U | counts of cooling and of easing rounds | lost to S on **validation** |
| Z (S + Z) | the combined stance, rounds 4–7 years before | lost to S on **validation** ($55.5k vs $52.4k) |

#### Why 2027 matters, and the rule for it
Under the current rule (choose by 2015–2019 validation), S + Z has already lost, and a 2027 result cannot change that **unless the rule for 2027 is written down beforehand**. The two pieces of evidence disagree, and neither is clean:

| Evidence | Says | Why it is not decisive |
|---|---|---|
| Validation 2015–2019 | S is better | a **quiet market** with few new rounds: a delayed-brake effect has little chance to show |
| 2020–2026 benchmark | S + Z is better | the COVID boom with heavy cooling, but these years were **already looked at** many times |

2027 is the first clean evidence: sales no choice has seen. It is a **rematch on new data**: S is the current choice and stays unless clearly beaten.

**The rule, written before any 2027 sale is seen:**

> S is the model. S + Z did not beat S on the 2015–2019 validation; it did better on the 2020–2026 benchmark, but those years had already been used, so they are not counted. S and S + Z are frozen now. On 2027 sales, scored once, S + Z replaces S only if its average error is lower than S's by at least **1% of S's 2027 error** and it wins in at least **95% of sale-by-sale bootstrap resamples**. Otherwise S stays.

- **Why not keep the $0.5k threshold:** it was set for validation, about 1% of S's error there ($52.4k). In the boom the errors are larger, so a fixed $0.5k would be a lower bar. A 1% relative threshold keeps the same spirit.
- **Why the bootstrap:** one year is noisier than five. Both models price the same 2027 sales, so they are compared sale by sale and resampled; the gain must not be chance. This also answers the audit's remark that no significance test was run.
- **Why not "either 2015–2019 by $0.5k or 2020–2027 by 1%":** "either" gives the challenger two chances to win, which lowers the bar, and 2020–2026 cannot count because it has been used. One deciding test, set before it is seen, scored once.
- The same rematch scores V and the hybrid once on 2027, under the same written rules.
- Alternative: once 2027 is in, extend the validation window to 2015–2026 and keep 2028 as the next untouched test.

#### Why keep RFS + BTO when flats sold is an official series
The RFS batches (1995–2000) counted every flat offered that year, **including unsold flats offered again**; BTO counts **new flats launched to order**, with no roll-over.

| Point | Detail |
|---|---|
| Where the roll-over matters | only the **2001–2004 sales** (RFS lag 4). From 2005 the column is BTO launches |
| How it is handled | the `rfs4` 0/1 column (+$63.2k for 2001–2004) absorbs the change of measure; it is **0 in every future year**, so a forecast never uses an RFS figure |
| Lag | S uses **lag 4** only; lag 5 was tested and came out slightly worse ($77.3k vs $76.4k) |

**The direct test, model T (official data only):** flats sold lag 4 + the policy stance + location and lease decay, no newspaper series and no `rfs4`. Rule set before the run: S must beat T by $0.5k on validation, otherwise T would have been used.

| | Validation 2015–2019 | 2020–2026 benchmark |
|---|---|---|
| **S** | **$52.4k** | **$75.2k** |
| T | $53.4k | $78.0k |

S passes by $1.0k (T wins 2018 and 2019, S wins 2015–2017), so it stays; T is the fallback. The two series measure different things: **launches** are supply decided earlier and completing about 4 years later, known in advance (2023–2026 launches give 2027–2030 supply); **flats sold** is take-up, which reflects demand as well as supply.

#### Why supply has no intensity and no 36-month window
| | Supply (RFS / BTO, flats sold) | Policy rounds |
|---|---|---|
| Size | already a **count of flats** | an event with no natural size, so a **1–3 intensity** gives it one |
| Timing | arrives at **one point**: flats are handed over about 4 years after launch, so a single **lag 4** | keeps working for **2–3 years** after the announcement (event study), so a **36-month window** |
| Data | **yearly totals** | exact **announcement months** |

Not yet tested: a multi-year supply window (e.g. launches 3–5 years before the sale, for projects completing early or late). It would be judged on validation like every candidate.

#### Hand-scored intensity vs counting by machine: Y, U and V
The 1–3 intensities in `policy_events.csv` are a judgement. Three mechanical alternatives were tested; none beat S on validation:

| Candidate | How rounds are counted | Validation 2015–2019 | 2020–2026 benchmark | Result |
|---|---|---|---|---|
| **S** | direction × **intensity 1–3** (judgement) | **$52.4k** | $75.2k | the model |
| **Y** (formerly C) | direction × **number of instruments**, capped at 3 | $54.3k | **$74.2k** | lost validation by $1.9k |
| **U** | counts of cooling and of easing rounds | $55.4k | $77.5k | lost validation |
| **V** | one count per policy type | **$48.5k** | $86.1k | won validation, failed the benchmark: overfitting |

- **Y**, the closest, learned almost the same coefficient (+$5.45k vs +$4.99k per point). Counting instruments treats every measure as equal, while the judgement can tell a big ABSD rise from a small tweak.
- The result: **broadly robust to replacing the judgement**, not "the judgement does not matter"; no significance test was run on the $1.9k gap.

#### The difference between U and V
Both drop the 1–3 intensity so that the regression weighs the rounds itself; they differ in **how finely they split the rounds**:

| | **U** | **V** |
|---|---|---|
| Split by | **direction**: cooling or easing | **policy type** |
| Columns | 2: `cool36`, `ease36` | 5: `pol_ssd`, `pol_absd`, `pol_loan`, `pol_bsd`, `pol_ease` |
| Each column counts | rounds in the 36 months before the sale | the same, per type |
| Learned ($k per round, trained 2001–2019) | cooling **+12.90**, easing **−6.35** | SSD **+28.00**, ABSD **+4.68**, loan **+6.76**, BSD / tax **+20.19**, easing **−7.30** |
| Validation 2015–2019 | $55.4k | **$48.5k** |
| 2020–2026 benchmark | $77.5k | $86.1k |
| Outcome | stable, but $2.9k worse than S on validation | **overfitting**: few rounds per type before 2020 (BSD / tax 1, ABSD 3, SSD 4) |

S uses one column instead: `tight36`, the sum of direction × intensity, with one slope. Every cooling coefficient in S, U and V is positive, because cooling comes when prices are already rising: they describe what kind of market comes with each kind of round, not what the round does to prices.

### Malls and popular schools at the sale date, the school-list check, and population
Three tests run on 31 October 2026, all with the same rules as the rest of the project: last year's values for every sale,
2015–2019 validation to choose, 2020–2026 scored once.

#### Malls and popular schools are now dated, like the MRT
Before this change, every sale since 2000 was measured to **today's** malls and popular schools: a 2005 sale near Westwood
Primary (opened 2013) or VivoCity (opened 2006) was credited with something that did not exist yet. Now:

| Column | Counted from |
|---|---|
| `dist_to_mall_km` | the mall's **opening month** (`hdb_data/malls_dated.csv`; closed months excluded, e.g. Century Square Sep 2017 – Jun 2018) |
| `dist_to_nearest_high_demand_primary_school_km` and the 1 km flag | the month the school started at its **home site** (`hdb_data/school_home_sites.csv`); holding sites used during rebuilding are ignored, because MOE measures the 1 km rule from the school's registered site |

- **Sources:** school history pages and the MOE Heritage Centre first, then the newspaper archive (scanned pages 2010–2018 and tablet and phone pages 2021–2026, all OCR'd), NLB Infopedia and news; Wikipedia only where nothing else was found. Each date has its source and a confidence.
- **What changed:** for **19.4%** of sales the nearest mall was farther away at the time of sale (37% of 2001 sales, almost none after 2023); the 1 km school flag switches off for **1.5%** (about 3% a year before 2013).
- **Schools whose home site starts after 2000** include Rosyth (Nov 2000), Canberra, Compassvale, Temasek, Pei Chun, Wellington, Bukit Timah, CHIJ Our Lady of the Nativity (2000–2001), Hong Wen and ACS (Junior) (2009), Frontier (2012), Westwood, Riverside, Sengkang Green, Punggol Green (2013), Alexandra (2014) and Angsana (June 2022, new site).

Test (`hdb_dated_location_test.py`): the recomputed undated columns match the lookup exactly (correlation 1.000, flag identical for 100% of sales), so every difference comes from the dating alone.

| Model | Validation 2015–2019 | Test 2020–2026 MAE | Bias | Mall $k/km | School $k/km | 1 km flag $k |
|---|---|---|---|---|---|---|
| S, undated (before) | 52.4 | 75.2 | −50.1 | −9.33 | +8.78 | +6.91 |
| **S1, dated (now the page model)** | 52.4 | 75.3 | −50.6 | −9.82 | +9.06 | +7.66 |
| S2, dated, only the 50 schools at their home site before 2000 | 52.4 | 75.2 | −50.5 | −9.77 | +7.17 | +2.72 |

**Adopted although it does not win on accuracy.** By the usual rule (beat by $0.5k on validation) the undated S would stay. Dating is not a new feature but a correction of **look-ahead**: the model no longer credits a flat with a mall or school that did not exist yet. Accuracy barely moves because the test years are not affected (by 2020 almost every mall and school on the lists existed) and town already carries much of the location effect. Rerun of `hdb_location_model.py` with the dated columns: S validation $52.4k, sealed test **$75.3k** (RMSE 107.4, bias −50.6).

#### Does today's list of popular schools matter? (S2)
The popular schools are the high-demand schools of recent P1 registration exercises (Phase 2B and 2C lists, 75 with a position). MOE publishes no popularity ranking by year, so the list is today's. S2 keeps only the 50 schools that were at their home site before 2000: the long-established ones.

- **Accuracy: the same** ($75.2k vs $75.3k).
- **The 1 km premium: not the same.** It falls from **+$7.7k** to **+$2.7k** when only the established schools count. Much of the measured premium comes from the 25 newer popular schools, mostly in newer towns (Punggol, Sengkang, Jurong West).
- So the premium is reported as **about +$3k to +$8k, depending on which schools count as popular**, not as one firm number. A year-by-year list (from the P1 ballot reports in the newspaper archive) would settle it; it is left as further work.

#### Which popular schools carry the premium: 2B vs 2C, bands, and the boundary rule
`hdb_school_test.py`, every school dated at its home site. Share of sales each 1 km flag covers:

| Flag | All years | 2020 on |
|---|---|---|
| Any of the 75, within 1 km (school point) | 68% | 73% |
| A **2B** school within 1 km | 40% | 40% |
| A **2C-only** school within 1 km (no 2B within 1 km) | 28% | 33% |
| Any of the 75, within 1 km, **land boundary from October 2021** | 69% | 77% |

| Version | Validation 2015–2019 | Test 2020–2026 | 1 km premium |
|---|---|---|---|
| **S: all 75, distance + 1 km flag (page model)** | 52.4 | 75.3 | **+$7.7k** |
| 2B only | 52.4 | 75.2 | +$7.7k |
| 2B and 2C-only as separate flags | 52.4 | 75.3 | 2B **+$8.2k**, 2C-only **+$7.3k** |
| All 75 with bands (1 km, 1–2 km) | 52.4 | 75.3 | 1 km −$1.4k, 1–2 km −$7.6k: not usable |
| Boundary rule from October 2021 | 52.4 | 75.5 | +$6.8k |

- **No version beats S** (the rule: by $0.5k on validation), so S stays.
- **The premium is not concentrated in the 2B list:** 2B schools carry only about $1k more than 2C schools.
- **Bands do not work in this form:** 96% of sales are within 2 km, so "beyond 2 km" is a small, unusual group and the bands split the same information as the distance three ways.
- **The distance slope is positive** (+$7–9k per km farther) in every version: read the school effect as the **step at the 1 km line**, not as "farther is better"; the slope picks up something else (newer, larger flats in estates farther from the older popular schools).
- **The boundary rule:** from October 2021 (the month after MOE announced the land-boundary measure on 9 September 2021, for the 2022 P1 exercise), the flag is measured from the school's land parcel (URA Master Plan) to the block's outline (HDB building outlines). It moves 5.6% of later sales into the 1 km flag and none out. The school test could not judge it (its models train only to 2019, before the rule existed), so `hdb_boundary_test.py` trains on years under the rule: to 2022 to choose (validation 2023, 34,183 training sales under the rule), then to 2023 for a check on 2024–2026 (59,923).

  | | Validation 2023 | Check 2024–2026 | Bias | Sales whose flag differs (4,068): MAE | 1 km premium (trained to 2023) |
  |---|---|---|---|---|---|
  | **S: school point (page model)** | **54.51** | **68.96** | −16.2 | **66.4** | +$6.4k |
  | SR: land boundary from Oct 2021 | 54.51 | 68.99 | −16.3 | 67.1 | +$5.5k |

  **The point rule stays.** The boundary rule ties on validation, and even on the 4,068 sales where the two rules disagree it misses slightly more ($67.1k vs $66.4k). Prices do not (yet) follow MOE's exact measure: the premium sits with flats that are near a popular school by plain distance, which are within 1 km by both rules. The 2024–2026 years were already part of the 2020–2026 benchmark, so that part is a check, not an untouched test.

In one line: **near a popular school (within 1 km) adds about +$7–8k, about the same for 2B and 2C schools; narrowing the list, adding a 1–2 km band or the boundary rule did not improve the forecast.**

#### Population: tested and rejected as a driver
`hdb_population_test.py` built a population table from five SingStat tables (data.gov.sg: M810001, M810381, M810011, M830101, M830201; 1995–2026, 32 columns) and back-tested 13 columns, each added on its own to **Income** and to **Income + BTO lag 4** (the driver table above).

| Base model | Reference MAE | Best population column | Verdict |
|---|---|---|---|
| Income | $77.5k | resident population growth $77.6k | no gain |
| Income + BTO lag 4 | $74.7k | change in residents aged 25–34 $74.4k | −$0.3k, but the coefficient has the wrong sign (−$14.1k per 10,000 more young adults): noise |

- **Growth** columns (total, residents, citizens, PRs, non-residents) change almost nothing: income already carries the slow demand trend.
- **The swings are non-residents**, who mostly cannot buy resale flats: in 2021 the total fell 4.1% (non-residents −10.7%, citizens −0.7%); applied to 2022 sales, total-population growth raised that year's error from $66.4k to $91.9k.
- **COVID broke the marriage series:** 22,651 marriages in 2020 gave 2021 sales an error of $189.9k.
- **Levels** (total, PRs, marriages) rise with income; the model confuses the two and forecasts $59k–$132k too low.
- Population may matter more for **private** property (PRs and foreigners can buy, expats rent); that is for the private-property model.

### Limitations
- The 1 km school flag uses MOE's centre-point rule for every year; from the 2022 Primary 1 exercise (announced September 2021) MOE measures from the school's land boundary. The boundary rule was tested and tied (validation 2023: 54.51 both), so the point rule stays; blocks near the 1 km edge may still be classed differently from MOE.
- Malls and schools are dated to their opening (since 31 October 2026), but the list of popular schools is today's: the 1 km premium is about +$3k to +$8k depending on which schools count as popular.
- Renovation, unit position and block upgrading are not in the data (see above).
- The age curve beyond 60 years is extrapolated; the 20–25-year hump reflects upgrading and flat design, not age itself.
- Market timing remains the largest error; one driver at a time; apart from the age curve every slope is a straight line; town effects are fixed.
- The QE group has only two examples (four without QE); the 2020 case also includes BTO construction delays.
- Policy rounds close together (e.g. Jan, Jun, Aug 2013) share the credit.
- Global events in the report multiply the formula and can double count (see "With and without the events"); a replay that matches one block's actual price closely is one example, not a measure of accuracy.
- Measured effects are associations from past data, not proof of cause. Use the benchmark MAE (about ±$75k, about 12% of the price, for the location model S in the report; about ±$65k for the recommended hybrid) as the uncertainty around any single flat's future price; errors grow the further ahead the forecast goes.

## Project history: what changed and why
Each step says what was asked and what was found.

| # | Step | What happened |
|---|---|---|
| 1 | When does new supply reach prices? | BTO launches at lag 0 and lag 1 were tested against lag 4. Lags 0/1 look strong only because HDB launches more flats when prices are already rising (reverse causality), and a March sale cannot know November's launches. Lag 4 kept. |
| 2 | Small fixes | Explorer page doctype and charset; `build_block_lookup.py` committed. |
| 3 | Why did the model start in 2005? | BTO data starts in 2001, so a 4-year lag only works from 2005 and 2001–2004 sales were dropped (answered in step 6). |
| 4 | Is the geocoding right? | How the lookup was geocoded; `check_block_lookup.py` and `fix_filled_blocks.py`; re-check reports 0 flags. |
| 5 | Lesson 3.4 in the report | Random forest, gradient boosting, XGBoost, LightGBM and a hybrid added, with a "your block today" row. |
| 6 | Bringing back 2001–2004 | RFS lag 4 + BTO lag 4 as one supply column, plus the `rfs4` 0/1 column for the 2001–2004 sales. Model H. |
| 7 | Are the 1997–2000 figures "dark data"? | Official flats sold and constructed (SingStat, from 1977) downloaded and tested; at a fair lag the newspaper series still did better. |
| 8 | Data leakage written up | Every rule used to keep future information out, and where the project is still exposed. |
| 9 | Choosing without the test years | Time-series cross-validation on 2015–2019, then a 2020–2026 benchmark scored once. |
| 10 | Model S in the report | Both supply measures + flats sold + the policy stance won the validation; the report switched to S. |
| 11 | Validating and tuning the hybrid | Plane S + LightGBM won the same cross-validation; 127 leaves chosen; deploy recommendation. |
| 12 | Looking inside the trees | `hdb_explain_models.py`: one real sale through every model; where the models miss. |
| 13 | The block's real flat; what the stance means | Picking a block sets the sliders to its typical flat; the stance described as a hot-market signal (positive coefficient), not a brake. |
| 14 | RFS years as yearly totals | RFS flats were offered every quarter: 19 quarterly batches found in the newspaper archive, a year = 4 × the average quarter. All scripts rerun: S CV $52.4k, benchmark $75.2k; hybrid CV $38.6k, benchmark $65.0k; same models chosen. |
| 15 | Is the newspaper series needed? | Model T (official flats sold + stance) added with a rule set before the run: S must beat T by $0.5k on validation. S won ($52.4k vs $53.4k); T benchmark $78.0k vs S $75.2k. |
| 16 | Forecasting from the past, and checking against real prices | See below. Built on 24 October and committed together, because the report changes were not committed as they were made. |
| 17 | Policy rounds counted by month, as in training | See below. |
| 18 | Income growth when there is less than 10 years of history | Trained to 2006–2009, the income slider offered only "0%" and "last 5 yrs" (income data starts in 2000) and opened on the 5-year rate, which in 2008 is the boom rate 5.74%. It now also offers **"since 2000 (all known)"** (4.17% a year in 2008) and opens on it; the slider moves in steps of 0.01%. |
| 19 | Policy rounds weighted by the regression, not by hand | Models U (cooling / easing counts) and V (counts by policy type), no 1–3 intensity. V won validation ($48.5k vs S $52.5k) but failed the 2020–2026 benchmark ($86.1k vs $75.2k): too few rounds per type. S kept; a minimum of 3 rounds per type is now required, added after this run and said so. See "Letting the regression weigh the policy rounds itself". |
| 21 | The July 2013 rule for older flats | CPF and HDB-loan limits for flats with under 60 years of lease (from 1 Jul 2013) tested as lease-decay effects W and X, not as a policy round. Validation $52.7k / $52.8k vs S $52.4k; S kept. W learned −$23k for affected flats: real, but already in S's curve. |
| 22 | What buyers actually paid for old flats | Median price per m² by flat age, 2001–2026: 40–49-year-old flats sold at 111% of a new flat's price per m² in 2010, 94% in 2013 and 75% in 2026. The turn matches the July 2013 limits; the model's curve separates location from age. |
| 23 | Two tests of the policy stance; lesson 3.5 tools | `hdb_policy_stance_cv.py`: the 1–3 judgement replaced by an instrument count (Y) and the stance 4–7 years back (Z, S+Z); S kept. `hdb_unsupervised_cv.py`: k-means, Isolation Forest and PCA inside the folds; S kept. |
| 24 | The course's own data | `hdb_course_compare.py`: on the course's 2017–2019 data, split and models, this project's features cut LightGBM's error from $24.3k to $17.6k (dated malls and schools; $17.5k undated) and win on every seed and on the next-year test; every feature group, added one at a time, lowers the error. |
| 25 | Outside audit | 'Sealed test' renamed the 2020–2026 benchmark; causal wording removed; the S/V choice, rare flat models and the remaining assumptions listed under "Where the project is still exposed". |
| 26 | How the distances are measured | MRT, CBD, malls and primary schools: haversine vs point-in-polygon, the fixed CBD point in Raffles Place, MRT dated to the sale month, and MOE's 2022 switch to the school land boundary for the 1 km rule. |
| 27 | Questions on the drivers | Town one-hot vs categories; how policy rounds and global events are used without knowing the future; which sales a round affects; the lagged stance Z and why it was not adopted; the minimum-rounds rule; the 2027 rematch rule (1% threshold and sale-by-sale bootstrap, written in advance); RFS + BTO vs flats sold (model T); why supply has no intensity or 36-month window; hand-scored intensity vs machine counts (Y, U, V). |
| 28 | Checking the school boundary and dating malls and schools | How the 75 popular (Phase 2B) schools were chosen and what is not recorded; yearly popular lists from the P1 ballot history 2009-2026; opening dates for malls and schools; building MOE's land-boundary 1 km flag from URA's Master Plan land use and HDB's building outlines, checked against OneMap SchoolQuery, time-gated at September 2021; school bands (1 km, 1-2 km) and announced moves of popular schools, as with MRT. Planned, not yet run. |
| 29 | Malls and schools dated; school-list check; population | Mall and popular-school columns now count a mall from its opening month and a school from the month it started at its home site (S1, adopted as a look-ahead correction: test $75.3k vs $75.2k). The 1 km school premium is +$7.7k with all 75 schools and +$2.7k with the 50 established before 2000, so it is reported as a range. Population (13 columns from SingStat) tested and rejected. See "Malls and popular schools at the sale date". |
| 30 | Which popular schools carry the premium | 2B only, 2B and 2C as separate flags, 1-2 km bands and the land-boundary rule from October 2021: none beats S; the 1 km premium is about +$7-8k for 2B and 2C alike. See "Which popular schools carry the premium". |
| 31 | Rerun with dated malls and schools (1 November 2026) | Every script and the Power BI tables rerun with the dated columns: plane S validation $52.4k (unchanged), 2020–2026 benchmark $75.3k (was $75.2k), default hybrid $64.7k (was $64.8k); the Model Selection page keeps the 24 October record. The course comparison rerun with dated columns, and its one-group-at-a-time table added. |

### Step 16 in detail
**16a. All drivers of the model slide at the same time.** *Asked:* move income, BTO launched and flats sold together instead of one driver at a time. *Done:* one slider per driver series of the chosen model. Each sets its series for every year after its last published figure. Income is a **yearly compounded** growth rate (the label says so, after the question "is this yearly or cumulative?"), with the real growth over the last 5, 10 and 20 years shown as reference points. BTO and flats sold are flats per year.

**16b. Train on sales up to any year and forecast from there.** *Asked:* "my future price always starts from today": train on sales up to, say, 2004 and predict 2009 and 2014. *Done:* `hdb_replay_models.py` runs `hdb_location_model.py` unchanged and refits the report model on sales up to each year E from 2006 to 2025 (2001–2005 is too short to fit). Each refit is scored on the 5 years after E over all blocks with the real drivers, and the report shows that score. Two modes: **"only what was known then"** (drivers, policy rounds and MRT stations announced by the end of E; the sliders set what follows) and **"hindsight"** (the drivers that really happened, so only the coefficients come from the training years). Over the 20 refits the typical miss was about $68k a sale (range $38.8k–$77.7k), and the bias changed sign with the market (table in "Using the report").

**16c. Sale year from 2001 and the flat's age.** The sale year now runs from 2001 to 10 years ahead. The age slider is the flat's age in the latest data year and follows it back in time; a sale year before the block was built shows "not built yet".

**16d. Actual prices next to the model.** *Asked:* a tile with the actual price in the chosen year, to see the difference. *Done:* `hdb_replay.js` holds the median resale price of every block and flat type in every year 1990–2026 (from all sales). The price path draws them as dots; the tile "Actual sale price in Y" shows the median, the number of sales and **model − actual** in $ and %.

**16e. Many policy rounds and global events, each in its own year.** *Asked:* "leaving out global events is a bit frivolous": only one round and one event could be chosen, and none could be chosen in "known then" mode. *Done:* add as many rows as needed, each with its own year; "+ the rounds that really happened" fills in the real ones up to the sale year; "clear" removes them. Events tagged low relevance in `global_events.csv` are marked "[low relevance, not averaged]". A note under the rows says how each is applied, and warns when the policy stance and an event within 3 years stand for the same market.

**16f. The price with and without the events.** *Asked:* the gap to the actual price both with and without the events, "based on pure truth calculation, not on your desire to close the gap". *Done:* display only; the model and its numbers are unchanged. The formula's price and the events' multiplier are shown separately, each with its gap to the actual price.

**What was found while testing (and not tuned away)**
- **The BTO and flats-sold sliders seemed not to move the price.** Both enter with a 4-year lag, so with a model trained to 2008 they only change sale years from 2013 on. Working as designed; the README now says so.
- **Setting income growth to the real 2% a year widened the gap** for Blk 469 Segar Rd in 2011. That is reported as found: the formula alone is 13–17% below the actual price there, in line with the all-blocks replay bias of the 2007 refit (−$26.3k).
- **The interest rate is not in model S**, so changes in SORA do not move its price (the income + interest rate driver failed its back-test).
- **Events can count the same move twice.** Event figures are measured on actual prices, which already include part of what income, supply and the stance explain, and the windows of nearby events overlap (Sep 2008's 3-year window contains May 2010's 1-year window). In "known then" mode an event after E also uses outcomes a forecaster at E could not know. The formula alone is the forecast; events are a what-if. A close match with events on one block is one example, not accuracy.
- **Policy timing: year in the report, month in training** (fixed in step 17). Training counts a round from the month after its announcement; the report counted a round placed in year T only from T+1, so a January round missed almost a whole year.
- **A cooling round raises the model price** (+$5.29k per stance point in the 2007 refit), because rounds come when the market is hot.

**Bugs fixed while building:** the replay data was read before the report had set up the latest data year (page failed to load); the "trained up to" selector and the model could disagree after switching drivers.

**Files:** changed `hdb-future-price.html`, `README.md`; new `hdb_replay_models.py` → `hdb_replay.js`.

### Step 17: policy rounds counted by month, as in training
*Found:* placing the Jan 2011 round in 2011 made the report count it only from 2012, while `hdb_location_model.py` counts a round for every sale from the month after its announcement, for 36 months. Moving it to "2010" by hand gave the training value for 2011 sales, but moving rounds one by one is tuning.
*Done:* the report now uses the training rule month by month. A sale in month m counts every round announced before the 1st of m and within the 36 months before it. The report's price for year S stands for a whole year of sales, so the stance for S is the **average over its 12 months**. A real round you add keeps its own month (moved to the year you pick); a "like <group>" choice has no month and counts as announced on 1 July. The note under each round says in how many months of the sale year it counts, e.g. "Jan 2011 … In 2011: 11 of 12 months, +2.75 on the stance". The rounds already in the data (before the training year) use the same monthly rule, so prices in every year move slightly. "+ the rounds that really happened" now includes rounds in the sale year itself, because they count from the month after.
*Example (Blk 469 Segar Rd, 4 room, trained to 2007, known then, all real rounds added):* stance 2010 +6.0, 2011 +8.8, 2012 +11.5, 2014 +3.3, 2015 0.

### Step 19: policy rounds weighted by the regression
*Asked:* the 1–3 intensity in `policy_events.csv` is a judgment; the regression should decide each weight, in the spirit of causality, not arbitrarily.
*Done:* `hdb_location_model.py` builds the round counts without intensity (`cool36`, `ease36`, and one count per policy type) and adds models U and V to the same cross-validation, with a rule set before the run (beat S by $0.5k). V won validation and failed the 2020–2026 benchmark; S stays, with the minimum-rounds rule added after the run (details and all coefficients in "Letting the regression weigh the policy rounds itself"). The report can run either kind of model: if a count model is ever chosen, its formula and policy notes show the counts instead of the stance.

### Step 18: income growth from all the history known then
*Found:* with a model trained to 2008, the income slider opened on 5.74% a year, the growth of 2003–2008, the top of a boom. The 10-year rate cannot be shown before 2010 because the income series starts in 2000. Income really fell in 2009 ($6,102 → $6,012) and reached $6,350 in 2010, about 2.0% a year from 2008, so the 5-year rate over-priced 2011 by about $25k.
*Done:* when there is no 10-year rate, the slider offers and opens on the yearly growth over **all the years known then** (since 2000): 4.17% for a model trained to 2008. It is a rule chosen before looking at the answer, the same for every training year, not a rate picked to match one block. The real 2008–2010 growth (2.0%) is hindsight; to see the model's own error without a guessed income path, use "what really happened (hindsight)".

## Data (prices and drivers from data.gov.sg)
- HDB resale flat prices — 5 datasets, Jan 1990 – Oct 2026
- Median monthly household employment income incl. employer CPF (SingStat) — 2000–2025
- URA private residential property price index, All Residential — quarterly, 1975–2026
- Per capita GDP in chained 2015 dollars (SingStat) — 1960–2025
- Singapore Overnight Rate Average, SORA (MAS) — monthly, 2005–2026
- BTO units launched per year — added up from a project list (town, project, units, launch date), 2001–2025 (`bto-2001-2024_with_RFS_WIS_residual_2000_2004_reconstructed.xlsx`)
- RFS flats available for selection, quarter by quarter 1995–2001, from HDB's notices in a newspaper archive (`rfs_quarterly_1995_2001.csv`; kept out of the repo); a year's figure is 4 × the average of the quarters found
- Private residential pipeline supply at Q4, EC excluded (URA via data.gov.sg), 1995–2026 (`Singapore_Private_Residential_Supply_Definitive_Merged_1995_2026_EC_History.xlsx`)
- MRT stations: own timeline + OneMap geocodes, corrected with Wikipedia opening dates and coordinates (`mrt/`)
- Block locations: own HDB master table geocoded with OneMap, with URA boundaries by point-in-polygon and distances to MRT, malls and Phase 2B primary schools (`hdb_block_lookup_from_V3F.csv`); new blocks filled with OneMap; checked block by block (`check_block_lookup.py`, 0 flags)
- Policy rounds and global events: `policy_events.csv` and `global_events.csv` (dates, descriptions and sources per row; edit them to add or correct rows)

## Files
| File | What it does |
|---|---|
| `hdb_resale_1990_2026.py` | Downloads the 5 resale datasets, combines them, fits the plane for each year |
| `hdb_pairs.py` | Fits the 5 input-pair planes → `hdb_pairs.js` |
| `hdb_drivers_model.py` | Downloads the 4 macro tables, reads the 2 supply workbooks (in `hdb_data/`) and the quarterly RFS batches (`rfs_quarterly_1995_2001.csv`; `RFS_MODE=quarter` for the old single-quarter values), fits and back-tests each driver model, measures policy/event effects on actual prices → `hdb_drivers.js` and `hdb_data/macro_annual.csv` |
| `policy_events.csv` | 30 policy rounds: announce and effective dates, segment, direction, intensity, instruments, description, source, confidence |
| `global_events.csv` | 19 global and local shocks: start and end dates, category, region, relevance to Singapore property |
| `fill_missing_blocks.py` | Splits the unmatched blocks into demolished and new; fills demolished ones from a surviving block on the same street and geocodes new ones with OneMap → `hdb_data/hdb_block_lookup_filled.csv` and `hdb_data/filled_blocks_report.csv` |
| `check_block_lookup.py` | Checks the CBD, mall and school columns of the block lookup (5 checks above; `--onemap` adds the OneMap cross-check) → `hdb_data/block_lookup_check.csv` and `hdb_data/mall_school_locations.csv`. Read-only |
| `fix_filled_blocks.py` | Recalculates the CBD, mall and school columns of the filled blocks from their own coordinates, using the positions found by `check_block_lookup.py` (keeps a `.bak` copy) |
| `hdb_tree_models.py` | Lesson 3.4: random forest, gradient boosting, XGBoost, LightGBM and two hybrids on the same sales and features as model H; forecast and valuation tests, feature importance, learned age curves, every model's price today per block and flat type → `hdb_trees.js`, `hdb_data/tree_backtest.csv` and two PNG charts |
| `hdb_hybrid_cv.py` | Validates the hybrid (plane S + LightGBM) with the 2015–2019 time-series cross-validation, then scores the chosen model once on 2020–2026; `--grid` also tunes LightGBM's settings the same way → `hdb_data/hybrid_cv.csv`, `hdb_data/hybrid_sealed.csv` |
| `hdb_explain_models.py` | Follows one real sale (Blk 267 Tampines St 21, 4 room) through every model trained on 2001–2019: the plane's terms, the questions each tree asked, running totals and per-feature contributions; and the benchmark error of plane S and the hybrid by flat type, age, price band, storey, year and town → `hdb_data/explain_models.json` |
| `hdb_replay_models.py` | Runs `hdb_location_model.py` unchanged, refits the report model on sales up to each year 2006–2025, scores each refit on the next 5 years, and takes the median price of every block and flat type in every year 1990–2026 → `hdb_replay.js` (read by `hdb-future-price.html` if present) |
| `hdb_location_model.py` | Joins blocks to sales, rebuilds MRT features for each sale month, fits the lease-decay curve, back-tests models A–F, prints the age curve and bias by age, writes the best model with every block and station → `hdb_location.js` |
| `hdb_lease_rule_cv.py` | Tests the 1 July 2013 CPF / HDB-loan limits for flats with under 60 years of lease: runs `hdb_location_model.py` unchanged, adds candidates W (a 0/1 step) and X (extra lease-decay bends from July 2013), cross-validates them against S on 2015–2019 and scores the 2020–2026 benchmark once. Prints only; nothing is written. S kept |
| `hdb_age_gap_check.py` | Median resale price per m² (and 4-room median) of each flat-age band, year by year 2001–2026; old flats as % of new, the dollar gap and growth since 2007 → `hdb_data/age_gap_by_year.csv` and `.png`. Prints only |
| `hdb_policy_stance_cv.py` | Runs `hdb_location_model.py` unchanged, then tests the policy stance: instrument-count intensity (Y) and the stance from rounds 4–7 years before (Z, S+Z), same cross-validation and rule; prints only |
| `hdb_unsupervised_cv.py` | Lesson 3.5 tools on model S, each fitted inside every fold: k-means location segments (KM8/16/32), Isolation Forest (ISO01, ISO1), PCA (PCA95); prints only |
| `hdb_course_compare.py` | The course's 2017–2019 data, split and models, with the course's features against this project's flat and location features: one split, five seeds, next-year tests, features added one group at a time; prints only |
| `mrt/` | Corrected station table, the script that builds it, and its two source tables (Wikipedia opening dates and coordinates) |

## Rebuild
```
conda activate dsai-m3
python hdb_resale_1990_2026.py   # about 5-10 minutes (data.gov.sg rate limit)
python hdb_pairs.py
python hdb_drivers_model.py
python hdb_location_model.py     # needs hdb_data/hdb_block_lookup_from_V3F.csv and hdb_data/mrt_stations_corrected.csv
python hdb_tree_models.py        # about 20 minutes; needs xgboost and lightgbm (pip install -U xgboost lightgbm)
python hdb_hybrid_cv.py          # about 5 minutes; add --grid to tune LightGBM (longer)
python hdb_explain_models.py     # about 5 minutes; one flat through every model, error by segment
python hdb_replay_models.py      # about 6-8 minutes; replay models 2006-2025 and actual block prices -> hdb_replay.js
python hdb_lease_rule_cv.py       # about 12-15 minutes; tests the July 2013 older-flat rule against model S (prints only)
python hdb_age_gap_check.py       # under a minute; what buyers actually paid for old vs new flats, year by year
python hdb_policy_stance_cv.py   # about 12 minutes; policy stance tests Y, Z, S+Z
python hdb_unsupervised_cv.py    # lesson 3.5 tools inside the folds: KM8/16/32, ISO01/ISO1, PCA95
python hdb_course_compare.py <course hdb_resale_2017_2019.csv> <Blocks.csv> <MRT_Stations.csv>   # course data comparison
```
To refill the block lookup after new resale data: delete `hdb_data/hdb_block_lookup_filled.csv`, run `hdb_location_model.py` (it lists the unmatched blocks), then `fill_missing_blocks.py`, `check_block_lookup.py`, `fix_filled_blocks.py`, `check_block_lookup.py` again (it should report 0 flags), and finally `hdb_location_model.py`.
Downloaded data stays in `hdb_data/`, which is kept out of the repo by `.gitignore`.
