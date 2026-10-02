# Retail Sales Forecasting and Promotion Effectiveness

Case study on the dunnhumby **Breakfast at the Frat** dataset. The project forecasts weekly unit sales for each
product-store series over a 13-week holdout, measures how price and promotions relate to demand, and simulates
pricing and promotion scenarios.

All numbers below come from one full run of `python -m src.run_pipeline` (about 9 minutes on a single CPU core, seed 42).
Re-running reproduces them; they are also saved in `outputs/`.

## 1. Business questions

1. How does product demand change over time?
2. Do price reductions increase sales?
3. Do sale tags (temporary price reductions), circular features and in-store displays influence demand?
4. Can weekly product-store sales be forecast from historical information?
5. How do the answers differ across products, categories, store locations and store price tiers?

## 2. Data

Source: `dunnhumby - Breakfast at the Frat.xlsx` and its user guide. The workbook is not redistributed here.

| Sheet | Rows | Key |
|---|---|---|
| dh Transaction Data | 524,950 | WEEK_END_DATE, STORE_NUM, UPC |
| dh Products Lookup | 58 (55 appear in transactions) | UPC |
| dh Store Lookup | 79 rows, 77 unique stores | STORE_ID (STORE_NUM in transactions) |

- 156 consecutive weeks, 2009-01-14 to 2012-01-04; 77 stores in TX, OH, KY and IN; tiers Value (19), Mainstream (43), Upscale (15).
- Four categories: bag snacks, cold cereal, frozen pizza, oral hygiene products.

**Differences from the user guide** (checked on the supplied file):

- The guide describes "five products"; the transactions contain **55 UPCs** (58 in the lookup).
- The real column headers are on row 2 of each sheet (row 1 is a title).
- The store lookup has 79 rows but 77 unique stores: stores 4503 and 17627 each appear twice with conflicting price-tier
  labels (Mainstream and Upscale). The first label is kept.
- The lookup has extra columns (store name, city, MSA code, average weekly baskets).

## 3. Project structure

```
frat_forecasting/
├── README.md
├── requirements.txt
├── config.yaml                 # paths, horizon, validation, model, interval, ablation, scenario settings
├── data/
│   ├── raw/                    # place the xlsx here (not committed)
│   └── processed/              # parquet cache created on first run
├── src/
│   ├── utils.py                # config, logging, WMAPE / RMSE / bias
│   ├── data_loading.py         # read xlsx, normalise columns, cache, join tables
│   ├── quality_checks.py       # data-quality flags, price cleaning, summary table
│   ├── panel.py                # contiguous weekly store-UPC panel, missing-week handling
│   ├── features.py             # feature groups, calendar, as-of history features, supervised rows
│   ├── models.py               # baselines, LightGBM, cold-start rule
│   ├── validation.py           # rolling-origin cross-validation and segment metrics
│   ├── intervals.py            # quantile regression + conformal calibration
│   ├── ablation.py             # feature-group ablation
│   ├── explain.py              # TreeSHAP driver attribution
│   ├── elasticity.py           # fixed-effects elasticity and promotion lift
│   ├── scenarios.py            # price and promotion simulations
│   ├── plots.py                # figures
│   └── run_pipeline.py         # runs all steps end to end
├── notebooks/
│   ├── 01_eda_data_quality.ipynb
│   └── build_notebook.py       # regenerates the notebook
├── tests/test_pipeline.py      # panel, leakage-safety and metric tests
└── outputs/                    # results (see section 10)
```

## 4. Setup and run

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp "/path/to/dunnhumby - Breakfast at the Frat.xlsx" data/raw/
python -m src.run_pipeline --config config.yaml
```

Options: `--fast` (fewer trees, quick smoke test), `--skip-ablation`, `--reload` (re-read the xlsx instead of the parquet cache).
The first run reads the workbook (about a minute) and caches parquet files in `data/processed/`.

```bash
python -m pytest -q tests        # 9 tests: panel, leakage safety, metrics
jupyter notebook notebooks/01_eda_data_quality.ipynb
```

## 5. Data-quality findings and handling

| Issue | Count | Handling |
|---|---|---|
| Duplicate week-store-UPC rows | 0 | None needed |
| UPCs or stores missing from lookups | 0 | None needed |
| Stores with conflicting tier labels | 2 | First label kept |
| Missing price / base price | 23 / 185 | Refilled from neighbouring weeks of the same series |
| Price = 0 | 1 | Treated as invalid, refilled |
| Units = 0 with visits > 0 | 5 | Excluded from training and scoring |
| Visits > units (impossible) | 2,309 | Flagged only |
| Units per visit above 6 | 48 | Flagged outlier, kept |
| Visits per household above 3 | 293 | Flagged outlier, kept |
| Price above base price | 6,047 (1,002 by more than 10%) | Kept; discount depth floored at 0 |
| Promo flag set but no price cut | 14,384 | Expected for feature or display |
| Price cut over 5% with no promo flag | 0 | None needed |
| Parking spaces missing | 51 of 77 stores | Column not used |
| Panel completeness | 524,950 of 660,660 possible rows (79%) | See section 6 |

Outliers are well under 1% of rows and are kept; headline metrics are essentially identical with and without them
(holdout WMAPE 0.2900 either way). 18 rows are excluded in total (5 zero-unit rows plus 13 whose price could not be refilled).

## 6. Modelling design and assumptions

**Panel.** 3,909 store-UPC series; only 1,225 have all 156 weeks, 751 start after week 1, 582 end before the last week and
2,431 have gaps inside their active span. Each series is made contiguous between its first and last observed week; the
50,223 missing weeks inside that span are inserted as zero units and flagged. Inserted rows may train the model but are
never scored. Weeks outside the span (not launched or discontinued) are not created. A gap could be a stockout, a delisting
or genuine zero demand; the data cannot always tell these apart, so this is an assumption.

**Direct multi-horizon forecasting.** Each training row is a (series, origin, horizon h) triple that predicts week origin + h,
for h = 1..13. Features come from two clearly separated sources:

- *Origin features* (prefix `o_`): rolling means and spread of units, non-promoted baseline, recent promo rate and discount,
  last price, history length. They use data up to and including the origin week only.
- *Known-in-advance features* for the target week: price, base price, discount, price relative to the category, promotion
  state, calendar and holiday flags, product and store attributes, and a year-ago lag (always at least 39 weeks old).

**Validation and holdout.** The holdout is the last 13 weeks, **2011-10-12 to 2012-01-04** (origin = week index 142). Three
earlier rolling-origin folds step back 13 weeks each (origins 129, 116, 103). A fold trains only on rows whose target week is at
or before its origin. No random splitting is used. `tests/test_pipeline.py` checks these properties.

**Future price and promotion are assumed known.** Prices and promotion flags for the holdout weeks are taken from the data,
as in normal retail planning where promotions are scheduled before they run. Forecast error from promotion plans changing is
therefore not captured. This is the main assumption and a limitation.

**Promotion encoding.** The three flags are mutually exclusive except feature + display, so promotions are encoded as five
states: none, TPR only, display only, feature only, feature + display.

**Cold start.** Of 42,630 observed holdout rows, 42,616 belong to series with history at the origin. The other 14 (0.03%) are
series launched after the origin; they use a rule (the same UPC's units per weekly basket across other stores, scaled by the
new store's baskets).

## 7. Methods

| Component | Approach |
|---|---|
| Baselines | Seasonal naive (same week last year, 13-week mean when missing) and 13-week moving average |
| Main model | LightGBM, Tweedie objective, one global model across all series |
| Intervals | LightGBM quantile models (10th and 90th percentile) with conformalised quantile regression (CQR) calibration on the previous CV fold; 80% nominal |
| Ablation | Feature groups added one at a time and removed one at a time, on origins 129 and 142 |
| Drivers | TreeSHAP via LightGBM `pred_contrib`, grouped by feature block |
| Elasticity | Log-log regression with store-UPC and week fixed effects, cluster-robust errors; spec A price only, spec B price plus promotion flags |
| Scenarios | Elasticity-based for everyday price changes; model-based for promotions (see section 9) |

## 8. Results

**Forecast accuracy** (WMAPE, lower is better; RMSE in units):

| Model | CV mean (3 folds) | Holdout WMAPE | Holdout RMSE | Holdout bias |
|---|---|---|---|---|
| LightGBM | 0.310 | **0.290** | 11.6 | +3.4% |
| 13-week moving average | 0.469 | 0.444 | 21.8 | -3.7% |
| Seasonal naive | 0.586 | 0.591 | 27.2 | +2.2% |

The LightGBM model beats the best baseline by about 35% on WMAPE in the holdout and in every CV fold.

**Holdout WMAPE by segment** (LightGBM):

| Segment | WMAPE | | Segment | WMAPE |
|---|---|---|---|---|
| Bag snacks | 0.250 | | Upscale stores | 0.266 |
| Cold cereal | 0.282 | | Mainstream stores | 0.289 |
| Frozen pizza | 0.343 | | Value stores | 0.330 |
| Oral hygiene | 0.434 | | No promotion | 0.285 |
| 13-51 weeks of history (386 rows) | 0.450 | | Any promotion | 0.25 to 0.35 |

Low-volume categories have higher percentage error because the same absolute miss is a larger share of small numbers.
Accuracy is reported against actuals only for observed weeks.

**Prediction intervals** (80% nominal, holdout): raw quantile intervals cover **81.1%** (mean width 18.2 units); after
conformal calibration coverage is **85.5%** (width 19.2). The calibrated intervals are somewhat conservative: coverage is
about 81% for cold cereal but 91% for oral hygiene, where the additive margin is large relative to low volumes.
Coverage falls with horizon: calibrated coverage is 89% at one week ahead and about 80% at weeks 12 and 13, where the raw quantile intervals under-cover (about 76%).

**What drives the forecasts** (share of mean |SHAP|): recent history 45.5%, promotion state 21.8%, price 18.5%, product
6.6%, calendar 4.4%, store 3.3%. The top single features are the 52-week average, promotion type and discount depth.

**Ablation** (mean WMAPE across origins 129 and 142, 150 trees):

| Added in order | WMAPE | | Removed from the full model | WMAPE change |
|---|---|---|---|---|
| History only | 0.432 | | Promotion | +0.034 |
| + price | 0.374 | | Price | +0.014 |
| + promotion | 0.326 | | Product | +0.007 |
| + product | 0.324 | | Calendar | +0.007 |
| + store | 0.321 | | Store | +0.001 |
| + calendar | 0.314 | | | |

Price and promotion information account for most of the gain over history alone.

**Price elasticity** (log-log, fixed effects; spec A total price response / spec B holding promo flags fixed):

| Category | Spec A | Spec B |
|---|---|---|
| Bag snacks | -1.56 | -1.36 |
| Cold cereal | -2.79 | -1.78 |
| Frozen pizza | -3.76 | -2.97 |
| Oral hygiene | -1.54 | -1.28 |
| All | -2.36 | -1.70 |

Demand is elastic (below -1) in every category. At product level (spec A) the median is -1.06, 31 of 55 products are below -1, and
3 have an implausible positive sign, which reflects weak within-product price variation. Treat single-product estimates as noisy.

**Promotion lift** (median units versus the series' own non-promoted weeks): TPR only 1.19x, display only 1.55x, feature only
2.13x, feature + display 3.58x overall (raw lifts; these include the discounts that usually accompany promotions).
Units per visit rise only about 8 to 14%, so most of the lift comes from more shopping trips, not bigger baskets.

## 9. Scenarios (13 holdout weeks, relative to the baseline forecast)

| Scenario | Method | Units | Revenue |
|---|---|---|---|
| Everyday price -10% | elasticity | +21.0% | +10.7% |
| Everyday price -5% | elasticity | +9.7% | +5.0% |
| Everyday price +5% | elasticity | -8.4% | -4.5% |
| Everyday price +10% | elasticity | -15.7% | -8.5% |
| Sale tag 5%, 10%, 15%, 20% off on un-promoted rows | LightGBM | +6.7%, +5.2%, +13.3%, +16.1% | +5.3%, +0.5%, +5.1%, +3.8% |
| Add display (no price cut) | LightGBM | +21.2% | +23.1% |
| Add circular feature (no price cut) | LightGBM | +31.7% | +34.4% |
| Feature + display (no price cut) | LightGBM | +79.1% | +84.1% |

Why two methods: in this data a price cut never occurs without a promotion flag, so the tree model has no basis for an
"everyday price" counterfactual and responds unreliably (in testing it predicted falling sales after a cut). Everyday price
changes therefore use the category elasticity (spec B). Promotion scenarios use combinations that do occur in the data.

**Treat the model-based sale-tag results with caution.** They are not monotone in depth (5% off gives a larger lift than 10%
off) and are well below what the spec A elasticity implies at deeper discounts, for example 20% off lifts cold cereal 19% in the
model versus 86% from the elasticity (`outputs/scenarios/scenario_elasticity_check.csv`). Display, feature and elasticity-based
price results are more stable. Revenue is shelf price times units; there is no cost data, so profit is not estimated.

## 10. Deliverables

| Deliverable | Location |
|---|---|
| Reproducible Python project | `src/`, `config.yaml`, `tests/` |
| README | this file |
| Data-quality and EDA notebook | `notebooks/01_eda_data_quality.ipynb` |
| Data-quality summary | `outputs/data_quality_summary.csv` |
| Product-store-week modelling dataset | `outputs/modelling_dataset.parquet` |
| Model comparison and validation results | `outputs/cv_results.csv`, `outputs/segment_metrics.csv` |
| Forecast file (store, UPC, week, forecast, 80% bounds, actual) | `outputs/forecasts.csv` |
| Interval coverage | `outputs/interval_coverage.csv` |
| Ablation | `outputs/ablation_results.csv` |
| Driver attribution | `outputs/driver_attribution/` |
| Elasticity and promotion lift | `outputs/elasticity_estimates.csv`, `outputs/promo_lift.csv` |
| Scenarios | `outputs/scenarios/` |
| Figures | `outputs/figures/` |
| Run summary | `outputs/run_summary.json` |
| Final report and presentation | Not yet written |

## 11. Limitations

- Promotion and price for the forecast window are taken as given; real forecasts would have to forecast or plan them.
- 77 stores in four states (41 of them in Texas in the transaction data) and four categories: results may not generalise.
- Elasticities and lifts are observational. Promotions are not randomly assigned and are often placed in peak weeks, so
  estimates may be confounded and should not be read as causal.
- Model-based sale-tag scenarios are noisy (see section 9); everyday price changes rely on a constant-elasticity assumption.
- Missing-week handling is an assumption (section 6); competitor activity and weather are not in the data.
- The 14 cold-start rows are too few to judge the fallback rule (it over-forecast them).
- Intervals are slightly conservative overall and uneven across categories.
- The 80% intervals are per series; they cannot be summed to give an interval for a total.

## 12. Data source

dunnhumby Source Files, "Breakfast at the Frat". Copyright dunnhumby; used for this assessment only.
Questions about the data: sourcefiles@dunnhumby.com.
