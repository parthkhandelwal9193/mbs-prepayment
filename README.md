# Agency MBS Prepayment and Rate-Risk Analysis

This project uses Freddie Mac loan-level performance data to study mortgage prepayments and their effect on agency mortgage-backed securities (MBS). It builds a monthly loan-performance panel, estimates prepayment behavior, and translates observed incentive–prepayment relationships into cash-flow and rate-shock scenarios.

The analysis focuses on a practical question: how does the relationship between refinance incentive and prepayment change across market regimes, and what does that imply for principal timing and interest-rate risk?

## Project workflow

1. **Data preparation** — Load Freddie Mac loan-level performance data and mortgage-rate data.
2. **Panel construction** — Build a monthly loan-level panel and identify voluntary payoff events.
3. **Feature engineering** — Construct refinance incentive, loan seasoning, burnout, and calendar features.
4. **Hazard modeling** — Train logistic-regression and XGBoost models to estimate loan-month payoff risk.
5. **Temporal evaluation** — Evaluate on chronological splits rather than random loan-month splits.
6. **Prepayment analysis** — Estimate balance-weighted CPR by refinance-incentive bucket and compare curves across periods.
7. **Cash-flow projection** — Convert CPR assumptions to monthly SMM and project scheduled principal, prepayments, and interest.
8. **Rate-shock scenarios** — Reprice the illustrative pool under parallel rate shocks and calculate WAL, effective duration, and convexity.

## Data and target

The project uses Freddie Mac Single-Family Loan-Level Dataset performance records. The processed panel contains loan-month observations, loan characteristics, market-rate features, and a binary event label.

The current event label is based on zero-balance code `01`, which Freddie Mac defines as a prepaid or matured voluntary payoff. The model experiments include both the existing event definition and a separately constructed next-month payoff target. These are different prediction targets, so their metrics should not be compared as if they measured the same task.

Rate incentive is defined as the borrower’s mortgage rate relative to the prevailing market mortgage rate. Positive incentive indicates a potential rate saving from refinancing; negative incentive indicates that refinancing at the market rate is not financially attractive.

## Modeling and evaluation

The loan-level models use chronological partitions:

| Partition | Period |
|---|---|
| Train | 2018–2021 |
| Validation | 2022 |
| Test | 2023–2024 |
| Holdout | 2025–2026 |

The validation period is used for model selection and probability calibration. The later test and holdout periods are used to examine out-of-time performance. Because the data are a loan-month panel, preserving chronology is important: random splits could place observations from different market periods into both training and evaluation sets.

Model performance is reported using ranking and probability metrics, including ROC-AUC, average precision, Brier score, and log loss. The 2023–2024 test period showed materially weaker loan-level ranking than the training period. This is consistent with a change in the distribution of refinance incentives: most test-period loan-months were deeply out of the money, leaving refinance incentive with less power to distinguish which individual borrowers would prepay. Public loan-level features also do not capture many household-level reasons for moving or paying off a mortgage.

Probability calibration reduced some aggregate overprediction but did not eliminate it in the 2023–2024 period. The saved current-event model probabilities and the separately constructed next-month payoff target have different timing definitions; comparisons between them are diagnostic, not a like-for-like forecast validation.

## Empirical prepayment curves

For each market period, the project calculates balance-weighted monthly payoff rates across refinance-incentive buckets and annualizes them as CPR. The resulting empirical S-curves show that payoff speeds generally rise with refinance incentive, but the strength of the response differs by period.

The 2023–2024 curve is flatter than the 2018–2021 curve over much of the positive-incentive range. This pattern is consistent with a lower observed refinance response during the rate-lock-in period. It does not, by itself, establish that borrower preferences changed: period differences may also reflect burnout, borrower composition, seasoning, market conditions, and data limitations. Thin incentive buckets are excluded from plotted curves to reduce the influence of noisy estimates.

## Illustrative cash-flow and rate-risk scenarios

The empirical curves are used to assign CPR assumptions under parallel rate shocks. CPR is converted to SMM, and the assumed prepayment speed is applied to a hypothetical pass-through pool to project monthly principal and interest cash flows. The analysis then calculates price per 100, weighted-average life, effective duration, and effective convexity.

For the 2023–2024 empirical curve, the `±100 bp` scenario produced:

| Scenario | CPR | Price per 100 | WAL |
|---|---:|---:|---:|
| Rates −100 bp | 17.23% | 104.13 | 4.79 years |
| Base case | 6.40% | 100.00 | 10.00 years |
| Rates +100 bp | 3.97% | 91.98 | 12.42 years |

At this shock size, the illustrative pool’s effective duration is 6.07 years and its effective convexity is −3.89 under the convention used in the code.

These are scenario-analysis results, not market-calibrated valuations. The cash-flow engine uses a hypothetical pool, fixed coupon and discount-yield assumptions, a simplified parallel rate shift, and empirical bucket-level CPR relationships. It is not an option-adjusted-spread model and should not be interpreted as a quoted market price or investment recommendation.

## Main findings

- Refinance incentive is associated with higher observed payoff speed, but the relationship varies across periods.
- The 2023–2024 test period contains a much larger share of loans with negative refinance incentive than the training period.
- Loan-level ranking is weaker in this out-of-time period, particularly among deeply out-of-the-money loans.
- The empirical 2023–2024 S-curve implies lower CPR under rate rallies than the 2018–2021 curve, resulting in slower principal return and higher illustrative duration.
- The XGBoost model is useful for studying loan-level risk, while the empirical S-curve is used for the current illustrative pool-level valuation scenarios because model probability calibration was not stable across periods.

## Repository structure

```text
data/
  raw/                 # Source files; not committed
  processed/           # Processed panels and development samples
src/                   # Data preparation, features, and model scripts
notebooks/             # Exploration and analysis
outputs/
  models/              # Saved models and metric tables
  predictions/         # Loan-month predictions
  valuation_baseline/  # Empirical curves, cash flows, and scenario tables
  figures_final/       # Final charts and chart data
```

## Reproducibility

Create and activate a Python environment, install the dependencies in `requirements.txt`, and run the project scripts from the repository root. Raw Freddie Mac files and other source data are not included in the repository; place them in the expected `data/raw/` location before rebuilding processed outputs.

The scripts use separate filenames for experimental models and outputs so that new experiments do not overwrite the original benchmark artifacts.

## Limitations and next steps

- The public loan-level data do not contain all household, property-market, or servicing information that can drive turnover-related payoffs.
- Zero-balance code `01` includes prepaid or matured voluntary payoffs; it is not a pure refinance label.
- The observed incentive-bucket curves combine loans with different characteristics and are not fully adjusted for borrower mix or loan seasoning.
- The cash-flow engine is illustrative and is not calibrated to a specific traded security, market curve, spread, or OAS.
- A production valuation model would require better pool/security specification, market-consistent rate scenarios, validated prepayment forecasts, and benchmarking against observed security prices or market analytics.
