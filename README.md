# Agency MBS Prepayment and Rate-Risk Analysis

This project uses Freddie Mac loan-level performance data to study mortgage prepayments and their effect on agency mortgage-backed securities (MBS). It builds a monthly loan-performance panel, estimates prepayment behavior, and translates observed incentive–prepayment relationships into cash-flow and rate-shock scenarios.

The analysis focuses on a practical question: how does the relationship between refinance incentive and prepayment change across market regimes, and what does that imply for principal timing and interest-rate risk?

> **Valuation results are illustrative scenarios—not market-calibrated prices, an OAS valuation, or investment advice.** They depend on the hypothetical pool, fixed coupon and discount-yield assumptions, empirical prepayment curves, and simplified parallel rate shocks described below and in the valuation scripts.

## Results

### Empirical prepayment curves

![Empirical prepayment S-curves by period](outputs/figures_final/fig1_s_curves.png)

### Price sensitivity to rate shocks

![Illustrative price versus rate shock](outputs/figures_final/fig2_price_vs_rate_shock.png)

### Weighted-average life sensitivity

![Illustrative WAL versus rate shock](outputs/figures_final/fig3_wal_vs_rate_shock.png)

### Duration and convexity

![Illustrative duration and convexity](outputs/figures_final/fig4_duration_convexity.png)

## Project workflow

1. **Data preparation** — Load Freddie Mac loan-level performance data and mortgage-rate data.
2. **Panel construction** — Build a monthly loan-level panel and identify voluntary payoff events.
3. **Feature engineering** — Construct refinance incentive, loan seasoning, burnout, and calendar features.
4. **Hazard modeling** — Train logistic-regression and XGBoost models to estimate loan-month payoff risk.
5. **Temporal evaluation** — Evaluate on chronological splits rather than random loan-month splits.
6. **Prepayment analysis** — Estimate balance-weighted CPR by refinance-incentive bucket and compare curves across periods.
7. **Cash-flow projection** — Convert CPR assumptions to monthly SMM and project scheduled principal, prepayments, and interest.
8. **Rate-shock scenarios** — Reprice the illustrative pool under parallel rate shocks and calculate WAL, effective duration, and convexity.

## Data

### Freddie Mac loan-level performance data

The project uses the Freddie Mac Single-Family Loan-Level Dataset. Freddie Mac provides the data through Clarity Data Intelligence; registration and sign-in are required to access the download page. Review Freddie Mac's current terms and conditions before using or redistributing the data.

- [Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)

### Mortgage-rate data

The project also uses Freddie Mac's Primary Mortgage Market Survey (PMMS) mortgage-rate data.

- [Freddie Mac PMMS archive](https://www.freddiemac.com/pmms/pmms_archives)

Download the historical series needed by the preparation scripts. The exact file, columns, and date coverage should match the inputs expected by `src/prepare_pmms.py`.

### Local data files

Raw source data are not included in this repository. Place downloaded files in the local paths expected by the scripts, under `data/raw/` where applicable. Check the input paths in each script before running; filenames and directory layouts from data providers can change.

Do not commit downloaded raw data unless you have verified that redistribution is allowed and that the files contain no restricted or sensitive information.

## Data and target definitions

The processed panel contains loan-month observations, loan characteristics, market-rate features, and a binary event label.

The current event label is based on zero-balance code `01`, which Freddie Mac defines as a prepaid or matured voluntary payoff. The model experiments include both the existing event definition and a separately constructed next-month payoff target. These are different prediction targets, so their metrics should not be compared as if they measured the same task.

Rate incentive is defined as the borrower's mortgage rate relative to the prevailing market mortgage rate. Positive incentive indicates a potential rate saving from refinancing; negative incentive indicates that refinancing at the market rate is not financially attractive.

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

## Illustrative valuation assumptions

> **Important interpretation:** The valuation outputs are scenario-analysis results for a hypothetical pass-through pool. They are not observed security prices, a market quote, a market-calibrated valuation, an OAS analysis, or an investment recommendation.

The empirical curves are used to assign CPR assumptions under parallel rate shocks. CPR is converted to SMM, and the assumed prepayment speed is applied to a hypothetical pass-through pool to project monthly principal and interest cash flows. The analysis then calculates price per 100, weighted-average life, effective duration, and effective convexity.

The results depend on the assumptions and conventions encoded in `valuation_baseline.py`, `valuation_empirical.py`, and the figure-generation scripts. In particular, the cash-flow engine uses a hypothetical pool, fixed coupon and discount-yield assumptions, empirical bucket-level CPR relationships, and simplified parallel rate shifts. Review those scripts before interpreting or reproducing the figures; do not treat their outputs as market-calibrated values.

For the 2023–2024 empirical curve, the `±100 bp` scenario produced:

| Scenario | CPR | Price per 100 | WAL |
|---|---:|---:|---:|
| Rates −100 bp | 17.23% | 104.13 | 4.79 years |
| Base case | 6.40% | 100.00 | 10.00 years |
| Rates +100 bp | 3.97% | 91.98 | 12.42 years |

At this shock size, the illustrative pool's effective duration is 6.07 years and its effective convexity is −3.89 under the convention used in the code. These figures describe this model setup only; they should not be generalized to a traded MBS or security.

## Main findings

- Refinance incentive is associated with higher observed payoff speed, but the relationship varies across periods.
- The 2023–2024 test period contains a much larger share of loans with negative refinance incentive than the training period.
- Loan-level ranking is weaker in this out-of-time period, particularly among deeply out-of-the-money loans.
- The empirical 2023–2024 S-curve implies lower CPR under rate rallies than the 2018–2021 curve, resulting in slower principal return and higher illustrative duration.
- The XGBoost model is useful for studying loan-level risk, while the empirical S-curve is used for the current illustrative pool-level valuation scenarios because model probability calibration was not stable across periods.

## Repository structure

```text
.
├── data/
│   ├── raw/                 # Downloaded source data; not committed
│   └── processed/           # Processed panels and development samples
├── src/                     # Data preparation, features, and model scripts
├── outputs/
│   ├── figures_final/        # Final charts and chart data
│   ├── models/               # Saved models and metric tables, if generated
│   ├── predictions/          # Loan-month predictions, if generated
│   └── valuation_baseline/  # Empirical curves, cash flows, and scenario tables
├── make_final_figures.py
├── valuation_baseline.py
└── valuation_empirical.py
```

Some directories shown above are created when running the workflow and may not be present in a fresh clone.

## Reproducing the analysis

Run commands from the repository root. The dataset is not bundled with this repository, and exact input filenames and paths depend on the downloaded files and the settings in the scripts. Inspect each script's input/output path configuration before running.

1. Clone the repository and change into its directory:

   ```bash
   git clone https://github.com/parthkhandelwal9193/mbs-prepayment.git
   cd mbs-prepayment
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the project's dependencies. If a `requirements.txt` is present:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   If no dependency file is present, inspect the imports in the scripts and install the required packages in the environment. Do not assume that a fresh environment is reproducible until the dependencies have been recorded.

4. Download the Freddie Mac loan-level data and PMMS rates from the official links above. Place the files in the paths expected by the preparation scripts; raw data are not included in this repository.

5. Run the preparation, feature, and modeling scripts in the order required by their inputs. The exact sequence and command-line arguments are script-specific; inspect each script's `main` block and path settings. The core scripts include:

   ```text
   src/prepare_pmms.py
   src/build_panel.py
   src/add_rate_features.py
   src/build_hazard_sample.py
   src/validate_features.py
   src/train_xgboost_hazard.py
   ```

6. Run the valuation scripts and figure-generation script only after their expected input files have been created:

   ```bash
   python valuation_baseline.py
   python valuation_empirical.py
   python make_final_figures.py
   ```

   These commands assume the scripts use repository-relative paths and that all required inputs exist. Check each script before running; adjust commands if it requires arguments or different paths.

7. Compare regenerated outputs with the committed figures and tables. Model results can vary if source data, package versions, script settings, or random seeds differ.

## Limitations

- The public loan-level data do not contain all household, property-market, or servicing information that can drive turnover-related payoffs.
- Zero-balance code `01` includes prepaid or matured voluntary payoffs; it is not a pure refinance label.
- The observed incentive-bucket curves combine loans with different characteristics and are not fully adjusted for borrower mix or loan seasoning.
- The cash-flow engine is illustrative and is not calibrated to a specific traded security, market curve, spread, or OAS.
- A production valuation model would require a more complete pool/security specification, market-consistent rate scenarios, validated prepayment forecasts, and benchmarking against observed security prices or market analytics.