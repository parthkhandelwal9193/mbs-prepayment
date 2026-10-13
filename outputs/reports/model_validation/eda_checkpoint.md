# MBS Prepayment EDA Checkpoint

Status: EDA and scenario-sensitivity work only. No XGBoost-based
cash-flow valuation, duration, or convexity has been calculated.

## Empirical valuation without XGBoost

| period | effective_duration_years | effective_convexity_market_convention |
| --- | --- | --- |
| train_2018_2021 | 4.9677 | -6.0538 |
| test_2023_2024 | 6.0727 | -3.8914 |
| holdout_2025_2026 | 4.9787 | -4.9830 |

These are illustrative empirical-curve valuation results, not
XGBoost-based risk measures.

## XGBoost performance

- Test isotonic: rows=1,389,985, events=6,723, ROC-AUC=0.5801, average precision=0.00906, Brier=0.004810, log loss=0.030471.
- Holdout isotonic: rows=1,101,653, events=8,588, ROC-AUC=0.6588, average precision=0.02117, Brier=0.007691, log loss=0.043867.

## Actual versus predicted payoff UPB

- Test: realized full-payoff rate=0.442406%, isotonic predicted rate=0.647545%, predicted/actual payoff dollars=1.464x.
- Holdout: realized full-payoff rate=0.849747%, isotonic predicted rate=0.918625%, predicted/actual payoff dollars=1.081x.
- These are full-voluntary-payoff UPB diagnostics, not total CPR.

## XGBoost rate-shock sensitivity

| cohort_month | shock_bp | loan_rows | cohort_upb | raw_upb_weighted_probability | isotonic_upb_weighted_probability |
| --- | --- | --- | --- | --- | --- |
| 2024-12-01 | -100 | 68057 | 18,318,532,604.38 | 2.590807% | 1.444278% |
| 2024-12-01 | 0 | 68057 | 18,318,532,604.38 | 1.405766% | 0.749694% |
| 2024-12-01 | 100 | 68057 | 18,318,532,604.38 | 0.903698% | 0.461964% |

This is one-month full-payoff probability sensitivity under a constant
historical PMMS shift. It is not a future rate-path forecast, CPR, cash-flow
projection, price, duration, or convexity.

## Current limitation

- Model feature count: 36; encoded columns: 44.
- Base December raw and isotonic predictions were reproduced exactly.
- Scenario feature support has not been reliably verified; do not use the
  ±100 bp model results for valuation.
- XGBoost duration and convexity: NOT COMPUTED.
- The target is a full voluntary payoff and excludes partial curtailments.
  Do not relabel its probability as total CPR.

## Decision

The empirical valuation and XGBoost performance/forecast diagnostics are
ready to preserve as separate EDA results. The current XGBoost rate-shock
outputs are preliminary sensitivity only. A model-driven cash-flow
valuation requires a verified scenario-feature audit and an explicit
full-payoff-to-cash-flow assumption. Do not retrain on the existing test
period and continue to describe it as out-of-sample.
