# Final Results Inventory

## Supported project results

### Empirical prepayment-curve valuation
- Inputs: observed balance-weighted incentive-bucket prepayment relationships.
- Outputs: illustrative cash-flow, price, weighted-average-life, effective
  duration, and effective convexity scenarios.
- Status: primary pool-level valuation result.

### XGBoost full-voluntary-payoff model
- Inputs: loan-level lagged refinance-incentive, loan, seasoning, burnout,
  calendar, and categorical features.
- Outputs: chronological test/holdout discrimination, calibration, and
  payoff-UPB forecast diagnostics.
- Status: supported loan-level predictive model.

### XGBoost rate-shock sensitivity
- December 2024 base predictions reproduced exactly for 68,057 loans.
- A −100 bp market-rate shock increased the full-cohort UPB-weighted isotonic
  payoff probability from 0.7497% to 1.1079%.
- A +100 bp market-rate shock reduced it to 0.5721%.
- Training-support coverage ranged from 70.8% of UPB at −100 bp to 50.1% at
  +100 bp.
- Status: bounded sensitivity diagnostic, not primary valuation input.

## Experimental outputs not used as project results

`value_xgb_common_support.py` and its output files apply one-month XGBoost
full-payoff probabilities as constant monthly hazards in a long-horizon
projection. They do not use scenario-consistent future rate paths, refreshed
path-dependent features, total-prepayment/CPR targets, or a security-specific
market calibration.

Do not report the experimental XGBoost duration or convexity as a final
result.
