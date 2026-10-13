from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs/model_scenario_eda/test_dec2024_xgb_rate_shock_feature_audit_predictions.parquet"
SUPPORT = ROOT / "outputs/model_scenario_eda/test_dec2024_xgb_rate_shock_feature_support_v2.csv"
OUT = ROOT / "outputs/model_scenario_eda"

pred = pd.read_parquet(INPUT)
support = pd.read_csv(SUPPORT)

pred = pred.merge(
    support[["shock_bp", "train_min", "train_max"]],
    on="shock_bp",
    how="left",
    validate="many_to_one",
)

x = pred["shocked_refi_incentive_pp_lag1"]
pred["in_training_range"] = x.between(pred["train_min"], pred["train_max"])

rows = []
for shock_bp, g in pred.groupby("shock_bp", sort=True):
    for sample, sub in [("full_cohort", g), ("in_support_only", g[g["in_training_range"]])]:
        upb = sub["current_actual_upb"].astype(float)
        rows.append({
            "shock_bp": shock_bp,
            "sample": sample,
            "loan_rows": len(sub),
            "share_of_loans": len(sub) / len(g),
            "cohort_upb": upb.sum(),
            "share_of_upb": upb.sum() / g["current_actual_upb"].astype(float).sum(),
            "raw_upb_weighted_probability": np.average(
                sub["raw_prediction"], weights=upb
            ) if len(sub) else np.nan,
            "isotonic_upb_weighted_probability": np.average(
                sub["isotonic_prediction"], weights=upb
            ) if len(sub) else np.nan,
        })

summary = pd.DataFrame(rows)
summary.to_csv(
    OUT / "test_dec2024_xgb_rate_shock_support_restricted_summary.csv",
    index=False,
)
print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
print(
    "\nSaved:",
    OUT / "test_dec2024_xgb_rate_shock_support_restricted_summary.csv",
)
