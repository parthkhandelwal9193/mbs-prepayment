from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs/model_scenario_eda/test_dec2024_xgb_rate_shock_feature_audit_predictions.parquet"
SUPPORT = ROOT / "outputs/model_scenario_eda/test_dec2024_xgb_rate_shock_feature_support_v2.csv"
OUT = ROOT / "outputs/model_scenario_eda"
OUT.mkdir(parents=True, exist_ok=True)

SHOCKS = [-100, 0, 100]

pred = pd.read_parquet(INPUT)
support = pd.read_csv(SUPPORT)

required_pred = {
    "loan_sequence_number",
    "monthly_reporting_period",
    "shock_bp",
    "current_actual_upb",
    "shocked_refi_incentive_pp_lag1",
    "raw_prediction",
    "isotonic_prediction",
}
missing = required_pred - set(pred.columns)
if missing:
    raise ValueError(f"Prediction file missing columns: {sorted(missing)}")

required_support = {"shock_bp", "train_min", "train_max"}
missing = required_support - set(support.columns)
if missing:
    raise ValueError(f"Support file missing columns: {sorted(missing)}")

pred = pred.loc[pred["shock_bp"].isin(SHOCKS)].copy()
pred = pred.merge(
    support[["shock_bp", "train_min", "train_max"]],
    on="shock_bp",
    how="left",
    validate="many_to_one",
)
pred["in_support"] = pred["shocked_refi_incentive_pp_lag1"].between(
    pred["train_min"],
    pred["train_max"],
)

key = ["loan_sequence_number", "monthly_reporting_period"]
counts = (
    pred.groupby(key, as_index=False)
    .agg(
        scenarios=("shock_bp", "nunique"),
        supported_scenarios=("in_support", "sum"),
        current_actual_upb=("current_actual_upb", "first"),
    )
)

common = counts.loc[
    counts["scenarios"].eq(len(SHOCKS))
    & counts["supported_scenarios"].eq(len(SHOCKS))
].copy()

if common.empty:
    raise ValueError("No common-support loans remain across all three scenarios.")

common_keys = common[key]
common_pred = pred.merge(
    common_keys,
    on=key,
    how="inner",
    validate="many_to_one",
).sort_values(["shock_bp", "loan_sequence_number"])

base_total_upb = (
    pred.loc[pred["shock_bp"].eq(0), "current_actual_upb"]
    .astype(float)
    .sum()
)
common_upb = common["current_actual_upb"].astype(float).sum()

summary = pd.DataFrame([{
    "cohort_month": pd.to_datetime(
        common["monthly_reporting_period"].iloc[0]
    ).strftime("%Y-%m-%d"),
    "all_loans": int(
        pred.loc[pred["shock_bp"].eq(0), "loan_sequence_number"].nunique()
    ),
    "common_support_loans": int(common["loan_sequence_number"].nunique()),
    "share_of_loans": float(
        common["loan_sequence_number"].nunique()
        / pred.loc[pred["shock_bp"].eq(0), "loan_sequence_number"].nunique()
    ),
    "all_cohort_upb": float(base_total_upb),
    "common_support_upb": float(common_upb),
    "share_of_upb": float(common_upb / base_total_upb),
    "training_incentive_min": float(support["train_min"].iloc[0]),
    "training_incentive_max": float(support["train_max"].iloc[0]),
}])

common_pred.to_parquet(
    OUT / "test_dec2024_xgb_rate_shock_common_support_predictions.parquet",
    index=False,
)
summary.to_csv(
    OUT / "test_dec2024_xgb_rate_shock_common_support_summary.csv",
    index=False,
)

print("Common-support cohort:")
print(summary.to_string(
    index=False,
    formatters={
        "share_of_loans": lambda x: f"{x:.2%}",
        "all_cohort_upb": lambda x: f"${x:,.0f}",
        "common_support_upb": lambda x: f"${x:,.0f}",
        "share_of_upb": lambda x: f"{x:.2%}",
        "training_incentive_min": lambda x: f"{x:.3f}",
        "training_incentive_max": lambda x: f"{x:.3f}",
    },
))
print("\nWrote:")
print(OUT / "test_dec2024_xgb_rate_shock_common_support_predictions.parquet")
print(OUT / "test_dec2024_xgb_rate_shock_common_support_summary.csv")
