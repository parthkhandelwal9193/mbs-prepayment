from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data/processed/hazard_dev_sample.parquet"
PRED_PATH = (
    ROOT / "outputs/predictions/"
    "xgb_lagged_pmms_regime_fixed_predictions.parquet"
)
OUT = ROOT / "outputs/reports/model_validation"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["loan_sequence_number", "monthly_reporting_period", "split"]
SPLITS = ["test", "holdout_2025_2026"]

sample = pd.read_parquet(
    SAMPLE_PATH,
    columns=KEYS + [
        "event",
        "current_actual_upb",
        "zero_balance_removal_upb",
    ],
)
pred = pd.read_parquet(
    PRED_PATH,
    columns=KEYS + ["event", "raw_prediction", "isotonic_prediction"],
)

for frame in (sample, pred):
    frame["monthly_reporting_period"] = pd.to_datetime(
        frame["monthly_reporting_period"], errors="raise"
    )

sample["event"] = pd.to_numeric(sample["event"], errors="raise").astype("int8")
pred["event"] = pd.to_numeric(pred["event"], errors="raise").astype("int8")
sample["current_actual_upb"] = pd.to_numeric(
    sample["current_actual_upb"], errors="coerce"
)
sample["zero_balance_removal_upb"] = pd.to_numeric(
    sample["zero_balance_removal_upb"], errors="coerce"
)
pred["raw_prediction"] = pd.to_numeric(pred["raw_prediction"], errors="raise")
pred["isotonic_prediction"] = pd.to_numeric(
    pred["isotonic_prediction"], errors="raise"
)

if not pred["raw_prediction"].between(0, 1).all():
    raise ValueError("Raw probabilities must be between 0 and 1.")
if not pred["isotonic_prediction"].between(0, 1).all():
    raise ValueError("Isotonic probabilities must be between 0 and 1.")

# Merge and verify exact loan-month/split alignment.
joined = sample.merge(
    pred.rename(columns={"event": "prediction_event"}),
    on=KEYS,
    how="inner",
    validate="one_to_one",
)

if len(joined) != len(pred):
    raise ValueError(
        f"Prediction/sample mismatch: predictions={len(pred):,}, "
        f"matched={len(joined):,}"
    )

mismatches = int(joined["event"].ne(joined["prediction_event"]).sum())
if mismatches:
    raise ValueError(f"Event-label mismatches: {mismatches:,}")

# Previous balance is valid as prediction exposure only when the prior
# observation is the immediately preceding calendar month.
joined = joined.sort_values(KEYS[:1] + ["monthly_reporting_period"]).copy()
loan = joined["loan_sequence_number"]
month_id = (
    joined["monthly_reporting_period"].dt.year * 12
    + joined["monthly_reporting_period"].dt.month
)
prior_month_id = month_id.groupby(loan, sort=False).shift(1)
joined["previous_month_upb"] = (
    joined.groupby("loan_sequence_number", sort=False)["current_actual_upb"]
    .shift(1)
)
joined["has_consecutive_prior_month"] = (month_id - prior_month_id).eq(1)

# Drop first observations and any nonconsecutive loan histories rather than
# inventing an exposure. Keep rows with positive prior-month UPB.
joined = joined.loc[
    joined["has_consecutive_prior_month"]
    & joined["previous_month_upb"].gt(0)
    & joined["split"].isin(SPLITS)
].copy()

if joined.empty:
    raise ValueError("No eligible test/holdout rows with prior-month UPB.")

# Predicted expected full-payoff principal uses prior-month balance.
joined["predicted_raw_payoff_upb"] = (
    joined["raw_prediction"] * joined["previous_month_upb"]
)
joined["predicted_isotonic_payoff_upb"] = (
    joined["isotonic_prediction"] * joined["previous_month_upb"]
)

# Realized principal uses the reported balance removed when the loan reaches
# zero balance. Only event rows contribute; non-events contribute zero.
joined["realized_payoff_upb"] = 0.0
is_event = joined["event"].eq(1)
missing_event_removal = int(
    joined.loc[is_event, "zero_balance_removal_upb"].isna().sum()
)
nonpositive_event_removal = int(
    joined.loc[is_event, "zero_balance_removal_upb"].fillna(0).le(0).sum()
)
if missing_event_removal or nonpositive_event_removal:
    raise ValueError(
        "Some matched event rows have missing/nonpositive removal UPB: "
        f"missing={missing_event_removal:,}, "
        f"nonpositive={nonpositive_event_removal:,}"
    )
joined.loc[is_event, "realized_payoff_upb"] = joined.loc[
    is_event, "zero_balance_removal_upb"
]

monthly = (
    joined.groupby(
        ["split", "monthly_reporting_period"],
        as_index=False,
        observed=True,
    )
    .agg(
        loan_month_rows=("event", "size"),
        actual_events=("event", "sum"),
        beginning_upb=("previous_month_upb", "sum"),
        realized_payoff_upb=("realized_payoff_upb", "sum"),
        predicted_raw_payoff_upb=("predicted_raw_payoff_upb", "sum"),
        predicted_isotonic_payoff_upb=(
            "predicted_isotonic_payoff_upb", "sum"
        ),
    )
)

for stem in ["realized", "predicted_raw", "predicted_isotonic"]:
    monthly[f"{stem}_payoff_rate"] = (
        monthly[f"{stem}_payoff_upb"] / monthly["beginning_upb"]
    )

monthly_path = OUT / "monthly_predicted_vs_realized_payoff_upb.csv"
monthly.to_csv(monthly_path, index=False)

summary_rows = []
for split, group in monthly.groupby("split", observed=True):
    denominator = group["beginning_upb"].sum()
    summary_rows.append({
        "split": split,
        "months": int(group["monthly_reporting_period"].nunique()),
        "loan_month_rows": int(group["loan_month_rows"].sum()),
        "actual_events": int(group["actual_events"].sum()),
        "prior_balance_sum_over_months": float(denominator),
        "realized_payoff_upb": float(group["realized_payoff_upb"].sum()),
        "predicted_raw_payoff_upb": float(
            group["predicted_raw_payoff_upb"].sum()
        ),
        "predicted_isotonic_payoff_upb": float(
            group["predicted_isotonic_payoff_upb"].sum()
        ),
        "realized_payoff_rate": float(
            group["realized_payoff_upb"].sum() / denominator
        ),
        "predicted_raw_payoff_rate": float(
            group["predicted_raw_payoff_upb"].sum() / denominator
        ),
        "predicted_isotonic_payoff_rate": float(
            group["predicted_isotonic_payoff_upb"].sum() / denominator
        ),
    })

summary = pd.DataFrame(summary_rows)
summary_path = OUT / "payoff_upb_forecast_summary.csv"
summary.to_csv(summary_path, index=False)

fig, axes = plt.subplots(2, 1, figsize=(11, 8))
for ax, split in zip(axes, SPLITS):
    g = monthly.loc[monthly["split"].eq(split)].sort_values(
        "monthly_reporting_period"
    )
    ax.plot(
        g["monthly_reporting_period"],
        100 * g["realized_payoff_rate"],
        label="Realized removal UPB / prior-month UPB",
        linewidth=1.8,
    )
    ax.plot(
        g["monthly_reporting_period"],
        100 * g["predicted_isotonic_payoff_rate"],
        label="Predicted probability × prior-month UPB",
        linestyle="--",
        linewidth=1.8,
    )
    ax.set_title(split)
    ax.set_ylabel("Full-payoff rate (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

axes[-1].set_xlabel("Month")
fig.suptitle("Predicted versus realized full-payoff UPB rate")
fig.tight_layout()
plot_path = OUT / "monthly_predicted_vs_realized_payoff_upb.png"
fig.savefig(plot_path, dpi=180)
plt.close(fig)

print("Matched predictions:", f"{len(pred):,}")
print("Eligible consecutive-prior-month rows:", f"{len(joined):,}")
print("Event-label mismatches:", mismatches)
print("Events with missing removal UPB:", missing_event_removal)
print("Events with nonpositive removal UPB:", nonpositive_event_removal)
print("\nSummary:")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\nSaved:")
print(monthly_path)
print(summary_path)
print(plot_path)
print(
    "\nThis is a balance-weighted full-loan-payoff diagnostic, not total CPR: "
    "partial curtailments are absent from the target."
)
