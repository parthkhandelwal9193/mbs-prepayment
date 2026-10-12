from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "panel_with_rates.parquet"


def main():
    panel = pd.read_parquet(PROCESSED / "panel_events.parquet")
    rates = pd.read_parquet(PROCESSED / "pmms_monthly.parquet")

    panel["monthly_reporting_period"] = pd.to_datetime(
        panel["monthly_reporting_period"]
    )
    rates["monthly_reporting_period"] = pd.to_datetime(
        rates["monthly_reporting_period"]
    )

    # Use current_interest_rate: it reflects the note rate in effect that month,
    # including modifications. It is preferable to original_interest_rate here.
    panel["current_interest_rate"] = pd.to_numeric(
        panel["current_interest_rate"], errors="coerce"
    )

    panel = panel.merge(
        rates[["monthly_reporting_period", "pmms_rate_pct"]],
        on="monthly_reporting_period",
        how="left",
        validate="many_to_one",
    )

    missing = int(panel["pmms_rate_pct"].isna().sum())
    print("Rows:", f"{len(panel):,}")
    print("Rows with missing PMMS:", f"{missing:,}")

    if missing:
        raise ValueError("PMMS rate missing for one or more loan-month dates.")

    # Rates are percentage points: 6.5 means 6.5%.
    panel["refi_incentive_pp"] = (
        panel["current_interest_rate"] - panel["pmms_rate_pct"]
    )
    panel["month_of_year"] = (
        panel["monthly_reporting_period"].dt.month.astype("int8")
    )
    panel["observation_year"] = (
        panel["monthly_reporting_period"].dt.year.astype("int16")
    )

    print("\nRefinance-incentive summary (percentage points):")
    print(
        panel["refi_incentive_pp"]
        .describe(percentiles=[0.01, 0.50, 0.99])
        .round(3)
        .to_string()
    )

    print("\nMean incentive by observation year:")
    print(
        panel.groupby("observation_year")["refi_incentive_pp"]
        .mean()
        .round(2)
        .to_string()
    )

    panel.to_parquet(OUT, index=False)
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()