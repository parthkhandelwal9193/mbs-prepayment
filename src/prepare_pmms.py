from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PMMS_FILE = RAW / "pmms_mortgage30us.csv"
START = pd.Timestamp("2018-01-01")
END = pd.Timestamp("2026-03-01")


def main():
    raw = pd.read_csv(PMMS_FILE)
    print("Input CSV columns:", raw.columns.tolist())

    # FRED CSV normally uses observation_date and MORTGAGE30US.
    date_col = next(
        c for c in raw.columns
        if c.strip().lower() in {"observation_date", "date"}
    )
    rate_col = next(
        c for c in raw.columns
        if c.strip().lower() in {"mortgage30us", "value", "rate"}
    )

    rates = raw[[date_col, rate_col]].rename(
        columns={date_col: "week_date", rate_col: "pmms_rate_pct"}
    )
    rates["week_date"] = pd.to_datetime(rates["week_date"], errors="coerce")
    rates["pmms_rate_pct"] = pd.to_numeric(
        rates["pmms_rate_pct"], errors="coerce"
    )

    rates = rates.dropna(subset=["week_date", "pmms_rate_pct"])
    rates = rates.sort_values("week_date").drop_duplicates(
        "week_date", keep="last"
    )

    # Keep only the time span overlapping our loan-performance data.
    rates = rates[
        rates["week_date"].between(START, END + pd.offsets.MonthEnd(0))
    ]

    # Freddie Mac loan performance uses YYYYMM periods. Represent each as
    # the first day of the month. Select the latest weekly PMMS observation
    # on or before that reporting month.
    months = pd.DataFrame({
        "monthly_reporting_period": pd.date_range(START, END, freq="MS")
    })
    months = pd.merge_asof(
        months.sort_values("monthly_reporting_period"),
        rates.sort_values("week_date"),
        left_on="monthly_reporting_period",
        right_on="week_date",
        direction="backward",
    )

    # First sample month can have no prior weekly observation after filtering.
    # For that month only, use the earliest PMMS observation in January 2018.
    if months["pmms_rate_pct"].isna().any():
        first_week = rates.loc[rates["week_date"] >= START].iloc[0]
        missing = months["pmms_rate_pct"].isna()
        months.loc[missing, "pmms_rate_pct"] = first_week["pmms_rate_pct"]
        months.loc[missing, "week_date"] = first_week["week_date"]

    months = months.rename(columns={"week_date": "pmms_observation_date"})
    out = PROCESSED / "pmms_monthly.parquet"
    months.to_parquet(out, index=False)

    print("Monthly rows:", len(months))
    print("Date span:",
          months["monthly_reporting_period"].min(),
          "to",
          months["monthly_reporting_period"].max())
    print("\nFirst rows:")
    print(months.head(6).to_string(index=False))
    print("\nAnnual rate summary:")
    summary = months.assign(
        year=months["monthly_reporting_period"].dt.year
    ).groupby("year")["pmms_rate_pct"].agg(["min", "mean", "max"])
    print(summary.round(2).to_string())
    print("\nSaved:", out)


if __name__ == "__main__":
    main()