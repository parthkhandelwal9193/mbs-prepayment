from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = P / "feature_validation.parquet"


def main():
    df = pd.read_parquet(P / "panel_with_rates.parquet")
    df = df.sort_values(
        ["loan_sequence_number", "monthly_reporting_period"]
    )

    # Ensure numeric rate and age fields.
    for col in [
        "current_interest_rate",
        "pmms_rate_pct",
        "loan_age",
        "remaining_months_to_legal_maturity",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["zero_balance_code"] = (
        df["zero_balance_code"].astype("string").str.strip()
    )
    df["refi_incentive_pp"] = (
        df["current_interest_rate"] - df["pmms_rate_pct"]
    )

    print("Rows:", f"{len(df):,}")
    print("Missing current interest rate:",
          int(df["current_interest_rate"].isna().sum()))
    print("Missing PMMS rate:", int(df["pmms_rate_pct"].isna().sum()))

    print("\nRate and incentive summary:")
    print(
        df[[
            "current_interest_rate",
            "pmms_rate_pct",
            "refi_incentive_pp",
        ]]
        .describe(percentiles=[0.01, 0.50, 0.99])
        .round(3)
        .to_string()
    )

    # Count code-01 payoffs and competing/other exits.
    df["code01_event"] = (
        df["zero_balance_code"].eq("01").fillna(False).astype("int8")
    )
    other_codes = {"02", "03", "09", "15", "16", "96"}
    df["other_exit_event"] = (
        df["zero_balance_code"].isin(other_codes).astype("int8")
    )

    print("\nExit rows by code:")
    print(df["zero_balance_code"].value_counts(dropna=False).head(12).to_string())

    # Verify terminal code rows are at the end of the loan history.
    last_date = df.groupby("loan_sequence_number")[
        "monthly_reporting_period"
    ].transform("max")
    terminal = df["zero_balance_code"].notna()
    print("\nTerminal rows not on the loan's last observed month:",
          int((terminal & df["monthly_reporting_period"].ne(last_date)).sum()))

    # Descriptive only: this is not yet the final hazard-model estimate.
    df["incentive_bin"] = pd.cut(
        df["refi_incentive_pp"],
        bins=[-20, -2, -1, 0, 1, 2, 3, 20],
        labels=["<=-2", "-2 to -1", "-1 to 0",
                "0 to 1", "1 to 2", "2 to 3", ">3"],
        right=False,
    )

    by_bin = df.groupby("incentive_bin", observed=False).agg(
        loan_months=("loan_sequence_number", "size"),
        code01_events=("code01_event", "sum"),
        other_exit_events=("other_exit_event", "sum"),
        average_loan_age=("loan_age", "mean"),
    )
    by_bin["crude_code01_rate_pct"] = (
        100 * by_bin["code01_events"] / by_bin["loan_months"]
    )

    print("\nCrude observed code-01 rate by refinance-incentive bucket:")
    print(by_bin.round(3).to_string())

    # Save a compact diagnostic table, not the full duplicate dataset.
    by_bin.to_csv(P / "incentive_bucket_check.csv")
    print("\nSaved:", P / "incentive_bucket_check.csv")


if __name__ == "__main__":
    main()