from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def audit(orig, perf):
    perf = perf.copy()
    perf["lid"] = pd.factorize(perf["loan_sequence_number"])[0]
    perf["m"] = perf["monthly_reporting_period"].dt.year * 12 + perf["monthly_reporting_period"].dt.month
    perf = perf.sort_values(["lid", "m"]).reset_index(drop=True)

    section("1. Join integrity")
    o_ids, p_ids = set(orig["loan_sequence_number"]), set(perf["loan_sequence_number"])
    print("loans in orig:", len(o_ids), "| loans in perf:", len(p_ids))
    print("perf loans missing from orig:", len(p_ids - o_ids))
    print("orig loans missing from perf:", len(o_ids - p_ids))

    section("2. Missing values in columns we use")
    print((orig.isna().mean() * 100).round(2).to_string())
    print()
    print((perf.drop(columns=["lid", "m"]).isna().mean() * 100).round(2).to_string())

    section("3. Value ranges (origination)")
    print(orig[["credit_score", "original_ltv", "original_cltv", "original_dti",
                "original_upb", "original_interest_rate", "original_loan_term"]]
          .describe(percentiles=[.01, .5, .99]).round(2).T.to_string())
    print("\nloan_purpose:\n", orig["loan_purpose"].value_counts(dropna=False).to_string())
    print("\nfirst payment year by vintage:")
    print(pd.crosstab(orig["origination_vintage"], orig["first_payment_date"].dt.year).to_string())

    section("4. Value ranges (performance)")
    print(perf[["current_actual_upb", "loan_age", "remaining_months_to_legal_maturity",
                "current_interest_rate"]].describe(percentiles=[.01, .5, .99]).round(2).T.to_string())
    print("\ndelinquency status (top):\n", perf["current_delinquency_status"].value_counts(dropna=False).head(8).to_string())
    print("\nmodification_flag:\n", perf["modification_flag"].value_counts(dropna=False).to_string())

    section("5. Time continuity per loan")
    gap = perf.groupby("lid")["m"].diff()
    print("rows where next month is not +1:", int((gap.dropna() != 1).sum()))
    age_step = perf.groupby("lid")["loan_age"].diff()
    print("rows where loan_age does not rise by 1:", int((age_step.dropna() != 1).sum()))

    section("6. Termination events (loan level)")
    term = perf[perf["zero_balance_code"].notna()]
    per_loan = term.groupby("lid").size()
    print("loans with a termination row:", len(per_loan), "| loans with >1:", int((per_loan > 1).sum()))
    print("loan-level zero balance codes:\n", term["zero_balance_code"].value_counts().to_string())
    last_idx = perf.groupby("lid").tail(1).index
    last_rows = perf.loc[last_idx]
    has_term = last_rows["zero_balance_code"].notna()
    print("\nloans whose LAST row carries the code:", int(has_term.sum()))
    print("terminated loans with rows AFTER the code row:",
          int(len(per_loan) - has_term.sum()))
    print("UPB on code-01 rows (should be ~0):")
    print(term.loc[term["zero_balance_code"] == "01", "current_actual_upb"].describe().round(2).to_string())
    print("\nstill-active loans at end of data:", int((~has_term).sum()))
    print("last observed month of still-active loans:")
    print(last_rows.loc[~has_term, "monthly_reporting_period"].value_counts().head(3).to_string())

    section("7. Sanity check: monthly prepayment rate (count based)")
    perf["is_pp"] = (
    perf["zero_balance_code"]
    .fillna("")
    .astype("string")
    .str.strip()
    .eq("01")
    .astype("int8")
)
    monthly = perf.groupby("monthly_reporting_period").agg(at_risk=("lid", "size"), pp=("is_pp", "sum"))
    monthly["smm_pct"] = (monthly["pp"] / monthly["at_risk"] * 100).round(3)
    monthly["cpr_pct"] = ((1 - (1 - monthly["smm_pct"] / 100) ** 12) * 100).round(1)
    yearly = perf.assign(year=perf["monthly_reporting_period"].dt.year).groupby("year").agg(
        at_risk=("lid", "size"), pp=("is_pp", "sum"))
    yearly["avg_smm_pct"] = (yearly["pp"] / yearly["at_risk"] * 100).round(3)
    yearly["approx_cpr_pct"] = ((1 - (1 - yearly["avg_smm_pct"] / 100) ** 12) * 100).round(1)
    print(yearly.to_string())
    monthly.to_csv(PROCESSED / "audit_monthly_prepay.csv")
    print("\nsaved data/processed/audit_monthly_prepay.csv")


def main():
    orig = pd.read_parquet(PROCESSED / "orig_all.parquet")
    perf = pd.read_parquet(PROCESSED / "perf_all.parquet")
    audit(orig, perf)


if __name__ == "__main__":
    main()