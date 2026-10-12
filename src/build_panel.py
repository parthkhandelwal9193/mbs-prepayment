from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

VINTAGES = range(2018, 2026)

# 0-based positions in the raw pipe-delimited files (orig=31 fields, perf=35 fields)
ORIG_USE = {
    0: "credit_score", 1: "first_payment_date", 7: "occupancy_status",
    8: "original_cltv", 9: "original_dti", 10: "original_upb", 11: "original_ltv",
    12: "original_interest_rate", 16: "property_state", 19: "loan_sequence_number",
    20: "loan_purpose", 21: "original_loan_term",
}
PERF_USE = {
    0: "loan_sequence_number", 1: "monthly_reporting_period", 2: "current_actual_upb",
    3: "current_delinquency_status", 4: "loan_age", 5: "remaining_months_to_legal_maturity",
    7: "modification_flag", 8: "zero_balance_code", 9: "zero_balance_effective_date",
    10: "current_interest_rate",
}


def read_selected(path, use_map):
    cols = sorted(use_map)
    names = [use_map[c] for c in cols]
    return pd.read_csv(
        path, sep="|", header=None, usecols=cols, names=names,
        dtype="string", engine="c",
    )


def check_field_count(path, expected):
    with open(path) as f:
        n = len(f.readline().rstrip("\n").split("|"))
    if n != expected:
        raise ValueError(f"{path.name}: found {n} fields, expected {expected}")


def clean_orig(df, vintage):
    for c in ["credit_score", "original_cltv", "original_dti", "original_upb",
              "original_ltv", "original_interest_rate", "original_loan_term"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["credit_score"] >= 9999, "credit_score"] = np.nan
    for c in ["original_cltv", "original_dti", "original_ltv"]:
        df.loc[df[c] >= 999, c] = np.nan
    df["first_payment_date"] = pd.to_datetime(df["first_payment_date"], format="%Y%m", errors="coerce")
    df["loan_sequence_number"] = df["loan_sequence_number"].str.strip()
    df["origination_vintage"] = vintage
    return df


def clean_perf(df, vintage):
    df["loan_sequence_number"] = df["loan_sequence_number"].str.strip()
    df["monthly_reporting_period"] = pd.to_datetime(df["monthly_reporting_period"], format="%Y%m", errors="coerce")
    for c in ["current_actual_upb", "loan_age", "remaining_months_to_legal_maturity", "current_interest_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["delinquency_num"] = pd.to_numeric(df["current_delinquency_status"], errors="coerce")
    df["zero_balance_code"] = df["zero_balance_code"].str.strip()
    df["zero_balance_effective_date"] = pd.to_datetime(df["zero_balance_effective_date"], format="%Y%m", errors="coerce")
    df["origination_vintage"] = vintage
    return df


def build_vintage(vintage):
    folder = RAW / f"sample_{vintage}"
    orig_path = folder / f"sample_orig_{vintage}.txt"
    perf_path = folder / f"sample_perf_{vintage}.txt"
    check_field_count(orig_path, 31)
    check_field_count(perf_path, 35)

    orig = clean_orig(read_selected(orig_path, ORIG_USE), vintage)
    perf = clean_perf(read_selected(perf_path, PERF_USE), vintage)

    assert orig["loan_sequence_number"].is_unique, f"{vintage}: duplicate loans in origination"
    dups = perf.duplicated(["loan_sequence_number", "monthly_reporting_period"]).sum()
    assert dups == 0, f"{vintage}: {dups} duplicate loan-months"

    orig.to_parquet(PROCESSED / f"orig_{vintage}.parquet", index=False)
    perf.to_parquet(PROCESSED / f"perf_{vintage}.parquet", index=False)
    print(f"{vintage}: loans={len(orig):,} loan_months={len(perf):,} "
          f"dates={perf['monthly_reporting_period'].min():%Y-%m}..{perf['monthly_reporting_period'].max():%Y-%m}")
    return len(orig), len(perf)


def main():
    total_loans = total_rows = 0
    for v in VINTAGES:
        l, r = build_vintage(v)
        total_loans += l
        total_rows += r
    print(f"\nTOTAL loans={total_loans:,} loan_months={total_rows:,}")

    orig_all = pd.concat([pd.read_parquet(PROCESSED / f"orig_{v}.parquet") for v in VINTAGES], ignore_index=True)
    orig_all.to_parquet(PROCESSED / "orig_all.parquet", index=False)
    perf_all = pd.concat([pd.read_parquet(PROCESSED / f"perf_{v}.parquet") for v in VINTAGES], ignore_index=True)
    perf_all.to_parquet(PROCESSED / "perf_all.parquet", index=False)
    print("combined origination:", orig_all.shape, "| combined performance:", perf_all.shape)
    print("unique loans across vintages:", orig_all["loan_sequence_number"].nunique())


if __name__ == "__main__":
    main()