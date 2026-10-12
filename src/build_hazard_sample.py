from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INPUT_FILE = PROCESSED / "dev_sample.parquet"
OUTPUT_FILE = PROCESSED / "hazard_dev_sample.parquet"

# Freddie Mac zero-balance code 01 means prepaid OR matured voluntary payoff.
# Here we define the target as early payoff only: code 01 with positive remaining
# months to legal maturity. Missing/zero remaining term is not counted as event.
PREPAY_CODE = "01"
OTHER_EXIT_CODES = {"02", "03", "09", "15", "16", "96"}
TERMINAL_CODES = OTHER_EXIT_CODES | {PREPAY_CODE}

COLUMNS = [
    "loan_sequence_number", "monthly_reporting_period", "loan_age",
    "current_actual_upb", "current_delinquency_status", "zero_balance_code",
    "current_interest_rate", "pmms_rate_pct", "refi_incentive_pp",
    "modification_flag", "remaining_months_to_legal_maturity", "original_upb",
    "original_ltv", "original_cltv", "original_dti", "credit_score",
    "original_loan_term", "loan_purpose", "occupancy_status", "property_state",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)
    available = set(pd.read_parquet(INPUT_FILE, engine="pyarrow").columns)
    missing = sorted(set(COLUMNS) - available)
    if missing:
        raise ValueError(f"Missing fields in dev sample: {missing}")

    df = pd.read_parquet(INPUT_FILE, columns=COLUMNS)
    string_cols = [
        "loan_sequence_number", "current_delinquency_status", "zero_balance_code",
        "modification_flag", "loan_purpose", "occupancy_status", "property_state",
    ]
    for col in string_cols:
        df[col] = df[col].astype("string").str.strip()
    df["monthly_reporting_period"] = pd.to_datetime(
        df["monthly_reporting_period"], errors="coerce"
    )
    if df["monthly_reporting_period"].isna().any():
        raise ValueError("Invalid monthly_reporting_period found")

    numeric = [
        "loan_age", "current_actual_upb", "current_interest_rate", "pmms_rate_pct",
        "refi_incentive_pp", "remaining_months_to_legal_maturity", "original_upb",
        "original_ltv", "original_cltv", "original_dti", "credit_score",
        "original_loan_term",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(
        ["loan_sequence_number", "monthly_reporting_period"]
    ).reset_index(drop=True)
    dup_input = int(df.duplicated(["loan_sequence_number", "monthly_reporting_period"]).sum())
    if dup_input:
        raise ValueError(f"Input has {dup_input} duplicate loan-month rows")

    terminal = df["zero_balance_code"].isin(TERMINAL_CODES).fillna(False)
    term_cum = terminal.astype("int8").groupby(df["loan_sequence_number"], sort=False).cumsum()
    terminal_before = term_cum.groupby(df["loan_sequence_number"], sort=False).shift(fill_value=0).gt(0)
    input_rows_after_terminal = int(terminal_before.sum())

    code01 = df["zero_balance_code"].eq(PREPAY_CODE).fillna(False)
    remaining = df["remaining_months_to_legal_maturity"]
    early_event = code01 & remaining.gt(0).fillna(False)
    code01_mature_or_unknown = code01 & ~remaining.gt(0).fillna(False)
    df["event"] = early_event.astype("int8")
    df["other_exit"] = (
        df["zero_balance_code"].isin(OTHER_EXIT_CODES).fillna(False)
        | code01_mature_or_unknown
    )

    # Keep the terminal month; remove only rows strictly after a terminal month.
    df = df.loc[~terminal_before].copy()

    is_event = df["event"].eq(1)
    is_current = df["current_delinquency_status"].eq("00").fillna(False)
    eligible = (is_current | is_event) & ~df["other_exit"]

    # Required predictors must be observed. Positive UPB only for non-event rows.
    essential = ["loan_age", "current_interest_rate", "pmms_rate_pct", "refi_incentive_pp"]
    eligible &= df[essential].notna().all(axis=1)
    eligible &= is_event | (is_current & df["current_actual_upb"].gt(0).fillna(False))
    eligible &= df["loan_sequence_number"].notna()
    df = df.loc[eligible].copy()

    dates = df["monthly_reporting_period"]
    df["split"] = "holdout_2025_2026"
    df.loc[dates < pd.Timestamp("2022-01-01"), "split"] = "train"
    df.loc[(dates >= "2022-01-01") & (dates < "2023-01-01"), "split"] = "validation"
    df.loc[(dates >= "2023-01-01") & (dates < "2025-01-01"), "split"] = "test"
    df["observation_year"] = dates.dt.year.astype("int16")
    df["month_of_year"] = dates.dt.month.astype("int8")

    df = df.sort_values(["loan_sequence_number", "monthly_reporting_period"]).reset_index(drop=True)
    prior_events = df.groupby("loan_sequence_number", sort=False)["event"].cumsum() - df["event"]
    final_after_event = int(prior_events.gt(0).sum())
    final_duplicates = int(df.duplicated(["loan_sequence_number", "monthly_reporting_period"]).sum())
    if final_after_event or final_duplicates:
        raise ValueError(f"Final audit failed: after_event={final_after_event}, duplicates={final_duplicates}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print("Input rows:", f"{len(terminal):,}")
    print("Input rows after first terminal (removed):", input_rows_after_terminal)
    print("Input duplicate loan-months:", dup_input)
    print("All code-01 rows:", int(code01.sum()))
    print("Code-01 with remaining term > 0 (early-prepay target):", int(early_event.sum()))
    print("Code-01 with remaining term <= 0 or missing (excluded/censored):", int(code01_mature_or_unknown.sum()))
    print("Final rows:", f"{len(df):,}")
    print("Final unique loans:", f"{df.loan_sequence_number.nunique():,}")
    print("Final events:", int(df.event.sum()))
    print("Rows after event in final panel:", final_after_event)
    print("Duplicate loan-months in final panel:", final_duplicates)
    print("Saved:", OUTPUT_FILE)

    print("\\nCounts by split:")
    by_split = df.groupby("split").agg(
        rows=("event", "size"), events=("event", "sum"),
        event_rate=("event", "mean"), loans=("loan_sequence_number", "nunique"),
    )
    print(by_split.to_string(float_format=lambda v: f"{v:.6f}"))
    print("\\nCounts by calendar year:")
    by_year = df.groupby(df.monthly_reporting_period.dt.year).agg(
        rows=("event", "size"), events=("event", "sum"), event_rate=("event", "mean"),
    )
    print(by_year.to_string(float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()