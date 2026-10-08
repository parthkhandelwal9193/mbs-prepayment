from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sample_2018"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ORIG_COLUMNS = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer",
    "maturity_date",
    "msa",
    "mi_percent",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty",
    "property_state",
    "property_type",
    "postal_code",
    "loan_sequence_number",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "servicer_name",
    "super_conforming_flag",
]

PERF_COLUMNS = [
    "loan_sequence_number",
    "monthly_reporting_period",
    "current_actual_upb",
    "current_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_deferred_upb",
    "due_date_of_last_paid_installment",
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maintenance_and_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss_calculation",
    "modification_cost",
    "step_modification_flag",
    "deferred_payment_plan",
    "estimated_loan_to_value",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
]


def read_pipe_file(path, columns):
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        dtype="string",
    )

    if df.shape[1] != len(columns):
        raise ValueError(
            f"{path} has {df.shape[1]} columns, "
            f"but the loader expects {len(columns)}"
        )

    return df


orig_path = RAW_DIR / "sample_orig_2018.txt"
perf_path = RAW_DIR / "sample_perf_2018.txt"

orig = read_pipe_file(orig_path, ORIG_COLUMNS)
perf = read_pipe_file(perf_path, PERF_COLUMNS)

orig["loan_sequence_number"] = orig["loan_sequence_number"].str.strip()
perf["loan_sequence_number"] = perf["loan_sequence_number"].str.strip()

perf["monthly_reporting_period"] = pd.to_datetime(
    perf["monthly_reporting_period"].str.strip(),
    format="%Y%m",
    errors="coerce",
)

perf["zero_balance_code"] = perf["zero_balance_code"].str.strip()

orig.to_parquet(
    PROCESSED_DIR / "orig_2018_raw.parquet",
    index=False,
)

perf.to_parquet(
    PROCESSED_DIR / "perf_2018_raw.parquet",
    index=False,
)

print("Origination shape:", orig.shape)
print("Performance shape:", perf.shape)
print(
    "Unique origination loans:",
    orig["loan_sequence_number"].nunique(),
)
print(
    "Unique performance loans:",
    perf["loan_sequence_number"].nunique(),
)
print(
    "Performance date range:",
    perf["monthly_reporting_period"].min(),
    "to",
    perf["monthly_reporting_period"].max(),
)
print("\nZero-balance codes:")
print(perf["zero_balance_code"].value_counts(dropna=False).head(15))
print("\nSaved files to data/processed/")