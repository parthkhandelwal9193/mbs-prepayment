from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data/processed/hazard_dev_sample.parquet"
PRED_PATH = ROOT / "outputs/predictions/xgb_lagged_pmms_regime_fixed_predictions.parquet"
META_PATH = ROOT / "outputs/models/xgboost_lagged_pmms_regime_fixed_metadata.joblib"
MODEL_PATH = ROOT / "outputs/models/xgboost_lagged_pmms_regime_fixed.json"
OUT = ROOT / "outputs/model_scenario_eda"
OUT.mkdir(parents=True, exist_ok=True)

NUMERIC = [
    "refi_incentive_pp", "loan_age", "current_interest_rate",
    "remaining_months_to_legal_maturity", "original_upb", "original_ltv",
    "original_cltv", "original_dti", "credit_score", "original_loan_term",
]
CATEGORICAL = ["loan_purpose", "occupancy_status", "modification_flag"]
SENTINELS = {
    "original_ltv": 999, "original_cltv": 999,
    "original_dti": 999, "credit_score": 9999,
}
PMMS_CHANGE_DATE = pd.Timestamp("2022-11-17")
COHORT_MONTH = pd.Timestamp("2024-12-01")
SHOCKS_BP = [-100, 0, 100]


def safe_feature_names(columns):
    result, used = [], set()
    for i, col in enumerate(columns):
        name = re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_") or f"feature_{i}"
        base, suffix = name, 1
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)
        result.append(name)
    return result


def build_features(df):
    df = df.copy()
    df["monthly_reporting_period"] = pd.to_datetime(df["monthly_reporting_period"])
    df["event"] = pd.to_numeric(df["event"], errors="raise").astype("int8")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, sentinel in SENTINELS.items():
        df.loc[df[col] == sentinel, col] = np.nan

    df = df.sort_values(
        ["loan_sequence_number", "monthly_reporting_period"]
    ).reset_index(drop=True)

    loan_id = df["loan_sequence_number"]
    dates = df["monthly_reporting_period"]
    month_id = dates.dt.year * 12 + dates.dt.month
    consecutive = month_id.sub(
        month_id.groupby(loan_id, sort=False).shift(1)
    ).eq(1)

    for col in NUMERIC:
        df[f"{col}_lag1"] = df.groupby(
            "loan_sequence_number", sort=False
        )[col].shift(1)
        df.loc[~consecutive, f"{col}_lag1"] = np.nan

    incentive = df["refi_incentive_pp"]
    in_money = (incentive > 0.5).astype("int8")
    df["months_in_money_prior"] = (
        in_money.groupby(loan_id, sort=False).cumsum() - in_money
    )
    df["max_incentive_prior"] = incentive.groupby(
        loan_id, sort=False
    ).transform(lambda s: s.cummax().shift(1))
    df["incentive_lag1_prior"] = df["refi_incentive_pp_lag1"]
    df["observation_year"] = dates.dt.year.astype("int16")
    df["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    df["post_pmms_change"] = (dates >= PMMS_CHANGE_DATE).astype("int8")

    for threshold in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        suffix = str(threshold).replace("-", "m").replace(".", "_")
        df[f"incentive_hinge_prior_{suffix}"] = (
            df["refi_incentive_pp_lag1"] - threshold
        ).clip(lower=0)

    for threshold in [6, 12, 18, 24, 30, 36, 60, 120]:
        df[f"age_hinge_prior_{threshold}"] = (
            df["loan_age_lag1"] - threshold
        ).clip(lower=0)

    lag_features = [f"{col}_lag1" for col in NUMERIC]
    engineered = [
        "months_in_money_prior", "max_incentive_prior",
        "incentive_lag1_prior", "observation_year",
        "month_sin", "month_cos", "post_pmms_change",
    ]
    engineered += [
        col for col in df.columns
        if col.startswith("incentive_hinge_prior_")
        or col.startswith("age_hinge_prior_")
    ]
    features = lag_features + engineered + CATEGORICAL
    return df, features


def encode_matrix(frame, features, all_columns, encoded_names):
    matrix = pd.get_dummies(
        frame[features],
        columns=CATEGORICAL,
        dummy_na=True,
        dtype=np.uint8,
    )
    matrix = matrix.reindex(columns=all_columns, fill_value=0)
    matrix.columns = encoded_names
    return matrix


def main():
    for path in [INPUT, PRED_PATH, META_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = joblib.load(META_PATH)
    saved_features = metadata["features"]
    encoded_names = metadata["encoded_columns"]
    calibrator = metadata["calibrator"]

    required = {
        "loan_sequence_number", "monthly_reporting_period", "split", "event",
        "current_actual_upb",
        *NUMERIC, *CATEGORICAL,
    }
    df = pd.read_parquet(INPUT, columns=sorted(required))
    df, features = build_features(df)

    if features != saved_features:
        raise ValueError("Reconstructed feature list does not match saved metadata")

    required_lags = [
        "refi_incentive_pp_lag1", "loan_age_lag1",
        "current_interest_rate_lag1",
        "remaining_months_to_legal_maturity_lag1",
    ]
    cohort = df.loc[
        df["split"].eq("test")
        & df["monthly_reporting_period"].eq(COHORT_MONTH)
        & df[required_lags].notna().all(axis=1)
    ].copy()

    all_columns = sorted(
        set().union(
            *[
                set(
                    pd.get_dummies(
                        df.loc[df["split"].eq(split), features],
                        columns=CATEGORICAL,
                        dummy_na=True,
                        dtype=np.uint8,
                    ).columns
                )
                for split in df["split"].dropna().unique()
            ]
        )
    )
    reconstructed_names = safe_feature_names(all_columns)
    if reconstructed_names != encoded_names:
        raise ValueError("Reconstructed encoded columns do not match saved metadata")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    base_matrix = encode_matrix(cohort, features, all_columns, encoded_names)
    base_raw = model.predict_proba(base_matrix)[:, 1]
    base_iso = calibrator.predict(base_raw)

    saved = pd.read_parquet(PRED_PATH)
    saved_cohort = saved.loc[
        saved["split"].eq("test")
        & (pd.to_datetime(saved["monthly_reporting_period"]) == COHORT_MONTH)
    ].copy()

    check = cohort[
        ["loan_sequence_number", "monthly_reporting_period"]
    ].merge(
        saved_cohort[
            ["loan_sequence_number", "monthly_reporting_period",
             "raw_prediction", "isotonic_prediction"]
        ],
        on=["loan_sequence_number", "monthly_reporting_period"],
        how="inner",
        validate="one_to_one",
    )
    if len(check) != len(cohort):
        raise ValueError(
            f"Base cohort mismatch: reconstructed={len(cohort):,}, "
            f"matched saved={len(check):,}"
        )

    raw_diff = np.max(np.abs(base_raw - check["raw_prediction"].to_numpy()))
    iso_diff = np.max(np.abs(base_iso - check["isotonic_prediction"].to_numpy()))
    if raw_diff > 1e-8 or iso_diff > 1e-8:
        raise ValueError(
            f"Base prediction reproduction failed: max raw diff={raw_diff:.3g}, "
            f"max isotonic diff={iso_diff:.3g}"
        )

    train = df.loc[df["split"].eq("train")].copy()
    rate_feature = "refi_incentive_pp_lag1"
    train_low, train_high = (
        train[rate_feature].min(), train[rate_feature].max()
    )

    audit_rows = []
    prediction_rows = []

    for shock_bp in SHOCKS_BP:
        shocked = cohort.copy()
        shocked[rate_feature] = shocked[rate_feature] - shock_bp / 100.0
        shocked["incentive_lag1_prior"] = shocked[rate_feature]

        for threshold in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            suffix = str(threshold).replace("-", "m").replace(".", "_")
            shocked[f"incentive_hinge_prior_{suffix}"] = (
                shocked[rate_feature] - threshold
            ).clip(lower=0)

        matrix = encode_matrix(shocked, features, all_columns, encoded_names)
        raw = model.predict_proba(matrix)[:, 1]
        isotonic = calibrator.predict(raw)

        upb = pd.to_numeric(
            shocked["current_actual_upb"], errors="coerce"
        ).to_numpy()
        prediction_rows.append(pd.DataFrame({
    "loan_sequence_number": shocked["loan_sequence_number"].to_numpy(),
    "monthly_reporting_period": shocked[
        "monthly_reporting_period"
    ].to_numpy(),
    "shock_bp": shock_bp,
    "current_actual_upb": upb,
    "shocked_refi_incentive_pp_lag1": shocked[
        "refi_incentive_pp_lag1"
    ].to_numpy(),
    "raw_prediction": raw,
    "isotonic_prediction": isotonic,
}))

        outside = (
            (shocked[rate_feature] < train_low)
            | (shocked[rate_feature] > train_high)
        )
        audit_rows.append({
            "shock_bp": shock_bp,
            "loan_rows": len(shocked),
            "cohort_upb": float(np.nansum(upb)),
            "raw_upb_weighted_probability": float(
                np.nansum(raw * upb) / np.nansum(upb)
            ),
            "isotonic_upb_weighted_probability": float(
                np.nansum(isotonic * upb) / np.nansum(upb)
            ),
            "rate_feature": rate_feature,
            "train_min": float(train_low),
            "train_max": float(train_high),
            "shocked_min": float(shocked[rate_feature].min()),
            "shocked_max": float(shocked[rate_feature].max()),
            "loans_below_train_min": int(outside.sum()),
            "loans_above_train_max": int(
                (shocked[rate_feature] > train_high).sum()
            ),
            "share_outside_train_range": float(outside.mean()),
        })

    predictions = pd.concat(prediction_rows, ignore_index=True)
    audit = pd.DataFrame(audit_rows).sort_values("shock_bp")

    predictions.to_parquet(
        OUT / "test_dec2024_xgb_rate_shock_feature_audit_predictions.parquet",
        index=False,
    )
    audit.to_csv(
        OUT / "test_dec2024_xgb_rate_shock_feature_support_v2.csv",
        index=False,
    )

    print("Base prediction reproduction passed.")
    print(f"Rows: {len(cohort):,}; max raw diff: {raw_diff:.3g}; "
          f"max isotonic diff: {iso_diff:.3g}")
    print("\nFeature-support audit:")
    print(audit.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()