from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

# Run from the project root with: python src/train_xgboost_hazard.py
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "hazard_dev_sample.parquet"
MODEL_DIR = ROOT / "outputs" / "models"
PRED_DIR = ROOT / "outputs" / "predictions"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

NUMERIC = [
    "refi_incentive_pp", "loan_age", "current_interest_rate",
    "remaining_months_to_legal_maturity", "original_upb", "original_ltv",
    "original_cltv", "original_dti", "credit_score", "original_loan_term",
]
CATEGORICAL = ["loan_purpose", "occupancy_status", "modification_flag"]
SENTINELS = {"original_ltv": 999, "original_cltv": 999,
             "original_dti": 999, "credit_score": 9999}
SPLITS = ["train", "validation", "test", "holdout_2025_2026"]
PMMS_CHANGE_DATE = pd.Timestamp("2022-11-17")


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


def print_metrics(label, y, probabilities):
    print(
        f"{label}: n={len(y):,}, events={int(y.sum()):,}, "
        f"rate={y.mean():.5%}, mean_pred={probabilities.mean():.5%}, "
        f"ROC_AUC={roc_auc_score(y, probabilities):.4f}, "
        f"PR_AUC={average_precision_score(y, probabilities):.5f}, "
        f"Brier={brier_score_loss(y, probabilities):.6f}, "
        f"LogLoss={log_loss(y, probabilities, labels=[0, 1]):.6f}"
    )
    deciles = pd.qcut(pd.Series(probabilities).rank(method="first"), 10, labels=False) + 1
    table = pd.DataFrame({"event": y, "prediction": probabilities, "decile": deciles})
    table = table.groupby("decile", observed=True).agg(
        rows=("event", "size"), mean_prediction=("prediction", "mean"),
        observed_rate=("event", "mean"),
    )
    print(table.to_string(float_format=lambda value: f"{value:.6f}"))


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT}")

    required = {
        "loan_sequence_number", "monthly_reporting_period", "split", "event",
        *NUMERIC, *CATEGORICAL,
    }
    available = set(pq.ParquetFile(INPUT).schema.names)
    missing = required - available
    if missing:
        raise ValueError(f"Required columns missing from input: {sorted(missing)}")

    df = pd.read_parquet(INPUT, columns=sorted(required))
    df["monthly_reporting_period"] = pd.to_datetime(
        df["monthly_reporting_period"], errors="coerce"
    )
    df["event"] = pd.to_numeric(df["event"], errors="raise").astype("int8")
    if not df["event"].isin([0, 1]).all():
        raise ValueError("event must contain only 0/1 values")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, sentinel in SENTINELS.items():
        df.loc[df[col] == sentinel, col] = np.nan

    df = df.sort_values(["loan_sequence_number", "monthly_reporting_period"]).reset_index(drop=True)
    loan_id = df["loan_sequence_number"]
    dates = df["monthly_reporting_period"]
    month_id = dates.dt.year * 12 + dates.dt.month
    consecutive_month = month_id.sub(month_id.groupby(loan_id, sort=False).shift(1)).eq(1)

    # Use strictly prior-calendar-month loan data; don't bridge missing months.
    for col in NUMERIC:
        lag_col = f"{col}_lag1"
        df[lag_col] = df.groupby("loan_sequence_number", sort=False)[col].shift(1)
        df.loc[~consecutive_month, lag_col] = np.nan

    incentive = df["refi_incentive_pp"]
    in_money = (incentive > 0.5).astype("int8")
    df["months_in_money_prior"] = (
        in_money.groupby(loan_id, sort=False).cumsum() - in_money
    )
    df["max_incentive_prior"] = incentive.groupby(loan_id, sort=False).transform(
        lambda series: series.cummax().shift(1)
    )
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
        "months_in_money_prior", "max_incentive_prior", "incentive_lag1_prior",
        "observation_year", "month_sin", "month_cos", "post_pmms_change",
    ]
    engineered += [
        col for col in df.columns
        if col.startswith("incentive_hinge_prior_") or col.startswith("age_hinge_prior_")
    ]
    features = lag_features + engineered + CATEGORICAL

    df = df[df["split"].isin(SPLITS)].copy()
    parts = {split: df.loc[df["split"] == split].copy() for split in SPLITS}
    required_lags = [
        "refi_incentive_pp_lag1", "loan_age_lag1", "current_interest_rate_lag1",
        "remaining_months_to_legal_maturity_lag1",
    ]
    for split in SPLITS:
        parts[split] = parts[split].loc[parts[split][required_lags].notna().all(axis=1)].copy()
        if parts[split].empty:
            raise ValueError(f"No usable rows remain in split {split}")
    print("Rows after exact prior-month filter:", {s: len(parts[s]) for s in SPLITS})

    def encode(frame):
        return pd.get_dummies(
            frame[features], columns=CATEGORICAL, dummy_na=True, dtype=np.uint8
        )

    matrices = {split: encode(parts[split]) for split in SPLITS}
    all_columns = sorted(set().union(*(set(matrix.columns) for matrix in matrices.values())))
    encoded_names = safe_feature_names(all_columns)
    for split, matrix in matrices.items():
        matrix = matrix.reindex(columns=all_columns, fill_value=0)
        matrix.columns = encoded_names
        matrices[split] = matrix

    # Tune only within train (2021); validation is reserved for calibration.
    train = parts["train"]
    tuning_mask = train["monthly_reporting_period"].dt.year.eq(2021)
    fit_rows, tune_rows = train.loc[~tuning_mask], train.loc[tuning_mask]
    if fit_rows.empty or tune_rows.empty:
        raise ValueError("Training split must include rows both before and during 2021")

    configs = [
        dict(max_depth=2, min_child_weight=10, subsample=.8, colsample_bytree=.8,
             reg_lambda=10., reg_alpha=.1, gamma=0.),
        dict(max_depth=3, min_child_weight=10, subsample=.8, colsample_bytree=.8,
             reg_lambda=10., reg_alpha=.1, gamma=0.),
        dict(max_depth=4, min_child_weight=25, subsample=.8, colsample_bytree=.8,
             reg_lambda=10., reg_alpha=.1, gamma=0.),
        dict(max_depth=3, min_child_weight=20, subsample=.9, colsample_bytree=.9,
             reg_lambda=20., reg_alpha=.1, gamma=0.),
    ]
    best = None
    for number, params in enumerate(configs, start=1):
        candidate = xgb.XGBClassifier(
            objective="binary:logistic", tree_method="hist", n_estimators=1800,
            learning_rate=.03, max_bin=128, eval_metric="logloss",
            early_stopping_rounds=100, n_jobs=2, random_state=SEED,
            scale_pos_weight=1., **params,
        )
        candidate.fit(
            matrices["train"].loc[fit_rows.index], fit_rows["event"].to_numpy(),
            eval_set=[(matrices["train"].loc[tune_rows.index], tune_rows["event"].to_numpy())],
            verbose=False,
        )
        tune_pred = candidate.predict_proba(matrices["train"].loc[tune_rows.index])[:, 1]
        tune_logloss = log_loss(tune_rows["event"], tune_pred, labels=[0, 1])
        print(
            f"Trial {number}: depth={params['max_depth']} "
            f"best_iter={candidate.best_iteration} 2021_logloss={tune_logloss:.6f} "
            f"AUC={roc_auc_score(tune_rows['event'], tune_pred):.4f}"
        )
        if best is None or tune_logloss < best["logloss"]:
            best = {"params": params, "iteration": candidate.best_iteration,
                    "logloss": tune_logloss}

    model = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", n_estimators=best["iteration"] + 1,
        learning_rate=.03, max_bin=128, n_jobs=2, random_state=SEED,
        scale_pos_weight=1., **best["params"],
    )
    model.fit(matrices["train"], train["event"].to_numpy(), verbose=False)

    # Calibrate probabilities on validation; calibration won't improve ROC-AUC.
    validation_raw = model.predict_proba(matrices["validation"])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    calibrator.fit(validation_raw, parts["validation"]["event"].to_numpy())
    print("Selected:", best["params"], "trees:", best["iteration"] + 1)

    metric_rows, prediction_frames = [], []
    for split in SPLITS:
        part = parts[split]
        raw = model.predict_proba(matrices[split])[:, 1]
        calibrated = calibrator.predict(raw)
        y = part["event"].to_numpy()
        print_metrics(f"{split} raw", y, raw)
        print_metrics(f"{split} calibrated", y, calibrated)
        for kind, pred in [("raw", raw), ("isotonic", calibrated)]:
            metric_rows.append({
                "split": split, "type": kind, "rows": len(y), "events": int(y.sum()),
                "rate": y.mean(), "mean_pred": pred.mean(),
                "roc_auc": roc_auc_score(y, pred),
                "pr_auc": average_precision_score(y, pred),
                "brier": brier_score_loss(y, pred),
                "log_loss": log_loss(y, pred, labels=[0, 1]),
            })
        prediction_frames.append(pd.DataFrame({
            "loan_sequence_number": part["loan_sequence_number"].to_numpy(),
            "monthly_reporting_period": part["monthly_reporting_period"].to_numpy(),
            "split": split, "event": y, "raw_prediction": raw,
            "isotonic_prediction": calibrated,
        }))

    # Save outputs only after all evaluation completes; all paths are defined above.
    model.save_model(str(MODEL_DIR / "xgboost_lagged_pmms_regime_fixed.json"))
    joblib.dump({
        "features": features, "encoded_columns": encoded_names,
        "params": best["params"], "iteration": best["iteration"],
        "calibrator": calibrator, "selection_year": 2021, "calibration_year": 2022,
    }, MODEL_DIR / "xgboost_lagged_pmms_regime_fixed_metadata.joblib")
    pd.DataFrame(metric_rows).to_csv(
        MODEL_DIR / "xgboost_lagged_pmms_regime_fixed_metrics.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        PRED_DIR / "xgb_lagged_pmms_regime_fixed_predictions.parquet", index=False
    )
    print("Saved model, metadata, metrics, and predictions successfully.")
    print(f"Model: {MODEL_DIR / 'xgboost_lagged_pmms_regime_fixed.json'}")
    print(f"Metrics: {MODEL_DIR / 'xgboost_lagged_pmms_regime_fixed_metrics.csv'}")
    print(f"Predictions: {PRED_DIR / 'xgb_lagged_pmms_regime_fixed_predictions.parquet'}")


if __name__ == "__main__":
    main()
