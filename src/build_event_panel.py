from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "panel_events.parquet"

# Provisional event definition; keep other termination codes separate.
VOLUNTARY_PREPAY_CODES = {"01"}
OTHER_EXIT_CODES = {"02", "03", "09", "15", "16", "96"}


def main():
    orig = pd.read_parquet(PROCESSED / "orig_all.parquet")
    perf = pd.read_parquet(PROCESSED / "perf_all.parquet")

    for df in (orig, perf):
        df["loan_sequence_number"] = (
            df["loan_sequence_number"].astype("string").str.strip()
        )

    perf["zero_balance_code"] = (
        perf["zero_balance_code"].astype("string").str.strip()
    )

    perf_ids = set(perf["loan_sequence_number"].dropna().unique())
    no_history = int((~orig["loan_sequence_number"].isin(perf_ids)).sum())
    print("Origination loans without performance rows:", no_history)

    panel = perf.merge(
        orig,
        on="loan_sequence_number",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched_rows = int((panel["_merge"] != "both").sum())
    print("Performance rows unmatched to origination:", unmatched_rows)
    if unmatched_rows:
        raise ValueError("Found performance rows without origination data")

    panel = panel.drop(columns="_merge")
    panel = panel.sort_values(
        ["loan_sequence_number", "monthly_reporting_period"]
    ).reset_index(drop=True)

    panel["terminal_code"] = panel["zero_balance_code"].fillna("").str.strip()
    panel["prepaid"] = panel["terminal_code"].isin(VOLUNTARY_PREPAY_CODES)
    panel["other_exit"] = panel["terminal_code"].isin(OTHER_EXIT_CODES)
    panel["terminal_event"] = panel["prepaid"] | panel["other_exit"]

    event_seen_before = panel.groupby("loan_sequence_number")[
        "terminal_event"
    ].transform(lambda s: s.cumsum().shift(fill_value=0) > 0)

    rows_after_event = int(event_seen_before.sum())
    print("Rows after a terminal event:", rows_after_event)
    if rows_after_event:
        raise ValueError("Servicing records appear after a terminal event")

    # Right-censoring: the loan remained in the panel through the last available month.
    panel["censored_at_data_end"] = ~panel.groupby(
        "loan_sequence_number"
    )["terminal_event"].transform("any")

    panel["observation_year"] = (
        panel["monthly_reporting_period"].dt.year.astype("int16")
    )
    panel["observation_month"] = (
        panel["monthly_reporting_period"].dt.month.astype("int8")
    )

    panel.to_parquet(OUT, index=False)

    print("Panel rows:", f"{len(panel):,}")
    print("Unique loans:", f"{panel['loan_sequence_number'].nunique():,}")
    print("Loans with code 01:",
          f"{panel.loc[panel['prepaid'], 'loan_sequence_number'].nunique():,}")
    print("Other-exit loans by code:")
    print(
        panel.loc[panel["other_exit"]]
        .groupby("terminal_code")["loan_sequence_number"]
        .nunique()
        .to_string()
    )
    print("Censored loans:",
          f"{panel.loc[panel['censored_at_data_end'], 'loan_sequence_number'].nunique():,}")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()