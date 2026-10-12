from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = P / "dev_sample.parquet"

SAMPLE_FRACTION = 0.30
SEED = 42


def main():
    df = pd.read_parquet(P / "panel_with_rates.parquet")

    ids = df["loan_sequence_number"].drop_duplicates()
    sampled_ids = ids.sample(
        frac=SAMPLE_FRACTION,
        random_state=SEED,
    )

    sample = df[df["loan_sequence_number"].isin(sampled_ids)].copy()

    print("Development sample rows:", f"{len(sample):,}")
    print("Development sample loans:",
          f"{sample['loan_sequence_number'].nunique():,}")
    print("Row fraction:", round(len(sample) / len(df), 4))
    print("Observed date range:",
          sample["monthly_reporting_period"].min(),
          "to",
          sample["monthly_reporting_period"].max())

    out = P / "dev_sample.parquet"
    sample.to_parquet(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()