from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRED = Path(
    "outputs/predictions/"
    "xgb_lagged_pmms_regime_fixed_predictions.parquet"
)
OUT = Path("outputs/reports")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PRED)
needed = {
    "split", "event", "raw_prediction", "isotonic_prediction",
    "monthly_reporting_period",
}
missing = needed - set(df.columns)
if missing:
    raise SystemExit(f"Prediction file lacks columns: {sorted(missing)}")

df["monthly_reporting_period"] = pd.to_datetime(df["monthly_reporting_period"])
df["event"] = pd.to_numeric(df["event"], errors="raise").astype(int)

# Equal-count deciles for test; raw and calibrated predictions on identical rows.
test = df.loc[df["split"].eq("test")].copy()
if test.empty:
    raise SystemExit("No rows labelled split='test' in predictions.")

test["decile"] = pd.qcut(
    test["isotonic_prediction"].rank(method="first"),
    q=10,
    labels=False,
) + 1

tables = []
for col, kind in [
    ("raw_prediction", "raw"),
    ("isotonic_prediction", "isotonic"),
]:
    t = (
        test.groupby("decile")
        .agg(
            rows=("event", "size"),
            events=("event", "sum"),
            mean_prediction=(col, "mean"),
            observed_event_rate=("event", "mean"),
        )
        .reset_index()
    )
    t["probability_type"] = kind
    tables.append(t)

deciles = pd.concat(tables, ignore_index=True)
deciles["mean_prediction_pct"] = 100 * deciles["mean_prediction"]
deciles["observed_event_rate_pct"] = 100 * deciles["observed_event_rate"]
deciles.to_csv(OUT / "xgb_test_actual_vs_predicted_deciles.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
colors = {"raw": "#d95f02", "isotonic": "#1b75bb"}

for kind in ["raw", "isotonic"]:
    t = deciles.loc[deciles["probability_type"].eq(kind)].sort_values("decile")
    x = t["mean_prediction_pct"].to_numpy()
    y = t["observed_event_rate_pct"].to_numpy()
    sizes = 25 + 100 * np.sqrt(t["rows"].to_numpy() / t["rows"].max())
    ax.plot(x, y, marker="o", linewidth=1.8, color=colors[kind], label=kind.title())
    ax.scatter(x, y, s=sizes, color=colors[kind], alpha=0.75, zorder=3)

upper = 1.6 * max(
    deciles["mean_prediction_pct"].max(),
    deciles["observed_event_rate_pct"].max(),
)
ax.plot([0, upper], [0, upper], "--", color="gray", label="Perfect calibration")
ax.set_xlim(0, upper)
ax.set_ylim(0, upper)
ax.set_xlabel("Mean predicted event probability (%)")
ax.set_ylabel("Observed event rate (%)")
ax.set_title("Test-set calibration by predicted-risk decile")
ax.grid(alpha=0.25)
ax.legend(title="Prediction")
fig.text(
    0.5, 0.01,
    "Each point is one equal-count decile; point size is proportional to √(rows). "
    "Test period: 2023–2024.",
    ha="center", fontsize=8,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "xgb_test_calibration.png", dpi=180)
plt.close(fig)

print(deciles[
    ["probability_type", "decile", "rows", "events",
     "mean_prediction_pct", "observed_event_rate_pct"]
].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved diagnostics to {OUT}")
