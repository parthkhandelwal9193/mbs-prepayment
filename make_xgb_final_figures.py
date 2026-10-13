from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs/model_scenario_eda/test_dec2024_xgb_rate_shock_support_restricted_summary.csv"
OUT = ROOT / "outputs/figures_final"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

full = (
    df.loc[df["sample"].eq("full_cohort")]
    .sort_values("shock_bp")
    .copy()
)
supported = (
    df.loc[df["sample"].eq("in_support_only")]
    .sort_values("shock_bp")
    .copy()
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

axes[0].plot(
    full["shock_bp"],
    100 * full["isotonic_upb_weighted_probability"],
    marker="o",
    linewidth=2.4,
    label="Full December 2024 cohort",
)
axes[0].plot(
    supported["shock_bp"],
    100 * supported["isotonic_upb_weighted_probability"],
    marker="o",
    linewidth=2.4,
    label="In-training-support subset",
)
axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)
axes[0].set_title("XGBoost payoff sensitivity to PMMS shocks")
axes[0].set_xlabel("Parallel market-rate shock (bp)")
axes[0].set_ylabel("UPB-weighted isotonic full-payoff probability (%)")
axes[0].set_xticks([-100, 0, 100])
axes[0].grid(alpha=0.25)
axes[0].legend(frameon=False)

axes[1].plot(
    supported["shock_bp"],
    100 * supported["share_of_loans"],
    marker="o",
    linewidth=2.4,
    label="Loan-count coverage",
)
axes[1].plot(
    supported["shock_bp"],
    100 * supported["share_of_upb"],
    marker="o",
    linewidth=2.4,
    label="UPB coverage",
)
axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Scenario observations within training support")
axes[1].set_xlabel("Parallel market-rate shock (bp)")
axes[1].set_ylabel("Share within 2018–2021 incentive range (%)")
axes[1].set_xticks([-100, 0, 100])
axes[1].set_ylim(0, 100)
axes[1].grid(alpha=0.25)
axes[1].legend(frameon=False)

fig.suptitle(
    "December 2024 XGBoost rate-shock sensitivity and feature support",
    fontsize=13,
    y=1.02,
)
fig.savefig(
    OUT / "fig7_xgb_sensitivity_support.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close(fig)

print(OUT / "fig7_xgb_sensitivity_support.png")
