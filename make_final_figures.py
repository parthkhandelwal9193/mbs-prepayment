from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import valuation_empirical as ve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "figures_final"
OUT.mkdir(parents=True, exist_ok=True)

REGIMES = {
    "train_2018_2021": ("2018–21 behaviour (train)", "#1f77b4"),
    "test_2023_2024": ("2023–24 behaviour (rate lock-in)", "#d62728"),
    "holdout_2025_2026": ("2025–26 behaviour (holdout)", "#2ca02c"),
}
SHOCKS_BP = np.arange(-150, 151, 25)


def load_all_s_curves():
    table = pd.read_csv(ve.CURVE_PATH)
    table["mid"] = table["bucket"].map(ve.parse_bucket)
    return table[table["rows"] >= ve.MIN_ROWS].copy()


def price_wal_at(curve, shock_bp):
    shock = shock_bp / 10000
    incentive = ve.BASE_INCENTIVE_PP - shock * 100
    cpr = ve.cpr_from_incentive(incentive, curve)
    price, wal = ve.price_and_wal(cpr, ve.BASE_YIELD + shock)
    return cpr, price, wal


def main():
    curves = ve.load_curves()
    s_curve = load_all_s_curves()

    names = {
        "train": "2018–21 (train)",
        "validation": "2022 (validation)",
        "test": "2023–24 (test)",
        "holdout_2025_2026": "2025–26 (holdout)",
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for split, label in names.items():
        z = s_curve[s_curve["split"] == split].sort_values("mid")
        ax.plot(z["mid"], z["cpr_upb"] * 100, marker="o", linewidth=2, label=label)
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set(xlabel="Refinance incentive (note rate minus market rate, percentage points)",
           ylabel="Annualised CPR (%)",
           title="Observed prepayment S-curves by period")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(0.01, 0.005, "Balance-weighted full payoffs; incentive buckets with fewer than "
             f"{ve.MIN_ROWS:,} loan-months are omitted.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig1_s_curves.png", dpi=180)
    plt.close(fig)

    records = []
    for regime, curve in curves.items():
        for shock_bp in SHOCKS_BP:
            cpr, price, wal = price_wal_at(curve, int(shock_bp))
            records.append({"regime": regime, "shock_bp": int(shock_bp), "cpr": cpr,
                            "price": price, "wal_years": wal})
    grid = pd.DataFrame(records)

    base_cpr = grid[(grid.regime == "test_2023_2024") & (grid.shock_bp == 0)]["cpr"].iloc[0]
    fixed = []
    for shock_bp in SHOCKS_BP:
        price, wal = ve.price_and_wal(base_cpr, ve.BASE_YIELD + shock_bp / 10000)
        fixed.append({"shock_bp": int(shock_bp), "price": price, "wal_years": wal})
    fixed = pd.DataFrame(fixed)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for regime, (label, color) in REGIMES.items():
        g = grid[grid.regime == regime]
        ax.plot(g["shock_bp"], g["price"], marker="o", linewidth=2, color=color, label=label)
    ax.plot(fixed["shock_bp"], fixed["price"], linestyle="--", color="black",
            label=f"No prepayment response (CPR fixed at {base_cpr:.1%})")
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set(xlabel="Parallel change in rates (bp; negative = rates fall)", ylabel="Price per 100",
           title="Price vs rate change: upside is capped when prepayments accelerate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_price_vs_rate_shock.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for regime, (label, color) in REGIMES.items():
        g = grid[grid.regime == regime]
        ax.plot(g["shock_bp"], g["wal_years"], marker="o", linewidth=2, color=color, label=label)
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set(xlabel="Parallel change in rates (bp; negative = rates fall)",
           ylabel="Weighted average life (years)",
           title="WAL shortens when rates fall and extends when rates rise")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_wal_vs_rate_shock.png", dpi=180)
    plt.close(fig)

    rows = []
    for regime, curve in curves.items():
        _, p_down, _ = price_wal_at(curve, -100)
        _, p0, _ = price_wal_at(curve, 0)
        _, p_up, _ = price_wal_at(curve, 100)
        dy = 0.01
        rows.append({"regime": regime,
                     "effective_duration": (p_down - p_up) / (2 * p0 * dy),
                     "convexity_market": (p_down + p_up - 2 * p0) / (p0 * dy ** 2) / 100})
    risk = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    labels = [REGIMES[r][0].replace(" behaviour", "").replace(" (", "\n(") for r in risk["regime"]]
    colors = [REGIMES[r][1] for r in risk["regime"]]
    axes[0].bar(labels, risk["effective_duration"], color=colors)
    axes[0].set(ylabel="Years", title="Effective duration (±100 bp)")
    axes[1].bar(labels, risk["convexity_market"], color=colors)
    axes[1].set(ylabel="Convexity (market convention)", title="Effective convexity (±100 bp)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_duration_convexity.png", dpi=180)
    plt.close(fig)

    grid.to_csv(OUT / "price_wal_grid.csv", index=False)
    risk.to_csv(OUT / "duration_convexity_100bp.csv", index=False)
    print("Created:")
    for p in sorted(OUT.glob("*")):
        print(" ", p.relative_to(ROOT))
    print("\nDuration/convexity (+/-100 bp):")
    print(risk.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
