from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CURVE_PATH = ROOT / "outputs" / "valuation_baseline" / "empirical_s_curve.csv"
OUT = ROOT / "outputs" / "valuation_baseline"
OUT.mkdir(parents=True, exist_ok=True)

POOL = 100_000_000.0
WAC = 0.05
COUPON = 0.045
TERM = 360
BASE_YIELD = 0.045
BASE_INCENTIVE_PP = 0.0
CPR_FLOOR = 0.005
CPR_CAP = 0.90
MIN_ROWS = 1000
SHOCKS = [0.005, 0.01]
REGIMES = {
    "train_2018_2021": "train",
    "test_2023_2024": "test",
    "holdout_2025_2026": "holdout_2025_2026",
}


def parse_bucket(label):
    low, high = [float(x) for x in re.findall(r"-?inf|-?\d+\.?\d*", str(label))[:2]]
    if np.isinf(low):
        mid = high - 0.5
    elif np.isinf(high):
        mid = low + 0.5
    else:
        mid = (low + high) / 2
    return mid


def load_curves():
    if not CURVE_PATH.is_file():
        raise FileNotFoundError(f"Run the S-curve step first: {CURVE_PATH}")
    table = pd.read_csv(CURVE_PATH)
    table["mid"] = table["bucket"].map(parse_bucket)
    curves = {}
    for name, split in REGIMES.items():
        part = table[(table["split"] == split) & (table["rows"] >= MIN_ROWS)]
        part = part.sort_values("mid")
        if len(part) < 4:
            raise ValueError(f"Too few usable buckets for {name}")
        curves[name] = (part["mid"].to_numpy(), part["cpr_upb"].to_numpy())
    return curves


def cpr_from_incentive(incentive_pp, curve):
    xs, ys = curve
    return float(np.clip(np.interp(incentive_pp, xs, ys), CPR_FLOOR, CPR_CAP))


def level_payment(balance, monthly_rate, months_left):
    return balance * monthly_rate / (1 - (1 + monthly_rate) ** (-months_left))


def price_and_wal(cpr, discount_yield):
    balance = POOL
    smm = 1 - (1 - cpr) ** (1 / 12)
    pv = 0.0
    weighted_principal = 0.0
    total_principal = 0.0
    for month in range(1, TERM + 1):
        if balance <= 1e-8:
            break
        payment = level_payment(balance, WAC / 12, TERM - month + 1)
        interest = balance * COUPON / 12
        scheduled = min(balance, max(0.0, payment - balance * WAC / 12))
        prepay = (balance - scheduled) * smm
        principal = scheduled + prepay
        pv += (interest + principal) / (1 + discount_yield / 12) ** month
        weighted_principal += month * principal
        total_principal += principal
        balance -= principal
    return pv / POOL * 100, weighted_principal / total_principal / 12


def main():
    curves = load_curves()
    detail, summary = [], []
    for regime, curve in curves.items():
        for shock in SHOCKS:
            prices, wals = {}, {}
            for label, move in [("down", -shock), ("base", 0.0), ("up", shock)]:
                incentive = BASE_INCENTIVE_PP - move * 100
                cpr = cpr_from_incentive(incentive, curve)
                price, wal = price_and_wal(cpr, BASE_YIELD + move)
                prices[label], wals[label] = price, wal
                detail.append({
                    "regime": regime, "shock_bp": shock * 10000, "case": label,
                    "incentive_pp": incentive, "cpr": cpr, "price": price, "wal_years": wal,
                })
            p0 = prices["base"]
            duration = (prices["down"] - prices["up"]) / (2 * p0 * shock)
            convexity = (prices["down"] + prices["up"] - 2 * p0) / (p0 * shock ** 2)
            summary.append({
                "regime": regime, "shock_bp": shock * 10000,
                "cpr_down": detail[-3]["cpr"], "cpr_base": detail[-2]["cpr"], "cpr_up": detail[-1]["cpr"],
                "price_down": prices["down"], "price_up": prices["up"],
                "wal_down": wals["down"], "wal_base": wals["base"], "wal_up": wals["up"],
                "eff_duration": duration, "convexity_market": convexity / 100,
            })
    summary_df = pd.DataFrame(summary)
    pd.DataFrame(detail).to_csv(OUT / "empirical_scenario_detail.csv", index=False)
    summary_df.to_csv(OUT / "empirical_scenario_summary.csv", index=False)
    print("Assumptions: base incentive", BASE_INCENTIVE_PP, "pp; WAC", WAC, "; coupon", COUPON,
          "; base yield", BASE_YIELD, "; CPR from empirical S-curve; buckets with <", MIN_ROWS, "rows dropped")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()
