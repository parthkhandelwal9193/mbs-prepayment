from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "valuation_baseline"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Illustrative pool assumptions—not calibrated to your data or current market.
POOL_BALANCE = 100_000_000.0
WAC = 0.05
PASS_THROUGH_COUPON = 0.045
TERM_MONTHS = 360
BASE_CPR = 0.08
BASE_DISCOUNT_YIELD = 0.045
RATE_SHOCK = 0.01  # 100 bp

# Illustrative prepayment response: CPR moves 15 percentage points per 100 bp.
# We will replace this assumed response only after the cash-flow engine is checked.
def scenario_cpr(rate_shock):
    return float(np.clip(BASE_CPR - 15.0 * rate_shock, 0.005, 0.60))


def monthly_payment(balance, monthly_rate, months_left):
    if months_left <= 0:
        return balance
    if abs(monthly_rate) < 1e-12:
        return balance / months_left
    return balance * monthly_rate / (
        1 - (1 + monthly_rate) ** (-months_left)
    )


def project_cashflows(cpr):
    balance = POOL_BALANCE
    smm = 1 - (1 - cpr) ** (1 / 12)
    rows = []

    for month in range(1, TERM_MONTHS + 1):
        if balance <= 1e-8:
            break

        months_left = TERM_MONTHS - month + 1
        scheduled_payment = monthly_payment(
            balance, WAC / 12, months_left
        )
        interest = balance * PASS_THROUGH_COUPON / 12
        scheduled_principal = min(
            balance,
            max(0.0, scheduled_payment - balance * WAC / 12),
        )

        balance_after_scheduled = max(
            0.0, balance - scheduled_principal
        )
        prepayment = min(
            balance_after_scheduled,
            balance_after_scheduled * smm,
        )
        principal = scheduled_principal + prepayment
        total_cashflow = interest + principal
        ending_balance = max(0.0, balance - principal)

        rows.append({
            "month": month,
            "begin_balance": balance,
            "interest": interest,
            "scheduled_principal": scheduled_principal,
            "prepayment": prepayment,
            "total_principal": principal,
            "total_cashflow": total_cashflow,
            "end_balance": ending_balance,
        })
        balance = ending_balance

    return pd.DataFrame(rows)


def price_and_wal(cpr, discount_yield):
    flows = project_cashflows(cpr)
    discount_factors = (
        1 + discount_yield / 12
    ) ** flows["month"]

    price_per_100 = (
        flows["total_cashflow"] / discount_factors
    ).sum() / POOL_BALANCE * 100

    wal_years = (
        (flows["month"] * flows["total_principal"]).sum()
        / flows["total_principal"].sum()
        / 12
    )

    return price_per_100, wal_years, flows


def main():
    scenarios = [
        ("rates_down_100bp", -RATE_SHOCK),
        ("base", 0.0),
        ("rates_up_100bp", RATE_SHOCK),
    ]

    results = []
    monthly_cashflows = []

    for scenario_name, rate_shock in scenarios:
        cpr = scenario_cpr(rate_shock)
        discount_yield = BASE_DISCOUNT_YIELD + rate_shock

        price, wal, flows = price_and_wal(cpr, discount_yield)

        results.append({
            "scenario": scenario_name,
            "rate_shock_bp": rate_shock * 10000,
            "assumed_cpr": cpr,
            "implied_smm": 1 - (1 - cpr) ** (1 / 12),
            "discount_yield": discount_yield,
            "price_per_100": price,
            "wal_years": wal,
        })

        flows.insert(0, "scenario", scenario_name)
        monthly_cashflows.append(flows)

    summary = pd.DataFrame(results).set_index("scenario")

    base_price = summary.loc["base", "price_per_100"]
    down_price = summary.loc["rates_down_100bp", "price_per_100"]
    up_price = summary.loc["rates_up_100bp", "price_per_100"]

    effective_duration = (
        down_price - up_price
    ) / (2 * base_price * RATE_SHOCK)

    effective_convexity = (
        down_price + up_price - 2 * base_price
    ) / (base_price * RATE_SHOCK ** 2)

    summary["base_effective_duration_years"] = effective_duration
    summary["base_effective_convexity"] = effective_convexity

    summary.reset_index().to_csv(
        OUTPUT / "illustrative_scenario_summary.csv",
        index=False,
    )
    pd.concat(monthly_cashflows, ignore_index=True).to_csv(
        OUTPUT / "illustrative_monthly_cashflows.csv",
        index=False,
    )

    print("Illustrative assumptions:")
    print({
        "pool_balance": POOL_BALANCE,
        "WAC": WAC,
        "pass_through_coupon": PASS_THROUGH_COUPON,
        "term_months": TERM_MONTHS,
        "base_CPR": BASE_CPR,
        "base_discount_yield": BASE_DISCOUNT_YIELD,
        "CPR_change": "15 percentage points per +/-100 bp, assumed",
    })
    print("\nScenario results:")
    print(
        summary.reset_index().to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("\nWrote illustrative outputs to:", OUTPUT)
    print("These are illustrative results, not market-calibrated valuations.")


if __name__ == "__main__":
    main()