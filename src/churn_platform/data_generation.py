"""
Synthetic telecom customer data generator.

Simulates the raw monthly customer feed (billing, usage, network quality,
support interactions) that in production arrives from the client's data
warehouse into S3 and is processed by Spark on EMR. Churn is generated from
a latent risk function of behavioral drivers so models can learn a real
signal, and LTV is derived from expected remaining tenure x monthly margin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ["month-to-month", "one-year", "two-year"]
PAYMENT_METHODS = ["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
INTERNET = ["dsl", "fiber", "none"]
SEGMENTS = ["consumer", "small_business", "enterprise"]


def generate_customers(n: int = 200_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, n)                                   # months
    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(PAYMENT_METHODS, n, p=[0.35, 0.15, 0.25, 0.25])
    internet = rng.choice(INTERNET, n, p=[0.35, 0.45, 0.20])
    segment = rng.choice(SEGMENTS, n, p=[0.75, 0.18, 0.07])
    autopay = rng.random(n) < 0.45
    paperless = rng.random(n) < 0.6
    num_lines = rng.integers(1, 6, n)
    intl_plan = rng.random(n) < 0.12
    streaming = rng.random(n) < 0.5

    base_charge = np.where(internet == "fiber", 75, np.where(internet == "dsl", 50, 25))
    monthly_charges = (
        base_charge + num_lines * 8 + streaming * 12 + intl_plan * 15
        + rng.normal(0, 6, n)
    ).clip(18, 190).round(2)
    total_charges = (monthly_charges * tenure * rng.uniform(0.92, 1.02, n)).round(2)

    # usage & network quality (last 3 months)
    data_gb_m1 = rng.gamma(3.0, 4.0, n)
    usage_trend = rng.normal(0, 0.25, n)                               # +growing, -declining
    data_gb_m2 = (data_gb_m1 * (1 - usage_trend * 0.5)).clip(0)
    data_gb_m3 = (data_gb_m1 * (1 - usage_trend)).clip(0)
    call_drop_rate = rng.beta(1.5, 40, n) + np.where(internet == "fiber", 0, 0.005)
    avg_download_mbps = np.where(internet == "fiber", rng.normal(280, 60, n),
                        np.where(internet == "dsl", rng.normal(45, 15, n), 0)).clip(0)

    support_tickets_90d = rng.poisson(0.6, n)
    late_payments_12m = rng.poisson(0.4, n)
    days_since_last_interaction = rng.integers(0, 365, n)
    price_increase_flag = rng.random(n) < 0.2

    # ---- latent churn risk ----------------------------------------------
    contract_risk = np.select(
        [contract == "month-to-month", contract == "one-year"], [2.3, 0.3], default=-1.2
    )
    z = (
        -2.9
        + contract_risk
        - 0.055 * tenure
        + 1.3 * (payment == "electronic_check")
        - 1.0 * autopay
        + 0.8 * support_tickets_90d
        + 0.7 * late_payments_12m
        + 32.0 * call_drop_rate
        + 2.2 * np.maximum(usage_trend, 0)          # declining usage = flight risk
        + 1.2 * price_increase_flag
        + 0.018 * (monthly_charges - monthly_charges.mean())
        - 0.6 * (segment == "enterprise")
        + rng.normal(0, 0.18, n)                    # irreducible noise
    )
    churn_prob = 1 / (1 + np.exp(-1.18 * z))       # scale sharpens class separation
    churned = (rng.random(n) < churn_prob).astype(int)

    # ---- LTV: expected remaining margin ----------------------------------
    margin_rate = 0.42
    expected_remaining_months = np.where(churned == 1, rng.uniform(0.5, 4, n),
                                         (1 - churn_prob) * rng.uniform(18, 42, n))
    ltv_actual = (monthly_charges * margin_rate * expected_remaining_months).round(2)

    df = pd.DataFrame({
        "customer_id": [f"CUST{100000000 + i}" for i in range(n)],
        "tenure_months": tenure,
        "contract_type": contract,
        "payment_method": payment,
        "internet_service": internet,
        "customer_segment": segment,
        "autopay_enrolled": autopay.astype(int),
        "paperless_billing": paperless.astype(int),
        "num_lines": num_lines,
        "intl_plan": intl_plan.astype(int),
        "streaming_bundle": streaming.astype(int),
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "data_gb_m1": data_gb_m1.round(2),
        "data_gb_m2": data_gb_m2.round(2),
        "data_gb_m3": data_gb_m3.round(2),
        "call_drop_rate": call_drop_rate.round(4),
        "avg_download_mbps": avg_download_mbps.round(1),
        "support_tickets_90d": support_tickets_90d,
        "late_payments_12m": late_payments_12m,
        "days_since_last_interaction": days_since_last_interaction,
        "price_increase_flag": price_increase_flag.astype(int),
        "churned": churned,
        "ltv_actual": ltv_actual,
    })
    return df


if __name__ == "__main__":
    import argparse, pathlib

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/raw/customers.parquet")
    args = p.parse_args()

    df = generate_customers(args.n, args.seed)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df):,} customers -> {args.out} | churn rate={df.churned.mean():.3f}")
