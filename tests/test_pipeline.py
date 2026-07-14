"""Smoke tests for the churn platform (pytest)."""
import numpy as np
import pandas as pd
import pytest

from churn_platform.data_generation import generate_customers
from churn_platform.monitoring import psi


def test_data_generation_shape_and_signal():
    df = generate_customers(n=5000, seed=7)
    assert len(df) == 5000
    assert df["customer_id"].is_unique
    assert 0.05 < df["churned"].mean() < 0.6
    # month-to-month should churn more than two-year contracts (sanity of signal)
    m2m = df[df.contract_type == "month-to-month"].churned.mean()
    two_yr = df[df.contract_type == "two-year"].churned.mean()
    assert m2m > two_yr


def test_psi_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 10_000)
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 10_000)
    b = rng.normal(1.5, 1, 10_000)
    assert psi(a, b) > 0.2


def test_feature_engineering_spark():
    pyspark = pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    from churn_platform.spark_etl import engineer_features

    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    pdf = generate_customers(n=500, seed=1)
    out = engineer_features(spark.createDataFrame(pdf)).toPandas()
    assert "usage_trend_pct" in out.columns
    assert "contract_type__month_to_month" in out.columns
    assert "contract_type" not in out.columns
    assert out["charge_per_line"].notna().all()
    spark.stop()
