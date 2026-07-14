"""
Spark feature-engineering job.

Locally runs on local[*]; in production this exact script is submitted to
AWS EMR via `infra/emr/submit_spark_job.sh` and processes the full 50M+
monthly customer records from S3, writing partitioned Parquet features back
to S3 for training and batch scoring.
"""
from __future__ import annotations

import argparse

import yaml
from pyspark.sql import SparkSession, functions as F


def build_spark(cfg: dict) -> SparkSession:
    s = cfg["spark"]
    return (
        SparkSession.builder.appName(s["app_name"])
        .master(s.get("master", "local[*]"))
        .config("spark.sql.shuffle.partitions", s["shuffle_partitions"])
        .config("spark.driver.memory", s.get("driver_memory", "4g"))
        .getOrCreate()
    )


def engineer_features(df):
    """Feature engineering: trends, ratios, interactions, risk flags."""
    df = (
        df
        # usage trend: negative slope across last 3 months = disengagement
        .withColumn("usage_trend_pct",
                    F.when(F.col("data_gb_m3") > 0,
                           (F.col("data_gb_m1") - F.col("data_gb_m3")) / F.col("data_gb_m3"))
                     .otherwise(F.lit(0.0)))
        .withColumn("avg_data_gb", (F.col("data_gb_m1") + F.col("data_gb_m2") + F.col("data_gb_m3")) / 3)
        .withColumn("charge_per_line", F.col("monthly_charges") / F.col("num_lines"))
        .withColumn("charges_per_tenure", F.col("total_charges") / F.greatest(F.col("tenure_months"), F.lit(1)))
        .withColumn("tickets_per_tenure",
                    F.col("support_tickets_90d") / F.greatest(F.col("tenure_months"), F.lit(1)))
        .withColumn("is_new_customer", (F.col("tenure_months") <= 6).cast("int"))
        .withColumn("high_value_flag", (F.col("monthly_charges") > 100).cast("int"))
        .withColumn("network_pain_score", F.col("call_drop_rate") * 100 + F.col("support_tickets_90d"))
        .withColumn("payment_risk",
                    ((F.col("payment_method") == "electronic_check") & (F.col("autopay_enrolled") == 0)).cast("int"))
        .withColumn("engagement_score",
                    F.lit(365.0) / (F.col("days_since_last_interaction") + F.lit(30.0)))
    )

    # categorical one-hot (kept explicit for scoring-time schema stability)
    for col, values in {
        "contract_type": ["month-to-month", "one-year", "two-year"],
        "payment_method": ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
        "internet_service": ["dsl", "fiber", "none"],
        "customer_segment": ["consumer", "small_business", "enterprise"],
    }.items():
        for v in values:
            safe = v.replace("-", "_")
            df = df.withColumn(f"{col}__{safe}", (F.col(col) == v).cast("int"))
    return df.drop("contract_type", "payment_method", "internet_service", "customer_segment")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    inp = args.input or cfg["data"]["raw_path"]
    out = args.output or cfg["data"]["features_path"]

    spark = build_spark(cfg)
    raw = spark.read.parquet(inp)
    feats = engineer_features(raw)
    feats.write.mode("overwrite").parquet(out)

    n = feats.count()
    print(f"[spark_etl] wrote {n:,} rows x {len(feats.columns)} cols -> {out}")
    spark.stop()


if __name__ == "__main__":
    main()
