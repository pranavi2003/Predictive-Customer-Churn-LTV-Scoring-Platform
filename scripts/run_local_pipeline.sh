#!/bin/bash
# One-command end-to-end local run: data -> spark features -> train -> score
set -euo pipefail
export PYTHONPATH=src
echo "=== 1/4 Generating raw data ==="
python -m churn_platform.data_generation --n 200000 --out data/raw/customers.parquet
echo "=== 2/4 Spark feature engineering ==="
python -m churn_platform.spark_etl --config config/config.yaml
echo "=== 3/4 Training stacked ensemble + LTV model ==="
python -m churn_platform.train --config config/config.yaml
echo "=== 4/4 Batch scoring population ==="
python -m churn_platform.score --config config/config.yaml
echo "=== DONE — see reports/metrics.json and data/processed/scored_customers.parquet ==="
