#!/bin/bash
# Submit the feature-engineering job to EMR (production: 50M+ records).
# Usage: bash infra/emr/submit_spark_job.sh <cluster-id>
set -euo pipefail
CLUSTER_ID=${1:?Provide an EMR cluster id}
BUCKET=s3://telecom-churn-platform

aws s3 cp src/churn_platform/spark_etl.py $BUCKET/code/spark_etl.py
aws s3 cp config/config.yaml $BUCKET/code/config.yaml

aws emr add-steps --cluster-id "$CLUSTER_ID" --steps "[{
  \"Type\": \"Spark\",
  \"Name\": \"churn-feature-engineering\",
  \"ActionOnFailure\": \"CONTINUE\",
  \"Args\": [
    \"--deploy-mode\", \"cluster\",
    \"--conf\", \"spark.sql.shuffle.partitions=2000\",
    \"--conf\", \"spark.dynamicAllocation.enabled=true\",
    \"$BUCKET/code/spark_etl.py\",
    \"--config\", \"$BUCKET/code/config.yaml\",
    \"--input\", \"$BUCKET/raw/\",
    \"--output\", \"$BUCKET/features/\"
  ]
}]"
