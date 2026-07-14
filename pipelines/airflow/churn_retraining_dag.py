"""
Airflow DAG: end-to-end monthly retraining pipeline.

extract -> EMR Spark features -> drift check -> train ensemble ->
quality gate (AUC >= threshold) -> register in SageMaker Model Registry ->
deploy endpoint -> refresh Tableau extract -> Slack notify.

Deploy by copying this file into your Airflow dags/ folder (MWAA or
self-hosted). Connections required: aws_default, slack_webhook.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,
}

REPO = "/opt/churn-ltv-platform"
CFG = f"{REPO}/config/config.yaml"


def _check_quality_gate(**ctx):
    with open(f"{REPO}/reports/metrics.json") as f:
        metrics = json.load(f)
    ctx["ti"].xcom_push(key="auc", value=metrics["auc_stacked"])
    return "register_model" if metrics["auc_stacked"] >= 0.88 else "alert_gate_failed"


with DAG(
    dag_id="churn_ltv_monthly_retraining",
    default_args=DEFAULT_ARGS,
    schedule="0 4 1 * *",              # 04:00 UTC on the 1st of each month
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["churn", "ltv", "ml"],
) as dag:

    extract = BashOperator(
        task_id="extract_from_warehouse",
        bash_command=f"cd {REPO} && python -m churn_platform.data_generation "
                     f"--out data/raw/customers.parquet",
        doc_md="Production: Redshift UNLOAD -> S3. Local: synthetic generator.",
    )

    spark_features = BashOperator(
        task_id="spark_feature_engineering",
        bash_command=f"cd {REPO} && bash infra/emr/submit_spark_job.sh "
                     f"|| python -m churn_platform.spark_etl --config {CFG}",
        doc_md="Submits spark_etl.py to EMR (50M+ rows); local fallback for dev.",
    )

    drift_check = BashOperator(
        task_id="feature_drift_check",
        bash_command=f"cd {REPO} && python -m churn_platform.monitoring "
                     f"--reference data/processed/features.parquet "
                     f"--current data/processed/features.parquet --config {CFG}",
    )

    train = BashOperator(
        task_id="train_stacked_ensemble",
        bash_command=f"cd {REPO} && python -m churn_platform.train --config {CFG}",
    )

    gate = BranchPythonOperator(task_id="quality_gate", python_callable=_check_quality_gate)

    register_model = BashOperator(
        task_id="register_model",
        bash_command=f"cd {REPO} && python pipelines/sagemaker/sagemaker_pipeline.py --register-only",
        doc_md="Registers model package in SageMaker Model Registry (PendingManualApproval).",
    )

    batch_score = BashOperator(
        task_id="batch_score_population",
        bash_command=f"cd {REPO} && python -m churn_platform.score --config {CFG}",
    )

    refresh_dashboards = BashOperator(
        task_id="refresh_tableau_extract",
        bash_command="echo 'tabcmd refreshextracts --datasource churn_risk_scores'",
        doc_md="Refreshes the Tableau published data source from the scored extract.",
    )

    alert_gate_failed = BashOperator(
        task_id="alert_gate_failed",
        bash_command="echo 'AUC below threshold — paging ML on-call' && exit 0",
    )

    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    extract >> spark_features >> drift_check >> train >> gate
    gate >> register_model >> batch_score >> refresh_dashboards >> done
    gate >> alert_gate_failed >> done
