"""
SageMaker Pipeline: processing -> training -> evaluation -> conditional
model registration. Mirrors the Airflow DAG for teams standardized on
SageMaker-native orchestration; Airflow triggers this pipeline in prod.

Requires AWS credentials + `pip install sagemaker`. Run:
    python pipelines/sagemaker/sagemaker_pipeline.py --upsert
"""
from __future__ import annotations

import argparse

import yaml


def build_pipeline(cfg: dict):
    # Imports inside the function so the repo works without the sagemaker SDK
    import sagemaker
    from sagemaker.processing import ProcessingInput, ProcessingOutput
    from sagemaker.sklearn.processing import SKLearnProcessor
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
    from sagemaker.workflow.functions import JsonGet
    from sagemaker.workflow.parameters import ParameterFloat, ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.step_collections import RegisterModel
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep

    role = cfg["aws"]["sagemaker_role_arn"]
    bucket = cfg["aws"]["s3_bucket"]
    sess = sagemaker.session.Session()

    input_data = ParameterString("InputData", default_value=f"{bucket}/features/")
    auc_threshold = ParameterFloat("AucThreshold", default_value=cfg["model"]["auc_threshold"])

    processor = SKLearnProcessor(framework_version="1.2-1", role=role,
                                 instance_type="ml.m5.4xlarge", instance_count=2)
    step_process = ProcessingStep(
        name="FeatureValidation",
        processor=processor,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
                 ProcessingOutput(output_name="test", source="/opt/ml/processing/test")],
        code="src/churn_platform/spark_etl.py",
    )

    estimator = SKLearn(entry_point="src/churn_platform/train.py",
                        framework_version="1.2-1", instance_type="ml.m5.4xlarge",
                        role=role, hyperparameters={"config": "config/config.yaml"})
    step_train = TrainingStep(name="TrainStackedEnsemble", estimator=estimator)

    eval_report = PropertyFile(name="EvalReport", output_name="evaluation",
                               path="metrics.json")
    step_eval = ProcessingStep(
        name="EvaluateHoldout", processor=processor,
        outputs=[ProcessingOutput(output_name="evaluation",
                                  source="/opt/ml/processing/evaluation")],
        code="src/churn_platform/score.py", property_files=[eval_report],
    )

    step_register = RegisterModel(
        name="RegisterChurnEnsemble", estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"], response_types=["application/json"],
        inference_instances=["ml.m5.xlarge"], transform_instances=["ml.m5.4xlarge"],
        model_package_group_name=cfg["aws"]["model_package_group"],
        approval_status="PendingManualApproval",
    )

    cond = ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_name=step_eval.name, property_file=eval_report,
                     json_path="auc_stacked"),
        right=auc_threshold,
    )
    step_gate = ConditionStep(name="AucQualityGate", conditions=[cond],
                              if_steps=[step_register], else_steps=[])

    return Pipeline(name="churn-ltv-retraining",
                    parameters=[input_data, auc_threshold],
                    steps=[step_process, step_train, step_eval, step_gate],
                    sagemaker_session=sess)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--upsert", action="store_true", help="create/update + start the pipeline")
    ap.add_argument("--register-only", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.upsert:
        p = build_pipeline(cfg)
        p.upsert(role_arn=cfg["aws"]["sagemaker_role_arn"])
        p.start()
        print("SageMaker pipeline upserted and started.")
    else:
        print("Dry run OK — pass --upsert with AWS credentials to deploy.")
