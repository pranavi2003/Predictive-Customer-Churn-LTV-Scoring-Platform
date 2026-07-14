"""
Model training: stacked ensemble (XGBoost + LightGBM -> logistic meta-learner)
for churn, and a LightGBM regressor for customer lifetime value (LTV).

Stacking uses out-of-fold (OOF) base-model predictions so the meta-learner
never sees leaked in-fold predictions. Artifacts + metrics are written to
`models/` and `reports/` and consumed by the serving API, batch scorer,
Airflow quality gate, and SageMaker model registry step.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

DROP_COLS = ["customer_id", "churned", "ltv_actual"]


def load_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def split_xy(df: pd.DataFrame, cfg: dict):
    y = df[cfg["model"]["target"]].values
    y_ltv = df[cfg["model"]["ltv_target"]].values
    X = df.drop(columns=DROP_COLS)
    return X, y, y_ltv


def train_stacked_ensemble(X_tr, y_tr, X_te, y_te, cfg: dict, model_dir: pathlib.Path):
    mcfg = cfg["model"]
    skf = StratifiedKFold(n_splits=mcfg["n_folds"], shuffle=True,
                          random_state=cfg["data"]["random_seed"])

    oof = np.zeros((len(X_tr), 2))
    test_preds = np.zeros((len(X_te), 2))
    xgb_models, lgb_models = [], []

    for fold, (i_tr, i_va) in enumerate(skf.split(X_tr, y_tr), 1):
        Xf, yf = X_tr.iloc[i_tr], y_tr[i_tr]
        Xv = X_tr.iloc[i_va]

        m_xgb = xgb.XGBClassifier(**mcfg["xgboost"], eval_metric="auc",
                                  random_state=fold, n_jobs=-1)
        m_xgb.fit(Xf, yf, verbose=False)
        oof[i_va, 0] = m_xgb.predict_proba(Xv)[:, 1]
        test_preds[:, 0] += m_xgb.predict_proba(X_te)[:, 1] / mcfg["n_folds"]
        xgb_models.append(m_xgb)

        m_lgb = lgb.LGBMClassifier(**mcfg["lightgbm"], random_state=fold,
                                   n_jobs=-1, verbose=-1)
        m_lgb.fit(Xf, yf)
        oof[i_va, 1] = m_lgb.predict_proba(Xv)[:, 1]
        test_preds[:, 1] += m_lgb.predict_proba(X_te)[:, 1] / mcfg["n_folds"]
        lgb_models.append(m_lgb)

        fold_auc = roc_auc_score(y_tr[i_va], oof[i_va].mean(axis=1))
        print(f"[fold {fold}] blended OOF AUC = {fold_auc:.4f}")

    meta = LogisticRegression(max_iter=1000)
    meta.fit(oof, y_tr)

    stack_test = meta.predict_proba(test_preds)[:, 1]
    metrics = {
        "auc_xgb": float(roc_auc_score(y_te, test_preds[:, 0])),
        "auc_lgb": float(roc_auc_score(y_te, test_preds[:, 1])),
        "auc_stacked": float(roc_auc_score(y_te, stack_test)),
        "pr_auc_stacked": float(average_precision_score(y_te, stack_test)),
        "meta_weights": meta.coef_.ravel().tolist(),
    }

    joblib.dump({"xgb_models": xgb_models, "lgb_models": lgb_models, "meta": meta,
                 "feature_names": list(X_tr.columns)},
                model_dir / "churn_ensemble.joblib")
    return metrics, stack_test


def train_ltv_model(X_tr, ltv_tr, X_te, ltv_te, cfg: dict, model_dir: pathlib.Path):
    m = lgb.LGBMRegressor(**cfg["model"]["ltv_model"], objective="regression_l1",
                          random_state=cfg["data"]["random_seed"], n_jobs=-1, verbose=-1)
    m.fit(X_tr, ltv_tr)
    pred = m.predict(X_te)
    joblib.dump({"model": m, "feature_names": list(X_tr.columns)},
                model_dir / "ltv_model.joblib")
    return {"ltv_mae": float(mean_absolute_error(ltv_te, pred)),
            "ltv_r2": float(r2_score(ltv_te, pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t0 = time.time()
    model_dir = pathlib.Path(cfg["paths"]["model_dir"]); model_dir.mkdir(exist_ok=True)
    reports = pathlib.Path(cfg["paths"]["reports_dir"]); reports.mkdir(exist_ok=True)

    df = load_features(cfg["data"]["features_path"])
    X, y, y_ltv = split_xy(df, cfg)
    X_tr, X_te, y_tr, y_te, ltv_tr, ltv_te = train_test_split(
        X, y, y_ltv, test_size=cfg["model"]["test_size"],
        stratify=y, random_state=cfg["data"]["random_seed"])
    print(f"Train: {len(X_tr):,}  Holdout: {len(X_te):,}  Churn rate: {y.mean():.3f}")

    churn_metrics, _ = train_stacked_ensemble(X_tr, y_tr, X_te, y_te, cfg, model_dir)
    ltv_metrics = train_ltv_model(X_tr, ltv_tr, X_te, ltv_te, cfg, model_dir)

    # baseline feature distribution for PSI drift monitoring
    baseline = X_tr.describe(percentiles=[.1, .25, .5, .75, .9]).to_dict()
    (model_dir / "training_baseline.json").write_text(json.dumps(baseline, default=str))

    metrics = {**churn_metrics, **ltv_metrics,
               "n_train": len(X_tr), "n_holdout": len(X_te),
               "training_seconds": round(time.time() - t0, 1)}
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    gate = cfg["model"]["auc_threshold"]
    status = "PASSED" if metrics["auc_stacked"] >= gate else "FAILED"
    print(f"\nQuality gate (AUC >= {gate}): {status}")


if __name__ == "__main__":
    main()
