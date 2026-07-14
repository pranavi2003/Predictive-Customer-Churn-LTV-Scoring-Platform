"""
Real-time churn & LTV scoring API (FastAPI).

In production this container image is deployed behind a SageMaker real-time
endpoint (or ECS/Fargate) and serves p50 < 40ms single-record scoring for
the retention-campaign decision engine.

Run locally:
    uvicorn churn_platform.serving.app:app --reload --port 8000
"""
from __future__ import annotations

import pathlib

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = pathlib.Path(__file__).resolve().parents[3] / "models"

app = FastAPI(title="Churn & LTV Scoring API", version="1.0.0")
_churn = joblib.load(MODEL_DIR / "churn_ensemble.joblib")
_ltv = joblib.load(MODEL_DIR / "ltv_model.joblib")


class CustomerFeatures(BaseModel):
    """Post-ETL feature vector (same schema Spark writes)."""
    features: dict = Field(..., description="feature_name -> value")


class ScoreResponse(BaseModel):
    churn_probability: float
    predicted_ltv: float
    revenue_at_risk: float
    risk_tier: str


def _tier(p: float) -> str:
    return "critical" if p > .85 else "high" if p > .6 else "medium" if p > .3 else "low"


@app.get("/health")
def health():
    return {"status": "ok", "model": "churn_ensemble+ltv", "features": len(_churn["feature_names"])}


@app.post("/score", response_model=ScoreResponse)
def score(payload: CustomerFeatures):
    X = pd.DataFrame([payload.features])[_churn["feature_names"]]
    base = np.zeros((1, 2))
    for m in _churn["xgb_models"]:
        base[:, 0] += m.predict_proba(X)[:, 1] / len(_churn["xgb_models"])
    for m in _churn["lgb_models"]:
        base[:, 1] += m.predict_proba(X)[:, 1] / len(_churn["lgb_models"])
    p = float(_churn["meta"].predict_proba(base)[0, 1])
    ltv = float(max(_ltv["model"].predict(X[_ltv["feature_names"]])[0], 0))
    return ScoreResponse(churn_probability=round(p, 4), predicted_ltv=round(ltv, 2),
                         revenue_at_risk=round(p * ltv, 2), risk_tier=_tier(p))
