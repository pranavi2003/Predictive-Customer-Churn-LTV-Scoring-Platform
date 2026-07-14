"""
Feature/score drift monitoring via Population Stability Index (PSI).

Airflow runs this daily against the newest scored batch; PSI > threshold on
key features or on the score distribution pages the ML on-call and can
auto-trigger the retraining DAG.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e_pct = np.clip(np.histogram(expected, qs)[0] / len(expected), 1e-6, None)
    a_pct = np.clip(np.histogram(actual, qs)[0] / len(actual), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def run_drift_check(reference_path: str, current_path: str, cfg: dict) -> dict:
    ref = pd.read_parquet(reference_path)
    cur = pd.read_parquet(current_path)
    numeric = ref.select_dtypes("number").columns.intersection(cur.columns)
    report = {c: round(psi(ref[c].values, cur[c].values), 4) for c in numeric}
    threshold = cfg["monitoring"]["psi_threshold"]
    drifted = {c: v for c, v in report.items() if v > threshold}
    return {"psi": report, "drifted_features": drifted,
            "retraining_recommended": len(drifted) > 0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(json.dumps(run_drift_check(args.reference, args.current, cfg), indent=2))
