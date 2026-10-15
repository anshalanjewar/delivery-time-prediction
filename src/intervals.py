"""
src/intervals.py
================
Prediction intervals via LightGBM quantile regression.

Trains two quantile LightGBM models:
  q10  (alpha = 0.10) — lower bound of 80% interval
  q90  (alpha = 0.90) — upper bound of 80% interval

Evaluates empirical coverage of the 80% interval on the test set
and saves both models to models/.

Outputs
-------
models/lgbm_q10.pkl            — lower-bound quantile pipeline
models/lgbm_q90.pkl            — upper-bound quantile pipeline
reports/tables/interval_coverage.csv  — empirical coverage summary
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    CLEANED_CSV, MODELS_DIR, RANDOM_SEED, TABLES_DIR, TARGET_COL
)
from features import build_features
from pipeline import wrap_model


TARGET_COVERAGE: float = 0.80
ALPHAS: dict[str, float] = {"q10": 0.10, "q90": 0.90}


def _load_split() -> tuple:
    """Reproduce exact 70/15/15 split."""
    df = pd.read_csv(CLEANED_CSV)
    X, y = build_features(df)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, object]:
    """Train and save quantile LightGBM pipelines.

    Parameters
    ----------
    X_train, y_train : Training data.

    Returns
    -------
    dict mapping quantile name (e.g. "q10") to fitted pipeline.
    """
    fitted: dict[str, object] = {}

    for name, alpha in ALPHAS.items():
        print(f"  Training LightGBM quantile alpha={alpha} ({name}) ...")
        estimator = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        pipe = wrap_model(X_train, estimator)
        pipe.fit(X_train, y_train)

        path = MODELS_DIR / f"lgbm_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipe, f)
        print(f"    Saved -> {path.name}")
        fitted[name] = pipe

    return fitted


def evaluate_coverage(
    fitted: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compute empirical coverage and average interval width on test set.

    Parameters
    ----------
    fitted  : Dict with "q10" and "q90" fitted quantile pipelines.
    X_test, y_test : Test data.

    Returns
    -------
    pd.DataFrame with coverage summary.
    """
    lo = fitted["q10"].predict(X_test)
    hi = fitted["q90"].predict(X_test)
    y = y_test.values

    in_interval = (y >= lo) & (y <= hi)
    coverage = float(in_interval.mean())
    avg_width = float((hi - lo).mean())
    median_width = float(np.median(hi - lo))

    summary = pd.DataFrame([{
        "target_coverage":   TARGET_COVERAGE,
        "empirical_coverage": round(coverage, 4),
        "avg_interval_width_min": round(avg_width, 2),
        "median_interval_width_min": round(median_width, 2),
        "n_test": len(y),
    }])

    out_path = TABLES_DIR / "interval_coverage.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nInterval coverage summary saved -> {out_path}")
    print(summary.to_string(index=False))
    return summary


def run_intervals() -> pd.DataFrame:
    """Full quantile training + coverage evaluation pipeline.

    Returns
    -------
    pd.DataFrame
        Coverage summary table.
    """
    X_train, _, X_test, y_train, _, y_test = _load_split()
    print(f"Train={len(X_train):,} | Test={len(X_test):,}")

    fitted = train_quantile_models(X_train, y_train)
    return evaluate_coverage(fitted, X_test, y_test)


if __name__ == "__main__":
    run_intervals()
