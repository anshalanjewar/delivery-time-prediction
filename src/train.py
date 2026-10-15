"""
src/train.py
============
Train and evaluate seven regression models on the Amazon Delivery dataset.

Workflow
--------
1. Load cleaned CSV  ->  build_features()  ->  (X, y).
2. Split 70 / 15 / 15  train / val / test  (stratified split not needed for
   regression; random_state=42 is sufficient for reproducibility).
3. For each model: wrap in preprocessing Pipeline, fit on train, evaluate on val.
4. Save metrics to  reports/tables/model_comparison_val.csv.
5. Serialize each fitted Pipeline to  models/<name>.pkl.

Models
------
* LinearRegression
* RandomForestRegressor   (n_estimators=200)
* XGBRegressor            (n_estimators=400, learning_rate=0.05)
* LGBMRegressor           (n_estimators=400, learning_rate=0.05)
* MLPRegressor            (hidden_layer_sizes=(128,64), max_iter=300)
* DecisionTree            (random_state=42, max_depth=8 — lightly tuned)
* KNeighbors              (n_neighbors=9 — lightly tuned; scaling via pipeline)
"""

from __future__ import annotations

import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CLEANED_CSV,
    MODELS_DIR,
    RANDOM_SEED,
    TABLES_DIR,
    TARGET_COL,
)
from features import build_features  # noqa: E402
from pipeline import wrap_model      # noqa: E402


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, R2.

    Parameters
    ----------
    y_true : array-like  — ground truth target values.
    y_pred : array-like  — model predictions.

    Returns
    -------
    dict with keys ``mae``, ``rmse``, ``r2``.
    """
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(_rmse(y_true, y_pred), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 6),
    }


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def _model_registry(random_state: int) -> dict[str, object]:
    """Return a dict of {model_name: unfitted estimator}.

    Parameters
    ----------
    random_state : int  — controls reproducibility for stochastic models.

    Returns
    -------
    dict[str, estimator]
    """
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        "MLPRegressor": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=300,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        # --- new models -------------------------------------------------------
        # max_depth=8: shallow enough to avoid severe overfitting on ~28k rows,
        # deep enough to capture non-linear interactions.
        "DecisionTree": DecisionTreeRegressor(
            max_depth=8,
            random_state=random_state,
        ),
        # n_neighbors=9: odd number avoids ties; pipeline's StandardScaler
        # ensures numeric features are on the same scale before distance calc.
        "KNeighbors": KNeighborsRegressor(
            n_neighbors=9,
            weights="distance",
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def run_training() -> pd.DataFrame:
    """Execute the full training and evaluation loop.

    Steps
    -----
    1. Load features.
    2. 70/15/15 split (train/val/test).
    3. For each model: build pipeline, fit on train, evaluate on val.
    4. Save comparison table and serialised pipelines.

    Returns
    -------
    pd.DataFrame
        Validation metrics table (one row per model).
    """
    # ── 1. Load & build features ──────────────────────────────────────────
    print("Loading cleaned data ...")
    df = pd.read_csv(CLEANED_CSV)
    X, y = build_features(df)
    print(f"  Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} cols")
    print(f"  Target: {TARGET_COL} | mean={y.mean():.1f} | std={y.std():.1f}")

    # ── 2. 70 / 15 / 15 split ─────────────────────────────────────────────
    # First carve off 30% temp  (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, shuffle=True
    )
    # Then split temp evenly -> 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
    )
    print(
        f"\nSplit: train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}"
    )

    # Persist split indices for later use (optional)
    split_info = pd.DataFrame({
        "set": (
            ["train"] * len(X_train)
            + ["val"] * len(X_val)
            + ["test"] * len(X_test)
        ),
        "index": list(X_train.index) + list(X_val.index) + list(X_test.index),
    })
    split_info.to_csv(TABLES_DIR / "data_splits.csv", index=False)

    # ── 3. Train & evaluate each model ───────────────────────────────────
    models = _model_registry(RANDOM_SEED)
    records: list[dict] = []

    for name, estimator in models.items():
        print(f"\n[{name}] Training ...")
        pipe = wrap_model(X_train, estimator)

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_sec = time.perf_counter() - t0

        y_pred_val = pipe.predict(X_val)
        metrics = evaluate(y_val.values, y_pred_val)

        row = {
            "model": name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "train_time_s": round(train_sec, 2),
        }
        records.append(row)

        print(
            f"  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}"
            f"  R2={metrics['r2']:.6f}  time={train_sec:.1f}s"
        )

        # ── Leakage check ─────────────────────────────────────────────────
        if metrics["r2"] > 0.99:
            print(
                f"  [WARNING] R2 > 0.99 on validation set for {name}. "
                "Checking for potential data leakage ..."
            )
            y_pred_train = pipe.predict(X_train)
            train_metrics = evaluate(y_train.values, y_pred_train)
            print(
                f"  Train metrics: MAE={train_metrics['mae']:.4f}"
                f"  R2={train_metrics['r2']:.6f}"
            )
            if train_metrics["r2"] > 0.999 and metrics["r2"] > 0.99:
                print(
                    "  [LEAKAGE FLAG] Near-perfect train AND val R2. "
                    "Review feature set for target-derived columns!"
                )

        # ── Persist pipeline ───────────────────────────────────────────────
        model_path = MODELS_DIR / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
        print(f"  Model saved -> {model_path.name}")

    # ── 4. Save comparison table ──────────────────────────────────────────
    results_df = pd.DataFrame(records).sort_values("rmse")
    out_path = TABLES_DIR / "model_comparison_val.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nComparison table saved -> {out_path}")

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_training()
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS (sorted by RMSE)")
    print("=" * 60)
    print(results.to_string(index=False))
