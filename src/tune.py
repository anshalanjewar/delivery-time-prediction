"""
src/tune.py
===========
Hyperparameter tuning for the best two models (LightGBM, XGBoost)
using RandomizedSearchCV on the training split only.

Workflow
--------
1. Reload cleaned data -> build_features -> 70/15/15 split (same seed).
2. Run RandomizedSearchCV (3-fold CV, n_iter=40, scoring=neg_MAE) on train.
3. Refit best estimator on full training set.
4. Evaluate default vs tuned on validation set.
5. Save comparison to reports/tables/tuning_comparison.csv.
6. Overwrite models/LightGBM.pkl and models/XGBoost.pkl with tuned versions.
"""

from __future__ import annotations

import sys
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLEANED_CSV, MODELS_DIR, RANDOM_SEED, TABLES_DIR, TARGET_COL
from features import build_features
from pipeline import wrap_model

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(rmse, 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 6),
    }


def _load_split() -> tuple:
    """Load features and reproduce the exact 70/15/15 split."""
    df = pd.read_csv(CLEANED_CSV)
    X, y = build_features(df)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
LGBM_SPACE: dict = {
    "model__n_estimators":    [200, 400, 600, 800],
    "model__learning_rate":   [0.01, 0.03, 0.05, 0.08, 0.1],
    "model__max_depth":       [4, 6, 8, 10, -1],
    "model__num_leaves":      [31, 63, 127, 255],
    "model__subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
    "model__min_child_samples":[10, 20, 30, 50],
    "model__reg_alpha":       [0, 0.01, 0.05, 0.1, 0.5],
    "model__reg_lambda":      [0, 0.01, 0.05, 0.1, 0.5],
}

XGB_SPACE: dict = {
    "model__n_estimators":    [200, 400, 600, 800],
    "model__learning_rate":   [0.01, 0.03, 0.05, 0.08, 0.1],
    "model__max_depth":       [4, 6, 8, 10],
    "model__subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
    "model__gamma":           [0, 0.1, 0.2, 0.5, 1.0],
    "model__min_child_weight":[1, 3, 5, 7],
    "model__reg_alpha":       [0, 0.01, 0.05, 0.1],
    "model__reg_lambda":      [0.5, 1, 1.5, 2],
}


# ---------------------------------------------------------------------------
# Tuning runner
# ---------------------------------------------------------------------------
def tune_model(
    name: str,
    estimator,
    param_space: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int = 40,
) -> tuple[dict, dict, object]:
    """Run RandomizedSearchCV and return default/tuned metric dicts + fitted pipeline.

    Parameters
    ----------
    name        : Human-readable model name.
    estimator   : Unfitted sklearn-compatible estimator (default params).
    param_space : Dict of hyperparameter distributions for RandomizedSearchCV.
    X_train, y_train : Training features and target.
    X_val, y_val     : Validation features and target.
    n_iter      : Number of random parameter settings sampled.

    Returns
    -------
    default_row : dict with validation metrics for default model.
    tuned_row   : dict with validation metrics for tuned model.
    tuned_pipe  : Fitted pipeline with best hyperparameters.
    """
    print(f"\n{'='*55}")
    print(f"  Tuning: {name}  (n_iter={n_iter}, cv=3)")
    print(f"{'='*55}")

    # ── Default metrics (load saved pkl) ──────────────────────────────────
    pkl_path = MODELS_DIR / f"{name}.pkl"
    with open(pkl_path, "rb") as f:
        default_pipe = pickle.load(f)
    y_pred_default = default_pipe.predict(X_val)
    def_m = _metrics(y_val.values, y_pred_default)
    default_row = {"model": name, "variant": "default", **def_m}
    print(f"  Default  -> MAE={def_m['mae']:.4f}  RMSE={def_m['rmse']:.4f}  R2={def_m['r2']:.6f}")

    # ── Build fresh pipeline for search ───────────────────────────────────
    pipe = wrap_model(X_train, estimator)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_space,
        n_iter=n_iter,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=True,
        verbose=1,
    )

    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    best_pipe = search.best_estimator_
    y_pred_tuned = best_pipe.predict(X_val)
    tun_m = _metrics(y_val.values, y_pred_tuned)
    tuned_row = {
        "model": name,
        "variant": "tuned",
        **tun_m,
        "best_params": str(search.best_params_),
        "search_time_s": round(elapsed, 1),
    }
    print(f"  Tuned    -> MAE={tun_m['mae']:.4f}  RMSE={tun_m['rmse']:.4f}  R2={tun_m['r2']:.6f}  ({elapsed:.0f}s)")
    print(f"  Best params: {search.best_params_}")

    # Persist tuned pipeline
    with open(pkl_path, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"  Saved tuned pipeline -> {pkl_path.name}")

    return default_row, tuned_row, best_pipe


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_tuning() -> pd.DataFrame:
    """Tune LightGBM and XGBoost, save comparison table, return DataFrame."""
    X_train, X_val, X_test, y_train, y_val, y_test = _load_split()
    print(f"Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

    configs = [
        (
            "LightGBM",
            LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
            LGBM_SPACE,
        ),
        (
            "XGBoost",
            XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0),
            XGB_SPACE,
        ),
    ]

    records: list[dict] = []
    for name, est, space in configs:
        def_row, tun_row, _ = tune_model(
            name, est, space, X_train, y_train, X_val, y_val, n_iter=40
        )
        records.extend([def_row, tun_row])

    df_out = pd.DataFrame(records)
    out_path = TABLES_DIR / "tuning_comparison.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nTuning comparison saved -> {out_path}")

    print("\n" + "="*60)
    print("TUNING COMPARISON (Validation Set)")
    print("="*60)
    show_cols = ["model", "variant", "mae", "rmse", "r2"]
    print(df_out[show_cols].to_string(index=False))
    return df_out


if __name__ == "__main__":
    run_tuning()
