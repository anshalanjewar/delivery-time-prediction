"""
src/evaluate.py
===============
Final evaluation of all trained models on the held-out test set.

Outputs
-------
reports/tables/test_results.csv          — MAE / RMSE / R2 for every model
reports/figures/04_mae_bar.png           — horizontal bar chart of test MAE
reports/figures/05_pred_vs_actual.png    — predicted vs actual scatter (best model)
reports/figures/06_residual_hist.png     — residual distribution (best model)
models/best_model.joblib                 — best pipeline (LightGBM tuned)
models/categories.json                   — dropdown option lists for the app
"""

from __future__ import annotations

import sys
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    CLEANED_CSV, FIGURE_DPI, FIGURES_DIR, MODELS_DIR,
    RANDOM_SEED, TABLES_DIR, TARGET_COL,
)
from features import build_features

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"savefig.dpi": FIGURE_DPI})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, R2."""
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 6),
    }


def _load_split() -> tuple:
    """Reproduce the exact 70/15/15 split used in training."""
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
# Plot helpers
# ---------------------------------------------------------------------------
def _plot_mae_bar(results_df: pd.DataFrame) -> None:
    """Horizontal bar chart of test MAE per model."""
    df = results_df.sort_values("mae", ascending=True)
    colors = ["#2ecc71" if i == 0 else "#4C72B0" for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df["model"], df["mae"], color=colors, edgecolor="white", height=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.set_xlabel("Mean Absolute Error (minutes)", fontsize=12)
    ax.set_title("Test Set MAE by Model", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, df["mae"].max() * 1.18)
    plt.tight_layout()
    path = FIGURES_DIR / "04_mae_bar.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


def _plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Scatter of predicted vs actual with identity line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8, color="#4C72B0", rasterized=True)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Delivery_Time (min)", fontsize=12)
    ax.set_ylabel("Predicted Delivery_Time (min)", fontsize=12)
    ax.set_title(f"Predicted vs Actual — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = FIGURES_DIR / "05_pred_vs_actual.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Residual histogram with KDE overlay."""
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=60, color="#4C72B0", edgecolor="white",
            linewidth=0.3, alpha=0.8, density=True)
    # KDE overlay
    from scipy.stats import gaussian_kde
    kde_x = np.linspace(residuals.min(), residuals.max(), 300)
    kde = gaussian_kde(residuals)
    ax.plot(kde_x, kde(kde_x), "r-", linewidth=2, label="KDE")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero")
    ax.set_xlabel("Residual (Predicted − Actual, min)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Residual Distribution — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = FIGURES_DIR / "06_residual_hist.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run_evaluation() -> pd.DataFrame:
    """Evaluate all saved models on the test set, save outputs.

    Returns
    -------
    pd.DataFrame
        Test metrics table sorted by RMSE.
    """
    _, _, X_test, _, _, y_test = _load_split()
    print(f"Test set: {len(X_test):,} rows")

    # ── Load all saved pipelines ──────────────────────────────────────────
    model_names = [
        "LightGBM", "XGBoost", "RandomForest", "MLPRegressor", "LinearRegression",
        "DecisionTree", "KNeighbors",
    ]
    records: list[dict] = []
    pipelines: dict = {}

    for name in model_names:
        pkl = MODELS_DIR / f"{name}.pkl"
        if not pkl.exists():
            print(f"  [SKIP] {name}.pkl not found")
            continue
        with open(pkl, "rb") as f:
            pipe = pickle.load(f)
        pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        m = _metrics(y_test.values, y_pred)
        records.append({"model": name, **m})
        print(f"  {name:18s} | MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | R2={m['r2']:.6f}")

    results_df = pd.DataFrame(records).sort_values("rmse")
    out_path = TABLES_DIR / "test_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nTest results saved -> {out_path}")

    # ── Best model ────────────────────────────────────────────────────────
    best_name = results_df.iloc[0]["model"]
    best_pipe = pipelines[best_name]
    y_pred_best = best_pipe.predict(X_test)
    print(f"\nBest model: {best_name}")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    _plot_mae_bar(results_df)
    _plot_pred_vs_actual(y_test.values, y_pred_best, best_name)
    _plot_residuals(y_test.values, y_pred_best, best_name)

    # ── Save best pipeline ────────────────────────────────────────────────
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipe, best_path)
    print(f"\nBest pipeline saved -> {best_path}")

    # ── Save categories for Streamlit app ─────────────────────────────────
    raw = pd.read_csv(CLEANED_CSV)
    categories: dict[str, list] = {}
    for col in ["Weather", "Traffic", "Vehicle", "Area", "Category"]:
        categories[col] = sorted(raw[col].str.strip().dropna().unique().tolist())

    cat_path = MODELS_DIR / "categories.json"
    with open(cat_path, "w") as f:
        json.dump(categories, f, indent=2)
    print(f"Categories saved -> {cat_path}")

    print("\n" + "="*60)
    print("TEST SET RESULTS (sorted by RMSE)")
    print("="*60)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    run_evaluation()
