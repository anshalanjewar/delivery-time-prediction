"""
src/explain.py
==============
SHAP explainability for the best tree-based model (LightGBM).

Generates
---------
reports/figures/07_shap_summary.png   — beeswarm summary plot
reports/figures/08_shap_bar.png       — mean |SHAP| bar chart

Uses 1 000 randomly sampled test rows for speed.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    CLEANED_CSV, FIGURE_DPI, FIGURES_DIR, MODELS_DIR,
    RANDOM_SEED, TARGET_COL,
)
from features import build_features

warnings.filterwarnings("ignore")
plt.rcParams.update({"savefig.dpi": FIGURE_DPI})

N_EXPLAIN = 1_000   # rows to explain


def _load_test() -> tuple[pd.DataFrame, np.ndarray]:
    """Reproduce the exact test split and return (X_test_raw, y_test)."""
    df = pd.read_csv(CLEANED_CSV)
    X, y = build_features(df)
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, shuffle=True
    )
    X_val, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
    )
    return X_test, y_test.values


def run_explain() -> None:
    """Compute SHAP values and save summary + bar plots.

    Steps
    -----
    1. Load best model pipeline (LightGBM.pkl).
    2. Extract the fitted preprocessor + raw LightGBM booster.
    3. Transform test rows through the preprocessor.
    4. Compute TreeExplainer SHAP values on N_EXPLAIN rows.
    5. Save beeswarm and bar plots.
    """
    import joblib

    X_test, y_test = _load_test()

    # ── Subsample for speed ───────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False)
    X_sample = X_test.iloc[idx].reset_index(drop=True)

    # ── Load pipeline ─────────────────────────────────────────────────────
    model_path = MODELS_DIR / "LightGBM.pkl"
    import pickle
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)

    # ── Extract preprocessor and model ────────────────────────────────────
    preprocessor = pipe.named_steps["preprocessor"]
    lgbm_model   = pipe.named_steps["model"]

    # Transform the sample through the preprocessor
    X_transformed = preprocessor.transform(X_sample)

    # Get feature names after OHE
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # ── SHAP TreeExplainer ────────────────────────────────────────────────
    print(f"Computing SHAP values for {len(X_sample)} rows ...")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_transformed_df)

    # ── Plot 1: Beeswarm summary ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_transformed_df,
        plot_type="dot",
        max_display=15,
        show=False,
    )
    plt.title("SHAP Summary Plot — LightGBM (Top 15 Features)",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    path1 = FIGURES_DIR / "07_shap_summary.png"
    plt.savefig(path1, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path1.name}")

    # ── Plot 2: Mean |SHAP| bar ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_transformed_df,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("Mean |SHAP| Feature Importance — LightGBM",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    path2 = FIGURES_DIR / "08_shap_bar.png"
    plt.savefig(path2, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path2.name}")

    print("SHAP explanation complete.")


if __name__ == "__main__":
    run_explain()
