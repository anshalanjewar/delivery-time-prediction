"""
tests/test_models.py
====================
Pytest suite confirming DT and KNN integration into the delivery-time project.
Covers pipeline structure, CSV completeness, best-model identity,
a deterministic prediction, and README presence.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
SRC_DIR     = ROOT / "src"
MODELS_DIR  = ROOT / "models"
TABLES_DIR  = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SEVEN_MODELS = [
    "LightGBM", "XGBoost", "RandomForest",
    "DecisionTree", "KNeighbors",
    "MLPRegressor", "LinearRegression",
]


# ── fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def val_df():
    return pd.read_csv(TABLES_DIR / "model_comparison_val.csv")


@pytest.fixture(scope="session")
def test_df():
    return pd.read_csv(TABLES_DIR / "test_results.csv")


@pytest.fixture(scope="session")
def best_pipe():
    return joblib.load(MODELS_DIR / "best_model.joblib")


@pytest.fixture(scope="session")
def categories():
    with open(MODELS_DIR / "categories.json") as f:
        return json.load(f)


def _load_pipe(name: str) -> Pipeline:
    with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════
# 1. CSV completeness
# ══════════════════════════════════════════════════════════════════════════
class TestCSVs:
    def test_val_has_seven_rows(self, val_df):
        assert len(val_df) == 7, f"val CSV has {len(val_df)} rows"

    def test_test_has_seven_rows(self, test_df):
        assert len(test_df) == 7, f"test CSV has {len(test_df)} rows"

    @pytest.mark.parametrize("model", SEVEN_MODELS)
    def test_model_in_val(self, val_df, model):
        assert model in val_df["model"].values

    @pytest.mark.parametrize("model", SEVEN_MODELS)
    def test_model_in_test(self, test_df, model):
        assert model in test_df["model"].values

    def test_val_has_train_time_col(self, val_df):
        assert "train_time_s" in val_df.columns

    @pytest.mark.parametrize("model", SEVEN_MODELS)
    def test_val_metrics_finite(self, val_df, model):
        row = val_df[val_df["model"] == model].iloc[0]
        for col in ["mae", "rmse", "r2"]:
            assert np.isfinite(row[col]), f"{model}.{col} is not finite"

    @pytest.mark.parametrize("model", SEVEN_MODELS)
    def test_test_metrics_finite(self, test_df, model):
        row = test_df[test_df["model"] == model].iloc[0]
        for col in ["mae", "rmse", "r2"]:
            assert np.isfinite(row[col]), f"{model}.{col} is not finite"


# ══════════════════════════════════════════════════════════════════════════
# 2. Pipeline structure
# ══════════════════════════════════════════════════════════════════════════
class TestPipelines:
    @pytest.mark.parametrize("name", SEVEN_MODELS)
    def test_is_sklearn_pipeline(self, name):
        pipe = _load_pipe(name)
        assert isinstance(pipe, Pipeline)

    @pytest.mark.parametrize("name", SEVEN_MODELS)
    def test_has_preprocessor_step(self, name):
        pipe = _load_pipe(name)
        assert "preprocessor" in pipe.named_steps

    @pytest.mark.parametrize("name", SEVEN_MODELS)
    def test_has_model_step(self, name):
        pipe = _load_pipe(name)
        assert "model" in pipe.named_steps

    @pytest.mark.parametrize("name", SEVEN_MODELS)
    def test_preprocessor_is_column_transformer(self, name):
        pipe = _load_pipe(name)
        assert isinstance(pipe.named_steps["preprocessor"], ColumnTransformer)

    @pytest.mark.parametrize("name", SEVEN_MODELS)
    def test_numeric_pipe_has_standard_scaler(self, name):
        pipe = _load_pipe(name)
        prep = pipe.named_steps["preprocessor"]
        found = False
        for tname, transformer, _ in prep.transformers:
            if tname == "num":
                for _, step in transformer.steps:
                    if isinstance(step, StandardScaler):
                        found = True
        assert found, f"{name}: StandardScaler not found in numeric sub-pipeline"

    def test_decision_tree_random_state(self):
        pipe = _load_pipe("DecisionTree")
        model = pipe.named_steps["model"]
        assert getattr(model, "random_state", None) == 42

    def test_knn_no_random_state_needed(self):
        """KNeighborsRegressor is deterministic; just confirm it loads."""
        pipe = _load_pipe("KNeighbors")
        assert pipe is not None


# ══════════════════════════════════════════════════════════════════════════
# 3. best_model.joblib is lowest val-MAE
# ══════════════════════════════════════════════════════════════════════════
class TestBestModel:
    _CLASS_TO_NAME = {
        "LGBMRegressor":         "LightGBM",
        "XGBRegressor":          "XGBoost",
        "RandomForestRegressor": "RandomForest",
        "DecisionTreeRegressor": "DecisionTree",
        "KNeighborsRegressor":   "KNeighbors",
        "MLPRegressor":          "MLPRegressor",
        "LinearRegression":      "LinearRegression",
    }

    def test_best_model_is_lowest_val_mae(self, best_pipe, val_df):
        best_class = type(best_pipe.named_steps["model"]).__name__
        best_name  = self._CLASS_TO_NAME.get(best_class, best_class)
        expected   = val_df.sort_values("mae").iloc[0]["model"]
        assert best_name == expected, (
            f"best_model.joblib is {best_name} but lowest val-MAE is {expected}"
        )

    def test_best_model_r2_above_zero_point_eight(self, best_pipe, val_df):
        best_class = type(best_pipe.named_steps["model"]).__name__
        best_name  = self._CLASS_TO_NAME.get(best_class, best_class)
        r2 = val_df[val_df["model"] == best_name].iloc[0]["r2"]
        assert r2 > 0.80, f"Best model val R2 = {r2:.4f}, expected > 0.80"


# ══════════════════════════════════════════════════════════════════════════
# 4. Deterministic prediction (specific inputs)
# ══════════════════════════════════════════════════════════════════════════
class TestPrediction:
    """
    Reference prediction:
      9 km, Medium, Sunny, motorcycle, prep=15, hour=17, Mon(0),
      Metropolitian, Snacks, age=28, rating=5.0
    Verified prediction: 88.19 min (LightGBM best model, seed=42 split).
    """
    EXPECTED = 88.19
    TOL      = 1.0   # allow ±1 min for floating-point / env differences

    @staticmethod
    def _make_row() -> pd.DataFrame:
        order_hour, day_of_week = 17, 0
        return pd.DataFrame([{
            "Agent_Age":     28,
            "Agent_Rating":  5.0,
            "Weather":       "Sunny",
            "Traffic":       "Medium",
            "Vehicle":       "motorcycle",
            "Area":          "Metropolitian",
            "Category":      "Snacks",
            "distance_km":   9.0,
            "order_hour":    float(order_hour),
            "day_of_week":   float(day_of_week),
            "is_weekend":    0.0,
            "is_peak":       0.0,   # hour 17 not in peak set
            "prep_time_min": 15.0,
        }])

    def test_point_estimate_close_to_reference(self, best_pipe):
        row   = self._make_row()
        point = float(best_pipe.predict(row)[0])
        assert abs(point - self.EXPECTED) <= self.TOL, (
            f"Prediction {point:.2f} differs from reference "
            f"{self.EXPECTED} by more than {self.TOL} min"
        )

    def test_point_estimate_positive(self, best_pipe):
        row   = self._make_row()
        point = float(best_pipe.predict(row)[0])
        assert point > 0

    def test_quantile_interval_ordered(self):
        with open(MODELS_DIR / "lgbm_q10.pkl", "rb") as f:
            q10 = pickle.load(f)
        with open(MODELS_DIR / "lgbm_q90.pkl", "rb") as f:
            q90 = pickle.load(f)
        row   = self._make_row()
        lower = float(q10.predict(row)[0])
        upper = float(q90.predict(row)[0])
        assert lower < upper, f"q10={lower:.2f} >= q90={upper:.2f}"


# ══════════════════════════════════════════════════════════════════════════
# 5. Figures & README
# ══════════════════════════════════════════════════════════════════════════
class TestArtifacts:
    def test_mae_bar_chart_exists(self):
        assert (FIGURES_DIR / "04_mae_bar.png").exists()

    def test_pred_vs_actual_exists(self):
        assert (FIGURES_DIR / "05_pred_vs_actual.png").exists()

    def test_residual_hist_exists(self):
        assert (FIGURES_DIR / "06_residual_hist.png").exists()

    def test_readme_contains_decision_tree(self):
        readme = (ROOT / "README.md").read_text()
        assert "DecisionTree" in readme

    def test_readme_contains_kneighbors(self):
        readme = (ROOT / "README.md").read_text()
        assert "KNeighbors" in readme

    def test_readme_says_seven_models(self):
        readme = (ROOT / "README.md").read_text()
        assert "7-model" in readme or "seven" in readme.lower()

    def test_requirements_has_plotly(self):
        req = (ROOT / "requirements.txt").read_text()
        assert "plotly" in req

    @pytest.mark.parametrize("name", SEVEN_MODELS)
    def test_pkl_exists(self, name):
        assert (MODELS_DIR / f"{name}.pkl").exists()
