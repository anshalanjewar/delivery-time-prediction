"""
src/pipeline.py
===============
Preprocessing pipeline for the Amazon Delivery feature matrix.

Provides
--------
    make_preprocessor(X_train) -> ColumnTransformer
    wrap_model(preprocessor, estimator) -> Pipeline

Design
------
Numeric columns
    Imputer  : median  (robust to skew / remaining NaNs after cleaning)
    Scaler   : StandardScaler

Categorical columns (object dtype)
    Imputer  : most_frequent
    Encoder  : OneHotEncoder(handle_unknown="ignore", sparse_output=False)

The preprocessor is *fitted on training data only* inside ``wrap_model``
and then transforms validation/test sets without leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer fitted to the training feature matrix.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training split feature matrix (not yet transformed).

    Returns
    -------
    ColumnTransformer
        *Unfitted* transformer; call ``fit`` or ``fit_transform`` on it
        before use (done automatically inside ``wrap_model``).
    """
    numeric_cols: list[str] = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols: list[str] = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def wrap_model(
    X_train: pd.DataFrame,
    estimator: object,
) -> Pipeline:
    """Wrap an sklearn-compatible estimator inside a full preprocessing Pipeline.

    The preprocessing step is defined based on column dtypes of ``X_train``
    but is *not yet fitted* here; fitting happens when you call
    ``pipeline.fit(X_train, y_train)``.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix — used only to infer column names and dtypes.
    estimator : sklearn estimator
        Any object implementing ``fit`` / ``predict`` (e.g. RandomForestRegressor).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Two-step pipeline: ``("preprocessor", ColumnTransformer)``,
        ``("model", estimator)``.
    """
    preprocessor = make_preprocessor(X_train)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
