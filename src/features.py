"""
src/features.py
===============
Feature engineering for the Amazon Delivery dataset.

Entry point
-----------
    build_features(df) -> tuple[pd.DataFrame, pd.Series]

Engineered features
-------------------
distance_km     : Haversine great-circle distance (store -> drop), km.
order_hour      : Hour extracted from Order_Time (0-23).
day_of_week     : Day-of-week from Order_Date (0=Mon … 6=Sun).
is_weekend      : 1 if day_of_week in {5, 6}, else 0.
is_peak         : 1 if order_hour in {12,13,14,19,20,21,22}, else 0.
prep_time_min   : Minutes from Order_Time to Pickup_Time.
                  Handles midnight crossing via modulo-1440 arithmetic.

Kept as-is (after cleaning)
----------------------------
Agent_Age, Agent_Rating, Weather, Traffic, Vehicle, Area, Category.

Dropped
-------
Order_ID, Store_Latitude, Store_Longitude, Drop_Latitude, Drop_Longitude,
Order_Date, Order_Time, Pickup_Time.
"""

from __future__ import annotations

import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import TARGET_COL  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EARTH_RADIUS_KM: float = 6_371.0

_PEAK_HOURS: frozenset[int] = frozenset({12, 13, 14, 19, 20, 21, 22})
_WEEKEND_DAYS: frozenset[int] = frozenset({5, 6})  # Mon=0 … Sun=6

_COLS_TO_DROP: list[str] = [
    "Order_ID",
    "Store_Latitude",
    "Store_Longitude",
    "Drop_Latitude",
    "Drop_Longitude",
    "Order_Date",
    "Order_Time",
    "Pickup_Time",
]


# ---------------------------------------------------------------------------
# Helper: haversine distance
# ---------------------------------------------------------------------------
def _haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """Compute element-wise haversine distance (km) between two lat/lon pairs.

    Parameters
    ----------
    lat1, lon1 : pd.Series  — store coordinates in decimal degrees.
    lat2, lon2 : pd.Series  — drop coordinates in decimal degrees.

    Returns
    -------
    pd.Series
        Distance in kilometres for each row.
    """
    r = math.radians(1)  # conversion factor
    phi1 = lat1 * r
    phi2 = lat2 * r
    d_phi = (lat2 - lat1) * r
    d_lam = (lon2 - lon1) * r

    a = (
        np.sin(d_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return _EARTH_RADIUS_KM * c


# ---------------------------------------------------------------------------
# Helper: parse "HH:MM:SS" or "HH:MM" -> total minutes since midnight
# ---------------------------------------------------------------------------
def _time_to_minutes(series: pd.Series) -> pd.Series:
    """Convert a time string column to total minutes since midnight.

    Accepts ``HH:MM:SS`` or ``HH:MM`` formats (already stripped).

    Parameters
    ----------
    series : pd.Series  — raw time strings.

    Returns
    -------
    pd.Series (float)
        Minutes since midnight (NaN for unparseable rows).
    """
    parts = series.str.split(":", expand=True)
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    return hours * 60 + minutes


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Engineer features from a cleaned Amazon Delivery DataFrame.

    All transformations are purely row-wise (no fit/transform leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``clean()`` — cleaned but not yet split or scaled.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric + raw categorical strings, no target).
    y : pd.Series
        Target vector ``Delivery_Time`` (minutes).
    """
    df = df.copy()

    # ── 1. Haversine distance ──────────────────────────────────────────────
    df["distance_km"] = _haversine_km(
        df["Store_Latitude"],
        df["Store_Longitude"],
        df["Drop_Latitude"],
        df["Drop_Longitude"],
    )

    # ── 2. order_hour (0-23) ──────────────────────────────────────────────
    order_mins = _time_to_minutes(df["Order_Time"])
    df["order_hour"] = (order_mins // 60).astype("Int64")

    # ── 3. day_of_week (0=Mon … 6=Sun) ───────────────────────────────────
    df["day_of_week"] = pd.to_datetime(
        df["Order_Date"], errors="coerce"
    ).dt.dayofweek

    # ── 4. is_weekend ─────────────────────────────────────────────────────
    df["is_weekend"] = df["day_of_week"].isin(_WEEKEND_DAYS).astype(int)

    # ── 5. is_peak ────────────────────────────────────────────────────────
    df["is_peak"] = df["order_hour"].isin(_PEAK_HOURS).astype(int)

    # ── 6. prep_time_min (handles midnight crossing) ──────────────────────
    order_mins_raw = _time_to_minutes(df["Order_Time"])
    pickup_mins_raw = _time_to_minutes(df["Pickup_Time"])
    # modulo 1440 ensures negative differences (crossing midnight) wrap correctly
    df["prep_time_min"] = (pickup_mins_raw - order_mins_raw) % 1440

    # ── 7. Extract target before dropping ─────────────────────────────────
    y: pd.Series = df[TARGET_COL].copy()

    # ── 8. Drop raw columns + target ──────────────────────────────────────
    drop_cols = _COLS_TO_DROP + [TARGET_COL]
    X: pd.DataFrame = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Cast Int64 -> float so sklearn pipelines don't choke on nullable int
    for col in ["order_hour", "day_of_week", "is_weekend", "is_peak"]:
        if col in X.columns:
            X[col] = X[col].astype(float)

    return X, y


# ---------------------------------------------------------------------------
# CLI convenience: run feature engineering on cleaned data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import CLEANED_CSV, PROCESSED_DIR  # noqa: F401

    raw_clean = pd.read_csv(CLEANED_CSV)
    X, y = build_features(raw_clean)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("\nFeature columns:\n", X.columns.tolist())
    print("\nSample X:\n", X.head(3).to_string())
    out = X.copy()
    out[TARGET_COL] = y.values
    out_path = PROCESSED_DIR / "features.csv"
    out.to_csv(out_path, index=False)
    print(f"\nFeatures saved -> {out_path}")
