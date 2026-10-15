"""
src/cleaning.py
===============
Data-cleaning pipeline for the Amazon Delivery dataset.

Entry point
-----------
    clean(df) -> pd.DataFrame

Cleaning steps (in order)
--------------------------
1. Drop exact duplicate rows.
2. Strip leading/trailing whitespace from all object columns.
3. Convert literal "NaN" / "nan" strings to real ``np.nan``.
4. Remove rows where ``Agent_Age`` is outside [AGENT_AGE_MIN, AGENT_AGE_MAX].
5. Remove rows where ``Agent_Rating`` is outside [AGENT_RATING_MIN, AGENT_RATING_MAX].
6. Remove rows where any of the four coordinate columns equals 0.
7. Remove ``Delivery_Time`` outliers using the IQR rule
   (< Q1 - IQR_MULTIPLIER*IQR  or  > Q3 + IQR_MULTIPLIER*IQR).

A before/after row-count summary is saved to
``reports/tables/cleaning_summary.csv``.

Usage
-----
    from src.cleaning import clean
    import pandas as pd
    df_clean = clean(pd.read_csv("data/raw/amazon_delivery.csv"))
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make src/ importable when running the module directly
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    AGENT_AGE_MAX,
    AGENT_AGE_MIN,
    AGENT_RATING_MAX,
    AGENT_RATING_MIN,
    IQR_MULTIPLIER,
    TABLES_DIR,
    TARGET_COL,
)

# Coordinate columns to check for zero values
_COORD_COLS: list[str] = [
    "Store_Latitude",
    "Store_Longitude",
    "Drop_Latitude",
    "Drop_Longitude",
]


def _record(
    records: list[dict[str, int | str]],
    step: str,
    rows_before: int,
    rows_after: int,
) -> None:
    """Append a step record to the audit list.

    Parameters
    ----------
    records:
        Running list of cleaning step dictionaries.
    step:
        Human-readable name of the cleaning step.
    rows_before:
        Row count entering this step.
    rows_after:
        Row count leaving this step.
    """
    records.append(
        {
            "step": step,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_dropped": rows_before - rows_after,
        }
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Amazon Delivery raw DataFrame.

    Parameters
    ----------
    df:
        Raw DataFrame loaded directly from ``amazon_delivery.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with all invalid rows removed.
        Missing values are *not* imputed here.

    Side effects
    ------------
    Saves ``reports/tables/cleaning_summary.csv`` with a row-count audit
    table showing rows dropped at each step.
    """
    records: list[dict[str, int | str]] = []
    n_raw = len(df)

    # -- Step 0: baseline -------------------------------------------------
    _record(records, "0_raw_input", n_raw, n_raw)

    # -- Step 1: Drop exact duplicates ------------------------------------
    df = df.drop_duplicates()
    _record(records, "1_drop_duplicates", n_raw, len(df))

    # -- Step 2: Strip whitespace from all string columns -----------------
    n_before = len(df)
    obj_cols: list[str] = df.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        df[col] = df[col].str.strip()
    _record(records, "2_strip_whitespace", n_before, len(df))

    # -- Step 3: Convert literal "NaN" / "nan" strings to real NaN --------
    n_before = len(df)
    for col in obj_cols:
        df[col] = df[col].replace(
            {"NaN": np.nan, "nan": np.nan, "": np.nan}
        )
    _record(records, "3_nan_strings_to_real_nan", n_before, len(df))

    # -- Step 4: Remove Agent_Age outside [18, 65] ------------------------
    n_before = len(df)
    df = df[df["Agent_Age"].between(AGENT_AGE_MIN, AGENT_AGE_MAX)]
    _record(records, f"4_agent_age_outside_{AGENT_AGE_MIN}_{AGENT_AGE_MAX}", n_before, len(df))

    # -- Step 5: Remove Agent_Rating outside [1, 5] -----------------------
    n_before = len(df)
    df = df[
        df["Agent_Rating"].between(AGENT_RATING_MIN, AGENT_RATING_MAX)
        | df["Agent_Rating"].isna()
    ]
    _record(
        records,
        f"5_agent_rating_outside_{AGENT_RATING_MIN}_{AGENT_RATING_MAX}",
        n_before,
        len(df),
    )

    # -- Step 6: Remove rows with zero coordinates ------------------------
    n_before = len(df)
    coord_mask = (df[_COORD_COLS] == 0).any(axis=1)
    df = df[~coord_mask]
    _record(records, "6_zero_coordinates", n_before, len(df))

    # -- Step 7: Remove Delivery_Time IQR outliers ------------------------
    n_before = len(df)
    q1: float = df[TARGET_COL].quantile(0.25)
    q3: float = df[TARGET_COL].quantile(0.75)
    iqr: float = q3 - q1
    lo: float = q1 - IQR_MULTIPLIER * iqr
    hi: float = q3 + IQR_MULTIPLIER * iqr
    df = df[df[TARGET_COL].between(lo, hi)]
    _record(records, f"7_delivery_time_iqr_outliers_[{lo},{hi}]", n_before, len(df))

    # -- Save audit table -------------------------------------------------
    summary_df = pd.DataFrame(records)
    out_path = TABLES_DIR / "cleaning_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"[cleaning] Cleaning summary saved -> {out_path}")

    n_clean = len(df)
    print(
        f"[cleaning] {n_raw:,} raw rows -> {n_clean:,} clean rows "
        f"({n_raw - n_clean:,} dropped, "
        f"{(n_raw - n_clean) / n_raw * 100:.1f}%)"
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import CLEANED_CSV, RAW_CSV  # noqa: F401

    raw = pd.read_csv(RAW_CSV)
    clean_df = clean(raw)
    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEANED_CSV, index=False)
    print(f"[cleaning] Cleaned CSV saved -> {CLEANED_CSV}")
