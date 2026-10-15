"""
src/config.py
=============
Central configuration: project-root-relative paths and global constants.
All paths are pathlib.Path objects derived from this file's location so the
project is portable across machines.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root anchors
# ---------------------------------------------------------------------------
# This file lives at  <project_root>/src/config.py
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

RAW_CSV: Path = RAW_DIR / "amazon_delivery.csv"
CLEANED_CSV: Path = PROCESSED_DIR / "amazon_delivery_cleaned.csv"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
APP_DIR: Path = PROJECT_ROOT / "app"

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TARGET_COL: str = "Delivery_Time"
TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# Cleaning thresholds
# ---------------------------------------------------------------------------
AGENT_AGE_MIN: int = 18
AGENT_AGE_MAX: int = 65
AGENT_RATING_MIN: float = 1.0
AGENT_RATING_MAX: float = 5.0
IQR_MULTIPLIER: float = 1.5
FIGURE_DPI: int = 200

# ---------------------------------------------------------------------------
# Ensure output directories exist at import time
# ---------------------------------------------------------------------------
for _dir in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
