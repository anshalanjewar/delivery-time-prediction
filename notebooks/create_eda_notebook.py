"""
notebooks/create_eda_notebook.py
=================================
Script to generate 01_eda.ipynb programmatically with nbformat,
then execute it with nbconvert.
Run from project root:  python notebooks/create_eda_notebook.py
"""

import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import nbformat  # noqa: E402
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook  # noqa: E402

# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------
cells = []

# ── Cell 0: title ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""# 01 — Exploratory Data Analysis
### Amazon Food Delivery Time Prediction
Target: `Delivery_Time` (minutes)
"""))

# ── Cell 1: imports & config ───────────────────────────────────────────────
cells.append(new_code_cell("""\
import sys
from pathlib import Path

# Make src importable from notebooks/
sys.path.insert(0, str(Path.cwd().parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import (
    RAW_CSV, FIGURES_DIR, TARGET_COL, FIGURE_DPI
)

# ── plot style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": FIGURE_DPI, "savefig.dpi": FIGURE_DPI})

print("Libraries loaded ✓")
print(f"Figures will be saved to: {FIGURES_DIR}")
"""))

# ── Cell 2: load & basic info ──────────────────────────────────────────────
cells.append(new_markdown_cell("## 1 · Load Data"))

cells.append(new_code_cell("""\
df = pd.read_csv(RAW_CSV)

print("=" * 55)
print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print("=" * 55)

print("\\n── dtypes ──")
print(df.dtypes.to_string())
"""))

# ── Cell 3: missing values ─────────────────────────────────────────────────
cells.append(new_markdown_cell("## 2 · Missing Values"))

cells.append(new_code_cell("""\
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
missing_df = missing_df[missing_df["missing_count"] > 0]
print("Columns with missing values:")
print(missing_df.to_string())
"""))

# ── Cell 4: duplicates ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 3 · Duplicate Rows"))

cells.append(new_code_cell("""\
n_dupes = df.duplicated().sum()
print(f"Exact duplicate rows: {n_dupes:,}")
"""))

# ── Cell 5: value counts ───────────────────────────────────────────────────
cells.append(new_markdown_cell("## 4 · Categorical Value Counts"))

cells.append(new_code_cell("""\
cat_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]

for col in cat_cols:
    # Strip whitespace for display (raw data has trailing spaces)
    vc = df[col].str.strip().value_counts(dropna=False)
    print(f"\\n{'─'*40}")
    print(f"  {col}  (unique = {vc.shape[0]})")
    print(f"{'─'*40}")
    print(vc.to_string())
"""))

# ── Cell 6: target distribution ────────────────────────────────────────────
cells.append(new_markdown_cell("## 5 · Target Distribution: `Delivery_Time`"))

cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# histogram + KDE
axes[0].hist(
    df[TARGET_COL], bins=40, color="#4C72B0", edgecolor="white",
    linewidth=0.4, alpha=0.85
)
axes[0].set_xlabel("Delivery_Time (min)", fontsize=12)
axes[0].set_ylabel("Count", fontsize=12)
axes[0].set_title("Distribution of Delivery_Time", fontsize=13, fontweight="bold")

# add stats box
stats = df[TARGET_COL].describe()
textstr = (
    f"mean  = {stats['mean']:.1f}\\n"
    f"median= {stats['50%']:.1f}\\n"
    f"std   = {stats['std']:.1f}\\n"
    f"min   = {stats['min']:.0f}\\n"
    f"max   = {stats['max']:.0f}"
)
axes[0].text(
    0.97, 0.97, textstr, transform=axes[0].transAxes,
    fontsize=9, verticalalignment="top", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
)

# Q-Q plot style — box plot to check shape
axes[1].boxplot(
    df[TARGET_COL].dropna(), vert=True,
    patch_artist=True,
    boxprops=dict(facecolor="#4C72B0", alpha=0.7),
    medianprops=dict(color="red", linewidth=2),
)
axes[1].set_ylabel("Delivery_Time (min)", fontsize=12)
axes[1].set_title("Box-Plot of Delivery_Time", fontsize=13, fontweight="bold")
axes[1].set_xticks([])

fig.suptitle("Target Variable: Delivery_Time", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()

save_path = FIGURES_DIR / "01_target_distribution.png"
fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
print(f"Saved → {save_path}")
plt.show()
"""))

# ── Cell 7: boxplot by Traffic ─────────────────────────────────────────────
cells.append(new_markdown_cell("## 6 · Delivery_Time by Traffic Condition"))

cells.append(new_code_cell("""\
# Strip whitespace in a copy
df_plot = df.copy()
df_plot["Traffic"] = df_plot["Traffic"].str.strip()

traffic_order = ["Low", "Medium", "High", "Jam"]
palette = {
    "Low": "#55a868",
    "Medium": "#f5bf2a",
    "High": "#dd8452",
    "Jam": "#c44e52",
}

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(
    data=df_plot,
    x="Traffic",
    y=TARGET_COL,
    order=traffic_order,
    palette=palette,
    width=0.55,
    linewidth=1.2,
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    ax=ax,
)
ax.set_title("Delivery_Time by Traffic Condition", fontsize=14, fontweight="bold")
ax.set_xlabel("Traffic Level", fontsize=12)
ax.set_ylabel("Delivery_Time (min)", fontsize=12)

# Annotate medians
for i, lvl in enumerate(traffic_order):
    med = df_plot.loc[df_plot["Traffic"] == lvl, TARGET_COL].median()
    ax.text(i, med + 4, f"{med:.0f}", ha="center", va="bottom",
            fontsize=9, color="black", fontweight="bold")

plt.tight_layout()
save_path = FIGURES_DIR / "02_boxplot_delivery_by_traffic.png"
fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
print(f"Saved → {save_path}")
plt.show()
"""))

# ── Cell 8: correlation heatmap ────────────────────────────────────────────
cells.append(new_markdown_cell("## 7 · Correlation Heatmap (Numeric Features)"))

cells.append(new_code_cell("""\
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor="white",
    square=True,
    ax=ax,
    annot_kws={"size": 10},
)
ax.set_title("Pearson Correlation – Numeric Columns", fontsize=14, fontweight="bold")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

save_path = FIGURES_DIR / "03_correlation_heatmap.png"
fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
print(f"Saved → {save_path}")
plt.show()
"""))

# ── Cell 9: summary ───────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 8 · EDA Summary"))

cells.append(new_code_cell("""\
print("── EDA Summary ──────────────────────────────────────")
print(f"  Total rows          : {len(df):,}")
print(f"  Total columns       : {df.shape[1]}")
print(f"  Columns w/ missings : {(df.isnull().sum() > 0).sum()}")
print(f"  Duplicate rows      : {df.duplicated().sum():,}")
print(f"  Target mean         : {df[TARGET_COL].mean():.1f} min")
print(f"  Target std          : {df[TARGET_COL].std():.1f} min")
print(f"  Target range        : [{df[TARGET_COL].min()}, {df[TARGET_COL].max()}] min")
print("─────────────────────────────────────────────────────")
"""))

# ---------------------------------------------------------------------------
# Build notebook and write to disk
# ---------------------------------------------------------------------------
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.10.0",
}

out_path = Path(__file__).parent / "01_eda.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written -> {out_path}")
