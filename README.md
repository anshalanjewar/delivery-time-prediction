# Food Delivery Time Prediction

A regression project that predicts **Delivery_Time** (minutes) for Amazon delivery orders using rider, weather, traffic, and location features. The best model achieves **RMSE ≈ 22.3 min** and **R² ≈ 0.817** on the held-out test set.

---

## Project Structure

```
ML_Project/
├── data/
│   ├── raw/                        # Original CSV (amazon_delivery.csv, 43 739 rows)
│   └── processed/                  # Cleaned CSV (40 079 rows after cleaning)
├── notebooks/
│   └── 01_eda.ipynb                # Fully executed EDA notebook
├── src/
│   ├── config.py                   # Pathlib paths + constants (seed 42)
│   ├── cleaning.py                 # clean(df) — 7-step pipeline
│   ├── features.py                 # build_features(df) -> (X, y)
│   ├── pipeline.py                 # ColumnTransformer + sklearn Pipeline
│   ├── train.py                    # 7-model training + validation metrics
│   ├── tune.py                     # RandomizedSearchCV (LightGBM + XGBoost)
│   ├── evaluate.py                 # Test-set evaluation + figures
│   ├── explain.py                  # SHAP summary & bar plots
│   └── intervals.py                # LightGBM quantile regression (80% PI)
├── app/
│   └── streamlit_app.py            # Interactive prediction UI
├── models/
│   ├── best_model.joblib           # Best pipeline (LightGBM tuned)
│   ├── categories.json             # Dropdown option lists
│   ├── lgbm_q10.pkl / lgbm_q90.pkl # Quantile models for 80% interval
│   └── *.pkl                       # All trained pipelines
├── reports/
│   ├── figures/                    # 8 PNG plots @ DPI 200
│   └── tables/                     # CSV result tables
├── requirements.txt
└── README.md
```

---

## Dataset

| Field | Detail |
|-------|--------|
| Source | `data/raw/amazon_delivery.csv` |
| Raw rows | 43,739 |
| Columns | 16 |
| Target | `Delivery_Time` (int, minutes, mean ≈ 124.9, std ≈ 51.9) |

**Columns:** `Order_ID`, `Agent_Age`, `Agent_Rating`, `Store_Latitude`, `Store_Longitude`, `Drop_Latitude`, `Drop_Longitude`, `Order_Date`, `Order_Time`, `Pickup_Time`, `Weather`, `Traffic`, `Vehicle`, `Area`, `Delivery_Time`, `Category`

---

## Cleaning Summary (`src/cleaning.py`)

| Step | Rows Before | Rows After | Dropped |
|------|------------|-----------|---------|
| Raw input | 43,739 | 43,739 | 0 |
| Drop duplicates | 43,739 | 43,739 | 0 |
| Strip whitespace | 43,739 | 43,739 | 0 |
| NaN strings → real NaN | 43,739 | 43,739 | 0 |
| Agent_Age outside [18, 65] | 43,739 | 43,701 | **38** |
| Agent_Rating outside [1, 5] | 43,701 | 43,648 | **53** |
| Zero coordinates | 43,648 | 40,153 | **3,495** |
| Delivery_Time IQR outliers | 40,153 | 40,079 | **74** |
| **Final** | — | **40,079** | **3,660 (8.4%)** |

---

## Feature Engineering (`src/features.py`)

| Feature | Description |
|---------|------------|
| `distance_km` | Haversine great-circle distance (store → drop) |
| `order_hour` | Hour of day (0–23) from `Order_Time` |
| `day_of_week` | 0=Mon … 6=Sun from `Order_Date` |
| `is_weekend` | 1 if Sat/Sun |
| `is_peak` | 1 if hour ∈ {12,13,14,19,20,21,22} |
| `prep_time_min` | Order→Pickup gap (mod-1440 for midnight crossing) |
| `Agent_Age` | Rider age (kept as-is) |
| `Agent_Rating` | Rider rating 1–5 (kept as-is) |
| `Weather` | Categorical: Sunny / Cloudy / Stormy / Fog / Windy / Sandstorms |
| `Traffic` | Categorical: Low / Medium / High / Jam |
| `Vehicle` | Categorical: motorcycle / scooter / van / bicycle |
| `Area` | Categorical: Urban / Metropolitian / Semi-Urban / Other |
| `Category` | Categorical: 16 product categories |

**Split:** 70% train (28,055) / 15% val (6,012) / 15% test (6,012) — `random_state=42`

---

## Preprocessing (`src/pipeline.py`)

```
ColumnTransformer
├── Numeric  → MedianImputer  → StandardScaler
└── Categorical → MostFrequentImputer → OneHotEncoder(handle_unknown="ignore")
```

Fitted on **training data only** — no leakage.

---

## Model Results

### Validation Set (before tuning)

| Model | MAE | RMSE | R² | Train Time |
|-------|-----|------|----|-----------|
| LightGBM | 17.23 | 22.13 | 0.8184 | 1.3 s |
| XGBoost | 17.23 | 22.18 | 0.8177 | 1.8 s |
| RandomForest | 17.24 | 22.44 | 0.8133 | 5.1 s |
| DecisionTree | 18.48 | 24.03 | 0.7859 | 0.5 s |
| MLPRegressor | 21.20 | 27.59 | 0.7177 | 42.7 s |
| LinearRegression | 25.95 | 32.92 | 0.5983 | 0.3 s |
| KNeighbors | 27.15 | 35.10 | 0.5432 | 0.2 s |

### Tuning Comparison — Validation Set (`RandomizedSearchCV`, n_iter=40, cv=3)

| Model | Variant | MAE | RMSE | R² |
|-------|---------|-----|------|----|
| LightGBM | default | 17.23 | 22.13 | 0.8184 |
| **LightGBM** | **tuned** | **16.88** | **21.82** | **0.8236** |
| XGBoost | default | 17.23 | 22.18 | 0.8177 |
| XGBoost | tuned | 16.90 | 21.83 | 0.8233 |

### Test Set (final, held-out)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **LightGBM (tuned)** | **17.35** | **22.33** | **0.8171** |
| XGBoost (tuned) | 17.33 | 22.37 | 0.8164 |
| RandomForest | 17.37 | 22.56 | 0.8132 |
| DecisionTree | 18.75 | 24.22 | 0.7849 |
| MLPRegressor | 21.54 | 28.15 | 0.7093 |
| LinearRegression | 26.02 | 33.19 | 0.5958 |
| KNeighbors | 27.40 | 35.25 | 0.5441 |

> No model exceeded R² > 0.99 — no data leakage detected.

---

## Prediction Intervals (`src/intervals.py`)

Two LightGBM quantile models (α = 0.10 and α = 0.90) provide an **80% prediction interval**.

| Target Coverage | Empirical Coverage | Avg Width | Median Width | Test n |
|----------------|-------------------|-----------|-------------|--------|
| 80% | **77.2%** | 55.6 min | 54.4 min | 6,012 |

Slight under-coverage (77% vs 80%) is expected from vanilla quantile regression without conformal calibration.

---

## Explainability (`src/explain.py`)

SHAP TreeExplainer on 1,000 test rows reveals top features:
1. `distance_km` — strongest positive driver
2. `Traffic` (Jam) — large positive effect
3. `prep_time_min` — longer prep = longer delivery
4. `order_hour` — peak hours increase time
5. `Agent_Rating` — higher rating → faster delivery (negative SHAP)
6. `Vehicle` (van) — slower than motorcycle

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clean data
python src/cleaning.py

# 3. Train all models
python src/train.py

# 4. Tune best two models
python src/tune.py

# 5. Test-set evaluation + figures
python src/evaluate.py

# 6. SHAP explanations
python src/explain.py

# 7. Prediction intervals
python src/intervals.py

# 8. Launch the app
streamlit run app/streamlit_app.py
```

---

## App Features

- Sliders for distance, prep time, order hour, rider age & rating
- Dropdowns for traffic, weather, vehicle, area, category (options from `models/categories.json`)
- Point estimate + 80% prediction interval displayed
- Traffic-level hint card
- About section with model limitations

---

## Limitations

| Limitation | Detail |
|-----------|--------|
| Straight-line distance | Haversine, not road distance |
| User-chosen traffic | No real-time congestion feed |
| Static weather labels | No live forecast |
| Training scope | Amazon India, 2022 data |
| Interval calibration | Vanilla quantile regression (77% vs 80% target) |

---

## Random Seed

All stochastic steps use `RANDOM_SEED = 42` from `src/config.py`.
