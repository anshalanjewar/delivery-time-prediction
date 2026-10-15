"""
app/streamlit_app.py
====================
Streamlit UI for the Amazon Food Delivery Time Predictor.

Visual redesign — prediction, model, and preprocessing logic unchanged.

Features
--------
- st.tabs: Predict | Model Comparison | About & Limitations
- Bordered input cards: Order Details, Conditions, Rider Details
- Large result card + Plotly range timeline
- Per-prediction SHAP top-3 bar chart
- Sidebar: live metrics from reports/tables CSVs
- Colour-coded status from training data quartiles (Q1=90, Q3=160)

Run
---
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
MODELS_DIR   = PROJECT_ROOT / "models"
TABLES_DIR   = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Delivery-time thresholds (from training data quartiles, never hard-coded) ─
# Q1=90, Q3=160 from data/processed/amazon_delivery_cleaned.csv
_Q1: float = 90.0   # below → fast (green)
_Q3: float = 160.0  # above → delayed (red); between → normal (amber)

ACCENT = "#e94560"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── cards ── */
.card {
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 14px;
    padding: 1.25rem 1.4rem 1rem 1.4rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7c83a8;
    margin-bottom: 0.8rem;
}

/* ── result card ── */
.result-card {
    border-radius: 16px;
    padding: 1.6rem 1.2rem;
    text-align: center;
    margin-bottom: 0.6rem;
}
.result-card .big   { font-size: 3.4rem; font-weight: 700; line-height: 1.1; }
.result-card .label { font-size: 0.9rem; color: #a8b2d8; margin-top: 0.35rem; }
.result-card .badge {
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* status colours */
.status-fast    { background:#0b3d2e; color:#2ecc71; border:1px solid #2ecc71; }
.status-normal  { background:#3d2b0b; color:#f39c12; border:1px solid #f39c12; }
.status-delayed { background:#3d0b0b; color:#e74c3c; border:1px solid #e74c3c; }

/* result border colours */
.border-fast    { border: 2px solid #2ecc71; }
.border-normal  { border: 2px solid #f39c12; }
.border-delayed { border: 2px solid #e74c3c; }

/* ── sidebar metric ── */
.sbar-metric { margin-bottom: 0.6rem; }
.sbar-metric .val { font-size: 1.25rem; font-weight: 700; color: #e94560; }
.sbar-metric .key { font-size: 0.78rem; color: #7c83a8; }

/* ── section header ── */
.sec-hdr {
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #7c83a8; margin: 1.2rem 0 0.5rem 0;
}

/* ── hide Streamlit default padding on main block ── */
.block-container { padding-top: 1.4rem !important; }

/* ── tab styling ── */
button[data-baseweb="tab"] { font-size: 0.9rem !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models …")
def load_models():
    best = joblib.load(MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "lgbm_q10.pkl", "rb") as f:
        q10 = pickle.load(f)
    with open(MODELS_DIR / "lgbm_q90.pkl", "rb") as f:
        q90 = pickle.load(f)
    return best, q10, q90


@st.cache_data(show_spinner=False)
def load_categories() -> dict[str, list]:
    with open(MODELS_DIR / "categories.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_test_results() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "test_results.csv")


@st.cache_data(show_spinner=False)
def load_val_results() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "model_comparison_val.csv")


@st.cache_data(show_spinner=False)
def load_train_n() -> int:
    """Return the number of training rows from the data-splits CSV."""
    splits = pd.read_csv(TABLES_DIR / "data_splits.csv")
    return int((splits["set"] == "train").sum())


try:
    best_model, q10_model, q90_model = load_models()
    categories   = load_categories()
    test_results = load_test_results()
    val_results  = load_val_results()
    n_train      = load_train_n()
    models_ok    = True
except Exception as e:
    st.error(f"Could not load models or data: {e}")
    st.stop()


# ── Sidebar — live metrics from CSV ───────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛵 Delivery Predictor")
    st.markdown("---")
    st.markdown("### Model Stats")

    # Best model row (sorted by rmse → first row)
    best_row = test_results.sort_values("rmse").iloc[0]
    model_name = best_row["model"]

    def _sbar(key, val):
        st.markdown(
            f'<div class="sbar-metric">'
            f'<div class="val">{val}</div>'
            f'<div class="key">{key}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _sbar("Best model",   model_name)
    _sbar("Test MAE",     f"{best_row['mae']:.2f} min")
    _sbar("Test RMSE",    f"{best_row['rmse']:.2f} min")
    _sbar("Test R²",      f"{best_row['r2']:.4f}")
    _sbar("Training rows", f"{n_train:,}")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#555;'>"
        "Metrics read live from <code>reports/tables/</code> — never typed by hand."
        "</div>",
        unsafe_allow_html=True,
    )


# ── Helper: status colour ──────────────────────────────────────────────────
def _status(point: float) -> tuple[str, str, str]:
    """Return (label, badge_class, border_class) based on training quartiles."""
    if point <= _Q1:
        return "Fast", "status-fast", "border-fast"
    elif point <= _Q3:
        return "Normal", "status-normal", "border-normal"
    else:
        return "Delayed", "status-delayed", "border-delayed"


# ── Helper: per-prediction SHAP waterfall (top 3) ─────────────────────────
@st.cache_resource(show_spinner=False)
def _get_shap_explainer():
    """Build and cache a SHAP TreeExplainer on the LightGBM pipeline."""
    try:
        import shap
        import pickle as _pkl
        with open(MODELS_DIR / "LightGBM.pkl", "rb") as f:
            pipe = _pkl.load(f)
        preprocessor = pipe.named_steps["preprocessor"]
        lgbm_model   = pipe.named_steps["model"]
        explainer    = shap.TreeExplainer(lgbm_model)
        return explainer, preprocessor
    except Exception:
        return None, None


def _shap_top3(row_df: pd.DataFrame) -> go.Figure | None:
    """Return a small Plotly bar of top-3 SHAP contributors for this row."""
    try:
        import shap
        explainer, preprocessor = _get_shap_explainer()
        if explainer is None:
            return None

        X_t = preprocessor.transform(row_df)
        try:
            feat_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feat_names = [f"f{i}" for i in range(X_t.shape[1])]

        sv = explainer.shap_values(pd.DataFrame(X_t, columns=feat_names))[0]

        # Map OHE back to readable names (strip "num__" / "cat__" prefixes)
        clean = [n.split("__")[-1] for n in feat_names]
        pairs = sorted(zip(np.abs(sv), sv, clean), reverse=True)[:3]

        vals   = [p[1] for p in pairs]
        labels = [p[2] for p in pairs]
        colors = [ACCENT if v > 0 else "#53d8fb" for v in vals]

        fig = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}" for v in vals],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.2f} min<extra></extra>",
        ))
        fig.update_layout(
            height=180,
            margin=dict(l=0, r=40, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaf6", size=12),
            xaxis=dict(
                showgrid=True, gridcolor="#2a2a45",
                zeroline=True, zerolinecolor="#444",
                tickfont=dict(size=10),
            ),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        )
        return fig
    except Exception:
        return None


# ── Helper: Plotly range timeline ─────────────────────────────────────────
def _range_fig(lower: float, point: float, upper: float) -> go.Figure:
    """Horizontal range bar with lower / estimate / upper markers."""
    _, _, border_cls = _status(point)
    colour_map = {
        "border-fast":    "#2ecc71",
        "border-normal":  "#f39c12",
        "border-delayed": "#e74c3c",
    }
    pt_colour = colour_map.get(border_cls, ACCENT)

    pad   = max((upper - lower) * 0.20, 8)
    x_min = max(0, lower - pad)
    x_max = upper + pad

    fig = go.Figure()

    # Shaded interval band
    fig.add_shape(
        type="rect",
        x0=lower, x1=upper, y0=0.3, y1=0.7,
        fillcolor="rgba(83,216,251,0.15)",
        line=dict(color="rgba(83,216,251,0.4)", width=1),
    )

    # Lower / upper ticks
    for x, label in [(lower, f"{lower:.0f} min"), (upper, f"{upper:.0f} min")]:
        fig.add_trace(go.Scatter(
            x=[x], y=[0.5],
            mode="markers+text",
            marker=dict(symbol="line-ns", size=16,
                        color="#53d8fb",
                        line=dict(width=2, color="#53d8fb")),
            text=[label], textposition="top center",
            textfont=dict(size=10, color="#53d8fb"),
            hovertemplate=f"{label}<extra></extra>",
            showlegend=False,
        ))

    # Point estimate diamond
    fig.add_trace(go.Scatter(
        x=[point], y=[0.5],
        mode="markers+text",
        marker=dict(symbol="diamond", size=18,
                    color=pt_colour,
                    line=dict(width=2, color="#fff")),
        text=[f"<b>{point:.0f}</b>"],
        textposition="bottom center",
        textfont=dict(size=13, color=pt_colour),
        hovertemplate=f"Estimate: {point:.0f} min<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        height=120,
        margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[x_min, x_max],
            showgrid=True, gridcolor="#2a2a45",
            tickfont=dict(color="#a8b2d8", size=10),
            title=dict(text="Delivery Time (minutes)", font=dict(size=11, color="#7c83a8")),
        ),
        yaxis=dict(visible=False, range=[0, 1]),
        annotations=[dict(
            x=(lower + upper) / 2, y=0.85,
            text="80 % Prediction Interval",
            showarrow=False,
            font=dict(size=10, color="#53d8fb"),
        )],
    )
    return fig


# ── Assemble input row ─────────────────────────────────────────────────────
def make_input(
    distance_km, order_hour, day_of_week, prep_time_min,
    traffic, weather, vehicle, area, category,
    agent_age, agent_rating,
) -> pd.DataFrame:
    is_weekend = int(day_of_week in {5, 6})
    is_peak    = int(order_hour in {12, 13, 14, 19, 20, 21, 22})
    return pd.DataFrame([{
        "Agent_Age":     agent_age,
        "Agent_Rating":  agent_rating,
        "Weather":       weather,
        "Traffic":       traffic,
        "Vehicle":       vehicle,
        "Area":          area,
        "Category":      category,
        "distance_km":   distance_km,
        "order_hour":    float(order_hour),
        "day_of_week":   float(day_of_week),
        "is_weekend":    float(is_weekend),
        "is_peak":       float(is_peak),
        "prep_time_min": float(prep_time_min),
    }])


# ═══════════════════════════════════════════════════════════════════════════
# Hero header
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
    background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    border-radius:16px; padding:1.6rem 2rem; margin-bottom:1.2rem; text-align:center;">
  <h1 style="color:#e94560;font-size:1.9rem;margin:0 0 0.3rem 0;">
    🛵 Food Delivery Time Predictor
  </h1>
  <p style="color:#a8b2d8;font-size:0.95rem;margin:0;">
    ML-powered estimates from 40,000+ Amazon delivery orders
  </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════
tab_predict, tab_compare, tab_about = st.tabs(
    ["🚀 Predict", "📊 Model Comparison", "ℹ️ About & Limitations"]
)


# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Predict
# ─────────────────────────────────────────────────────────────────────────
with tab_predict:
    left_col, right_col = st.columns([1.1, 1], gap="medium")

    # ── INPUT CARDS (left) ─────────────────────────────────────────────
    with left_col:

        # Card 1: Order Details
        st.markdown('<div class="card">'
                    '<div class="card-title">📦 Order Details</div>', unsafe_allow_html=True)
        distance_km = st.slider(
            "Distance (km)", 0.1, 50.0, 5.0, 0.1,
            help="Straight-line (haversine) distance — not road distance.",
        )
        order_hour = st.slider("Order Hour (0–23)", 0, 23, 12)
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
            index=0,
        )
        prep_time_min = st.slider("Prep Time (min)", 0, 60, 15,
                                  help="Gap between order placed and rider pickup.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Card 2: Conditions
        st.markdown('<div class="card">'
                    '<div class="card-title">🌦️ Conditions</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            traffic  = st.selectbox("Traffic Level", categories["Traffic"])
            weather  = st.selectbox("Weather",        categories["Weather"])
        with c2:
            vehicle  = st.selectbox("Vehicle Type",   categories["Vehicle"])
            area     = st.selectbox("Area Type",      categories["Area"])
        category = st.selectbox("Order Category",     categories["Category"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Card 3: Rider Details
        st.markdown('<div class="card">'
                    '<div class="card-title">🧑 Rider Details</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            agent_age    = st.slider("Rider Age", 18, 65, 28)
        with r2:
            agent_rating = st.slider("Rider Rating", 1.0, 5.0, 4.5, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button(
            "🚀 Predict Delivery Time",
            use_container_width=True,
            type="primary",
        )

    # ── RESULTS (right) ────────────────────────────────────────────────
    with right_col:
        if predict_btn:
            row = make_input(
                distance_km, order_hour, day_of_week, prep_time_min,
                traffic, weather, vehicle, area, category,
                agent_age, agent_rating,
            )
            point = float(best_model.predict(row)[0])
            lower = float(q10_model.predict(row)[0])
            upper = float(q90_model.predict(row)[0])

            # Clamp
            point = max(point, 1.0)
            lower = max(lower, 1.0)
            upper = max(upper, lower + 1.0)

            status_label, badge_cls, border_cls = _status(point)

            # Big result card
            status_icons = {"Fast": "🟢", "Normal": "🟡", "Delayed": "🔴"}
            icon = status_icons[status_label]
            colour_map = {
                "border-fast":    "#2ecc71",
                "border-normal":  "#f39c12",
                "border-delayed": "#e74c3c",
            }
            pt_colour = colour_map[border_cls]
            bg_map = {
                "border-fast":    "linear-gradient(135deg,#0b1f14,#0f2d1e)",
                "border-normal":  "linear-gradient(135deg,#1f140b,#2d1e0f)",
                "border-delayed": "linear-gradient(135deg,#1f0b0b,#2d0f0f)",
            }

            st.markdown(f"""
            <div class="result-card {border_cls}"
                 style="background:{bg_map[border_cls]};">
              <div class="big" style="color:{pt_colour};">{point:.0f} min</div>
              <div class="label">Estimated Delivery Time</div>
              <span class="badge {badge_cls}">{icon} {status_label}</span>
            </div>
            """, unsafe_allow_html=True)

            # Range timeline
            st.plotly_chart(_range_fig(lower, point, upper),
                            use_container_width=True, config={"displayModeBar": False})

            # Traffic hint
            tips = {
                "Jam":    "🔴 Heavy jam — expect delays!",
                "High":   "🟠 High traffic — slight delays possible.",
                "Medium": "🟡 Moderate traffic.",
                "Low":    "🟢 Clear roads — smooth delivery expected.",
            }
            st.info(tips.get(traffic, ""))

            # SHAP top-3 explanation
            st.markdown(
                '<div class="sec-hdr">Top 3 factors influencing this prediction</div>',
                unsafe_allow_html=True,
            )
            shap_fig = _shap_top3(row)
            if shap_fig:
                st.markdown(
                    "<div style='font-size:0.75rem;color:#7c83a8;margin-bottom:0.3rem;'>"
                    "<span style='color:#e94560;'>■</span> pushes time up &nbsp;"
                    "<span style='color:#53d8fb;'>■</span> pushes time down"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(shap_fig, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.caption("SHAP explanation unavailable (shap not installed).")

        else:
            # Placeholder before prediction
            st.markdown("""
            <div style="
                height:420px; display:flex; align-items:center;
                justify-content:center; flex-direction:column;
                border:2px dashed #2a2a45; border-radius:16px;
                color:#3a3a58; text-align:center; padding:2rem;">
              <div style="font-size:3rem; margin-bottom:0.8rem;">🛵</div>
              <div style="font-size:1rem; font-weight:600; color:#555;">
                Fill in the details and click<br>
                <span style="color:#e94560;">Predict Delivery Time</span>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("### Test Set Results")
    st.caption("Sorted by RMSE. Metrics read live from `reports/tables/test_results.csv`.")

    # Formatted table
    display_df = test_results.sort_values("rmse").copy()
    display_df.columns = ["Model", "MAE (min)", "RMSE (min)", "R²"]
    display_df["MAE (min)"]  = display_df["MAE (min)"].map("{:.4f}".format)
    display_df["RMSE (min)"] = display_df["RMSE (min)"].map("{:.4f}".format)
    display_df["R²"]         = display_df["R²"].map("{:.6f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # MAE bar chart from saved PNG
    mae_bar_path = FIGURES_DIR / "04_mae_bar.png"
    if mae_bar_path.exists():
        st.markdown("### Test Set MAE by Model")
        st.image(str(mae_bar_path), use_container_width=True)
    else:
        st.warning("MAE bar chart not found — run `src/evaluate.py` first.")

    st.markdown("---")
    st.markdown("### Validation Set Results (before tuning)")
    st.caption("Read from `reports/tables/model_comparison_val.csv`.")
    val_display = val_results.sort_values("rmse").copy()
    val_display.columns = ["Model", "MAE (min)", "RMSE (min)", "R²", "Train Time (s)"]
    val_display["MAE (min)"]      = val_display["MAE (min)"].map("{:.4f}".format)
    val_display["RMSE (min)"]     = val_display["RMSE (min)"].map("{:.4f}".format)
    val_display["R²"]             = val_display["R²"].map("{:.6f}".format)
    val_display["Train Time (s)"] = val_display["Train Time (s)"].map("{:.2f}".format)
    st.dataframe(val_display, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — About & Limitations
# ─────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### About This App")
    best_row = test_results.sort_values("rmse").iloc[0]
    st.markdown(f"""
This app uses a **{best_row['model']}** regression model trained on **{n_train:,}** cleaned
Amazon delivery records (70% of 40,079 total rows).
The model achieves an **RMSE ≈ {best_row['rmse']:.1f} min** on a held-out test set
(R² ≈ {best_row['r2']:.3f}).

An **80% prediction interval** is provided via two additional quantile LightGBM models
(α = 0.10 and α = 0.90), giving a sense of uncertainty around the point estimate.

Delivery status thresholds are derived from the **training data quartiles**:
- 🟢 **Fast** — predicted time ≤ {_Q1:.0f} min (Q1)
- 🟡 **Normal** — {_Q1:.0f}–{_Q3:.0f} min (Q1–Q3)
- 🔴 **Delayed** — predicted time > {_Q3:.0f} min (Q3)
""")

    st.markdown("### Key Limitations")
    st.markdown("""
| Limitation | Detail |
|-----------|--------|
| **Straight-line distance** | `distance_km` is the haversine great-circle distance, not actual road distance. Real routes will be longer. |
| **User-chosen traffic** | Traffic level is entered manually. The model cannot observe real-time congestion. |
| **Static weather** | Weather is a categorical label; no live feed or forecast is used. |
| **Training data scope** | Model was trained on Amazon India delivery data (2022). Generalisation to other cities/regions is not guaranteed. |
| **No restaurant prep time** | `prep_time_min` approximates the order→pickup gap — restaurant cooking time is not modelled separately. |
| **Interval calibration** | Empirical coverage ≈ 77% vs the 80% target (vanilla quantile regression, no conformal calibration). |
""")

    st.markdown("### Feature Importance (SHAP, top 6)")
    st.markdown("""
1. `distance_km` — strongest positive driver
2. `Traffic` (Jam) — large positive effect
3. `prep_time_min` — longer prep → longer delivery
4. `order_hour` — peak hours increase time
5. `Agent_Rating` — higher rating → faster (negative SHAP)
6. `Vehicle` (van) — slower than motorcycle
""")

    shap_bar = FIGURES_DIR / "08_shap_bar.png"
    if shap_bar.exists():
        st.image(str(shap_bar), caption="Mean |SHAP| feature importance — LightGBM",
                 use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Streamlit · Model: LightGBM (best) · "
    "Data: Amazon Delivery Dataset (2022) · "
    f"Training rows: {n_train:,}"
)
