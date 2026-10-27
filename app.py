from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ---------------------------------------------------------------------
# Power Plant Energy Prediction | ANN Regression
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "powerplant_data.csv"
MODEL_PATH = BASE_DIR / "best_model.pt"
SCALER_PATH = BASE_DIR / "scaler.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"
FEATURE_LABELS = {
    "AT": ("Ambient Temperature", "°C"),
    "V": ("Exhaust Vacuum", "cm Hg"),
    "AP": ("Ambient Pressure", "mbar"),
    "RH": ("Relative Humidity", "%"),
}

st.set_page_config(
    page_title="PowerPulse AI | Energy Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#172033; --muted:#64748b; --blue:#4f7cff; --teal:#22c7a9; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #f6f8ff 0%, #f7fbff 48%, #f1fcfa 100%); }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
.hero {
  padding: 2rem 2.1rem; border-radius: 24px;
  background: linear-gradient(120deg, #14213d 0%, #243d73 55%, #167c80 100%);
  color: white; box-shadow: 0 18px 45px rgba(25, 45, 90, .16);
  margin-bottom: 1.4rem;
}
.hero h1 { color: white !important; font-size: clamp(2rem, 4vw, 3rem); margin: .25rem 0 .6rem 0; }
.hero p { color: #e3edff; font-size: 1.05rem; max-width: 760px; margin-bottom: .4rem; }
.eyebrow { color: #70f0d5; font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.metric-card {
  background: rgba(255,255,255,.88); border: 1px solid #e3eafa;
  border-radius: 18px; padding: 1rem 1.2rem; min-height: 112px;
  box-shadow: 0 8px 24px rgba(30, 50, 90, .045);
}
.metric-label { color: #65748b; font-size: .85rem; font-weight: 600; }
.metric-value { color: #15213a; font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; margin-top: .25rem; }
.panel {
  background: rgba(255,255,255,.9); border: 1px solid #e4eaf5;
  border-radius: 20px; padding: 1.3rem 1.4rem; margin-bottom: 1rem;
}
.small-note { color: #64748b; font-size: .88rem; }
.stButton > button[kind="primary"] { border-radius: 12px; font-weight: 700; min-height: 3rem; }
div[data-testid="stSidebar"] { background: #f1f5ff; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


class ANN(nn.Module):
    """Architecture matching the trained checkpoint."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        return self.model(x)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


@st.cache_resource
def load_predictor():
    model = ANN()
    # weights_only is supported in recent PyTorch versions. Checkpoint is a state_dict.
    try:
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with open(SCALER_PATH, "rb") as file:
        fitted_scaler = pickle.load(file)
    return model, fitted_scaler


def predict_one(values, model, fitted_scaler):
    row = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)
    scaled = fitted_scaler.transform(row)
    with torch.no_grad():
        result = model(torch.tensor(scaled, dtype=torch.float32)).item()
    return float(result)


df = load_data()
model, fitted_scaler = load_predictor()

# Sidebar
with st.sidebar:
    st.markdown("## ⚡ PowerPulse AI")
    st.caption("ANN-powered energy prediction")
    st.divider()
    page = st.radio("NAVIGATION", ["Predict Energy", "Explore Dataset", "Model Insights"], label_visibility="visible")
    st.divider()
    st.markdown("**Model stack**")
    st.markdown("- PyTorch · Artificial Neural Network\n- scikit-learn · StandardScaler\n- Streamlit · Web application")
    st.caption("Educational project • Predictions are estimates.")

# Hero
st.markdown("""
<div class="hero">
  <div class="eyebrow">DEEP LEARNING · REGRESSION</div>
  <h1>PowerPulse AI</h1>
  <p>Estimate power plant net electrical output from environmental and operational conditions using a trained Artificial Neural Network.</p>
  <div style="margin-top:1rem;color:#d4e4ff;font-size:.88rem;">4 model inputs&nbsp;&nbsp; • &nbsp;&nbsp;PyTorch ANN&nbsp;&nbsp; • &nbsp;&nbsp;Interactive analytics</div>
</div>
""", unsafe_allow_html=True)

if page == "Predict Energy":
    left, right = st.columns([1.03, .97], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Configure operating conditions")
        st.write("Adjust the input values to estimate net electrical energy output.")
        values = {}
        cols = st.columns(2)
        for i, feature in enumerate(FEATURES):
            label, unit = FEATURE_LABELS[feature]
            min_value = float(df[feature].min())
            max_value = float(df[feature].max())
            default = float(df[feature].median())
            with cols[i % 2]:
                values[feature] = st.slider(
                    f"{label} ({unit})",
                    min_value=min_value,
                    max_value=max_value,
                    value=default,
                    step=0.01,
                    help=f"Observed dataset range: {min_value:.2f}–{max_value:.2f} {unit}",
                )
        predict_clicked = st.button("⚡ Predict net output", type="primary", use_container_width=True)
        st.markdown('<p class="small-note">For reliable use, keep inputs within the observed dataset ranges.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Prediction result")
        prediction = predict_one(values, model, fitted_scaler)
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#e9f1ff,#e5fbf5);border:1px solid #dce8f5;border-radius:18px;padding:1.5rem;margin:.7rem 0 1rem 0;">'
            f'<div style="color:#526782;font-weight:600;font-size:.9rem;">Estimated net output</div>'
            f'<div style="font-family:Space Grotesk,sans-serif;font-size:2.8rem;font-weight:700;color:#16284b;">{prediction:,.2f} <span style="font-size:1rem;font-weight:600;">MW</span></div>'
            f'<div style="color:#54716f;font-size:.85rem;">ANN regression estimate</div></div>',
            unsafe_allow_html=True,
        )
        st.write("**Input snapshot**")
        snapshot = pd.DataFrame([
            {"Parameter": FEATURE_LABELS[k][0], "Value": f"{v:.2f} {FEATURE_LABELS[k][1]}"}
            for k, v in values.items()
        ])
        st.dataframe(snapshot, hide_index=True, use_container_width=True)
        st.download_button(
            "Download prediction report (CSV)",
            data=pd.DataFrame([{**values, "Predicted_PE": prediction}]).to_csv(index=False).encode("utf-8"),
            file_name="powerpulse_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    st.info("This is an educational machine-learning demo, not a certified industrial control or safety system.")

elif page == "Explore Dataset":
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    with a:
        st.markdown(f'<div class="metric-card"><div class="metric-label">OBSERVATIONS</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="metric-card"><div class="metric-label">INPUT FEATURES</div><div class="metric-value">{len(FEATURES)}</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown(f'<div class="metric-card"><div class="metric-label">MISSING VALUES</div><div class="metric-value">{int(df.isna().sum().sum())}</div></div>', unsafe_allow_html=True)
    st.markdown("")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Net output distribution")
        st.bar_chart(df[TARGET].round(0).value_counts().sort_index(), height=300)
        st.markdown('</div>', unsafe_allow_html=True)
    with chart_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Feature correlation")
        st.dataframe(df[FEATURES + [TARGET]].corr().round(3), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("#### Data preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    st.download_button("Download source dataset", df.to_csv(index=False).encode("utf-8"), "powerplant_data.csv", "text/csv")

else:
    st.subheader("Model insights")
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">R² SCORE</div><div class="metric-value">{metrics["r2"]:.3f}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">MAE</div><div class="metric-value">{metrics["mae"]:.2f}</div><div class="small-note">MW</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">RMSE</div><div class="metric-value">{metrics["rmse"]:.2f}</div><div class="small-note">MW</div></div>', unsafe_allow_html=True)
        st.caption("Metrics are calculated on a held-out 20% test split using random_state=42 and the supplied trained checkpoint.")
    except (OSError, KeyError, ValueError):
        st.warning("Model metrics file could not be loaded.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### ANN architecture")
    st.code("4 input features → Dense(6) + ReLU → Dense(6) + ReLU → Dense(1 output)", language="text")
    st.markdown("""
**Pipeline**
1. Split data into training and test sets (80/20).
2. Fit `StandardScaler` on training inputs only.
3. Transform the input features with the fitted scaler.
4. Pass scaled values through the trained PyTorch ANN.
5. Return the predicted net electrical output.
""")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown(
    '<div style="text-align:center;color:#728097;font-size:.85rem;">PowerPulse AI · ANN Regression Project · Built with PyTorch and Streamlit</div>',
    unsafe_allow_html=True,
)
