import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import io

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
MODEL_PATH = 'aqi_model.pkl'

# Feature names and ranges (must match training order exactly!)
FEATURES = {
    'PM2.5':  (0.0, 500.0, 50.0,  'µg/m³'),
    'PM10':   (0.0, 1000.0, 100.0, 'µg/m³'),
    'NO':     (0.0, 500.0, 20.0,  'ppb'),
    'NO2':    (0.0, 500.0, 30.0,  'ppb'),
    'NOx':    (0.0, 1000.0, 50.0, 'ppb'),
    'NH3':    (0.0, 500.0, 10.0,  'ppb'),
    'CO':     (0.0, 50.0, 1.0,    'mg/m³'),
    'SO2':    (0.0, 500.0, 10.0,  'ppb'),
    'O3':     (0.0, 500.0, 30.0,  'ppb'),
    'Benzene':(0.0, 100.0, 5.0,   'µg/m³'),
    'Toluene':(0.0, 500.0, 10.0,  'µg/m³'),
}

# Quick-load presets so a demo doesn't require manually dragging every slider
PRESETS = {
    "🌿 Good day":       {'PM2.5': 20,  'PM10': 40,  'NO': 5,  'NO2': 10, 'NOx': 15,  'NH3': 5,  'CO': 0.4, 'SO2': 5,  'O3': 20, 'Benzene': 1,  'Toluene': 3},
    "🌤️ Moderate day":   {'PM2.5': 70,  'PM10': 120, 'NO': 20, 'NO2': 35, 'NOx': 55,  'NH3': 15, 'CO': 1.2, 'SO2': 15, 'O3': 40, 'Benzene': 4,  'Toluene': 12},
    "🏭 Polluted day":   {'PM2.5': 180, 'PM10': 320, 'NO': 60, 'NO2': 90, 'NOx': 150, 'NH3': 40, 'CO': 3.5, 'SO2': 60, 'O3': 90, 'Benzene': 15, 'Toluene': 60},
    "🚨 Hazardous day":  {'PM2.5': 400, 'PM10': 800, 'NO': 200,'NO2': 250,'NOx': 500, 'NH3': 200,'CO': 15,  'SO2': 200,'O3': 200,'Benzene': 60, 'Toluene': 250},
}

CATEGORY_INFO = [
    (50,  "Good",                           "#00e400", "✅ Air quality is satisfactory. Enjoy outdoor activities!"),
    (100, "Moderate",                       "#ffff00", "⚠️ Acceptable for most, but sensitive individuals should limit prolonged outdoor exertion."),
    (150, "Unhealthy for Sensitive Groups",  "#ff7e00", "🔶 Children, elderly, and those with respiratory conditions should reduce outdoor activity."),
    (200, "Unhealthy",                      "#ff0000", "🛑 Everyone may experience health effects. Avoid prolonged outdoor exertion."),
    (300, "Very Unhealthy",                 "#8f3f97", "☠️ Health alert! Avoid all outdoor activities."),
    (float('inf'), "Hazardous",             "#7e0023", "🚨 Emergency conditions! Stay indoors and use air purifiers."),
]

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"❌ Model file '{MODEL_PATH}' not found. Please place it in the app directory.")
        st.stop()

model = load_model()

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, inputs, aqi, category

st.title("🌫️ Air Quality Index (AQI) Predictor")
st.caption("Random Forest Regressor | 11 pollutant features ")

# ─────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────
st.sidebar.header("🧪 Pollutant Levels")

preset_choice = st.sidebar.selectbox("Quick preset", ["— choose a preset —"] + list(PRESETS.keys()))
if preset_choice != "— choose a preset —" and st.sidebar.button("Apply preset"):
    for feat, val in PRESETS[preset_choice].items():
        st.session_state[feat] = float(val)
    st.rerun()

inputs = {}
for feat, (min_val, max_val, default, unit) in FEATURES.items():
    inputs[feat] = st.sidebar.slider(f"{feat} ({unit})", min_val, max_val, st.session_state.get(feat, default), key=feat)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict AQI", type="primary", use_container_width=True)
clear_history_btn = st.sidebar.button("🗑️ Clear history", use_container_width=True)
if clear_history_btn:
    st.session_state.history = []

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_aqi_category(aqi):
    for threshold, category, color, advice in CATEGORY_INFO:
        if aqi <= threshold:
            return category, color, advice

def make_gauge(aqi, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        number={'suffix': " AQI"},
        gauge={
            'axis': {'range': [0, 500]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50],   'color': '#e6f7e6'},
                {'range': [50, 100], 'color': '#ffffcc'},
                {'range': [100, 150],'color': '#ffe0b3'},
                {'range': [150, 200],'color': '#ffb3b3'},
                {'range': [200, 300],'color': '#e0b3e6'},
                {'range': [300, 500],'color': "#a87c82"},
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
    return fig

def predict_aqi(feature_dict):
    vector = np.array([[feature_dict[feat] for feat in FEATURES.keys()]])
    pred = model.predict(vector)[0]
    return max(0.0, float(pred))

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab_predict, tab_history, tab_about = st.tabs(
    ["🔮 Predict", "📈 History & Trend", "ℹ️ Model Info"]
)

# ───────────────── PREDICT TAB ─────────────────
with tab_predict:
    if predict_btn:
        aqi = predict_aqi(inputs)
        category, color, advice = get_aqi_category(aqi)

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            **inputs,
            "AQI": round(aqi, 1),
            "Category": category,
        })

        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.plotly_chart(make_gauge(aqi, color), use_container_width=True)
        with col2:
            st.markdown(f"<h2 style='color:{color}; margin-bottom:0;'>{category}</h2>", unsafe_allow_html=True)
            st.metric("Predicted AQI", f"{aqi:.1f}")
            st.info(advice)

        st.subheader("📊 Input Pollutant Levels")
        col_left, col_right = st.columns(2)
        with col_left:
            st.bar_chart(inputs)
        with col_right:
            with st.expander("View Raw Values", expanded=True):
                df_display = pd.DataFrame({
                    "Pollutant": list(inputs.keys()),
                    "Value": list(inputs.values()),
                    "Unit": [u for _, _, _, u in FEATURES.values()]
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Downloadable report for this single prediction
            report_df = df_display.copy()
            report_df.loc[len(report_df)] = ["Predicted AQI", round(aqi, 1), category]
            csv_buf = io.StringIO()
            report_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download this result as CSV",
                data=csv_buf.getvalue(),
                file_name=f"aqi_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("ℹ️ Feature Importance"):
            if hasattr(model, 'feature_importances_'):
                imp_df = pd.DataFrame({
                    "Feature": list(FEATURES.keys()),
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)
                st.bar_chart(imp_df.set_index("Feature"))
            else:
                st.write("This model does not expose feature importances.")
    else:
        st.info("👈 Adjust pollutant levels in the sidebar (or apply a preset) and click **Predict AQI**.")
        st.subheader("📊 Input Pollutant Levels (current sliders)")
        st.bar_chart(inputs)

# ───────────────── HISTORY TAB ─────────────────
with tab_history:
    st.subheader("📈 Prediction history (this session)")
    if not st.session_state.history:
        st.info("No predictions yet — run a prediction from the **Predict** tab to start building a trend.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.line_chart(hist_df.set_index("time")["AQI"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        csv_buf = io.StringIO()
        hist_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download full history as CSV",
            data=csv_buf.getvalue(),
            file_name=f"aqi_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ───────────────── ABOUT / MODEL INFO TAB ─────────────────
with tab_about:
    st.subheader("ℹ️ Model Info")
    st.write(f"**Model type:** Random Forest Regressor")
    st.write(f"**Features used:** {len(FEATURES)}")
    st.write(f"**Feature order:** {list(FEATURES.keys())}")
    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({
            "Feature": list(FEATURES.keys()),
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

