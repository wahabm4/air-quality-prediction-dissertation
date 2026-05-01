import streamlit as st
st.set_page_config(
    page_title="Forecast · London Air Quality",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import get_data, make_fig, render_nav, stat_card
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras

# Register custom objects globally for Lambda layers
keras.utils.get_custom_objects().update({'tf': tf, 'K': tf.keras.backend, 'np': np})

# ─── Load BiLSTM v2 (Robust Scratch Rebuild) ─────────────────────────
@st.cache_resource
def load_bilstm():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundle_path = os.path.join(base, "lstm_bundle_v2.pkl")
    model_path = os.path.join(base, "best_lstm_v2.keras")

    if not os.path.exists(bundle_path) or not os.path.exists(model_path):
        return None, None

    # 1. Build architecture from scratch to avoid faulty Lambda serialization
    def build_architecture():
        inputs = tf.keras.layers.Input(shape=(60, 4), name="input")
        
        # BiLSTM 1
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True), 
            name="bilstm_1"
        )(inputs)
        x = tf.keras.layers.Dropout(0.2, name="drop_1")(x)
        
        # BiLSTM 2
        bilstm_2_out = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True), 
            name="bilstm_2"
        )(x)
        
        # Attention Mechanism
        attn_score = tf.keras.layers.Dense(1, name="attn_score")(bilstm_2_out)
        attn_weights = tf.keras.layers.Softmax(axis=1, name="attn_weights")(attn_score)
        attn_apply = tf.keras.layers.Multiply(name="attn_apply")([bilstm_2_out, attn_weights])
        
        # Solid sum layer using tf.reduce_sum directly
        attn_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name="attn_sum")(attn_apply)
        
        # Fully Connected + Residual
        dense_1 = tf.keras.layers.Dense(32, activation='relu', name="dense_1")(attn_sum)
        drop_2 = tf.keras.layers.Dropout(0.2, name="drop_2")(dense_1)
        
        residual_proj = tf.keras.layers.Dense(32, name="residual_proj")(attn_sum)
        residual_add = tf.keras.layers.Add(name="residual_add")([drop_2, residual_proj])
        
        outputs = tf.keras.layers.Dense(1, name="output")(residual_add)
        
        return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="AttentionBiLSTM")

    # 2. Initialize and load
    bundle = joblib.load(bundle_path)
    model = build_architecture()
    
    try:
        model.load_weights(model_path)
    except Exception as e:
        st.error(f"Failed to load weights: {e}")
        return None, None
        
    return bundle, model

# ─── 7-Day Forecast Generation ──────────────────────────────────────
def generate_forecast(df, bundle, model, user_temp, user_wind, user_press):
    scaler = bundle["scaler"]
    features = bundle["features"] # ['pm25_mean', 'temp_avg', 'wind_speed', 'pressure']
    
    # Retrieve lookback from model input shape to ensure compatibility
    try:
        lookback = model.input_shape[1]
        if lookback is None: 
            lookback = bundle.get("lookback", 30)
    except:
        lookback = bundle.get("lookback", 30)
    
    # 1. Take the required rows of real data from the CSV for the features
    real_data = df[df["sensor_count"] > 0].copy().sort_values("date").reset_index(drop=True)
    history = real_data[features].tail(lookback).values.copy() 
    
    # 2. Override the most recent weather regime (last 5 days) with the user's slider values
    # This ensures the LSTM 'feels' the weather change rather than treating it as a single-day outlier.
    regime_days = min(5, lookback)
    history[-regime_days:, 1] = user_temp
    history[-regime_days:, 2] = user_wind
    history[-regime_days:, 3] = user_press
    
    forecast = []
    
    current_window = history.copy()
    
    # Predict 7 days ahead
    for day in range(7):
        # 3. Scale the (30, 4) array using the bundle's MinMaxScaler
        scaled_window = scaler.transform(current_window)
        
        # 4. Reshape to (1, 30, 4) and call model.predict()
        X_input = scaled_window.reshape(1, lookback, len(features))
        pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        
        # 5. Inverse transform
        pred = pred_scaled * scaler.data_range_[0] + scaler.data_min_[0]
        pred = round(float(pred), 1)
        forecast.append(pred)
        
        # 6. For 7-day forecast: append each prediction as the next row's pm25_mean with gentle weather drift
        next_temp = current_window[-1, 1] + 0.1
        next_wind = current_window[-1, 2] * 1.02
        next_press = current_window[-1, 3] * 0.999
        
        next_wind = np.clip(next_wind, 0, 80)
        next_press = np.clip(next_press, 950, 1045)
        
        next_row = np.array([[pred, next_temp, next_wind, next_press]])
        current_window = np.vstack([current_window[1:], next_row])
        
    return forecast

# ─── Live Data Fetch ────────────────────────────────────────────────
def fetch_live_conditions():
    import requests

    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 51.5074,
                "longitude": -0.1278,
                "current": "temperature_2m,wind_speed_10m,surface_pressure",
                "timezone": "Europe/London",
            },
            timeout=5,
        )

        r.raise_for_status()
        data = r.json()
        current = data.get("current", {})

        result = {
            "temp": round(current.get("temperature_2m", 12.0), 1),
            "wind": round(current.get("wind_speed_10m", 4.5) * 3.6, 1),
            "press": round(current.get("surface_pressure", 1013.0), 1),
            "ok": True,
        }
        return result

    except Exception as e:
        return {"ok": False, "error": str(e)}

# ─── WHO Classification ────────────────────────────────────────────
def who_classify(pm25):
    if pm25 <= 5:
        return "Good", "#22C55E"
    elif pm25 <= 15:
        return "Moderate", "#F59E0B"
    elif pm25 <= 35:
        return "Unhealthy", "#EF4444"
    else:
        return "Very Unhealthy", "#991B1B"

# ═══════════════════════════════════════════════════════════════════════
# FORECAST PAGE
# ═══════════════════════════════════════════════════════════════════════
render_nav("Forecast")
st.markdown('<div class="page-body">', unsafe_allow_html=True)

# ✅ Initialize session state
if "temp" not in st.session_state:
    st.session_state["temp"] = 12.0

if "wind" not in st.session_state:
    st.session_state["wind"] = 15.0

if "press" not in st.session_state:
    st.session_state["press"] = 1013.0

bundle, model = load_bilstm()
df = get_data()

if bundle is None or model is None:
    st.error("Model files not found! Please ensure lstm_bundle_v2.pkl and best_lstm_v2.keras are in the project root.")
    st.stop()

metrics = bundle["metrics"] # {'mae': 2.34, 'rmse': 3.18}
arima_rmse = 7.61

# ─── Top Metric Row ─────────────────────────────────────────────────
improvement = round((1 - metrics["rmse"] / arima_rmse) * 100)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(stat_card("Model", "AttentionBiLSTM v2", "Sensor data only", "#E2E8F0"), unsafe_allow_html=True)
with m2:
    st.markdown(stat_card("Test MAE", f"{metrics['mae']:.2f}", "µg/m³", "#0D9488"), unsafe_allow_html=True)
with m3:
    st.markdown(stat_card("Test RMSE", f"{metrics['rmse']:.2f}", "µg/m³", "#0D9488"), unsafe_allow_html=True)
with m4:
    st.markdown(stat_card("vs ARIMA", f"−{improvement}%", "improvement", "#22C55E"), unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ─── Input Panel + Prediction Result ───────────────────────────────
col_input, col_result = st.columns([2, 3])

with col_input:
    st.markdown('<div class="chart-container"><div class="chart-title">Configure Conditions</div>', unsafe_allow_html=True)

    # Fetch live data button
    if st.button("🌍 Fetch today's London conditions"):
        result = fetch_live_conditions()
        if result["ok"]:
            # Update both the internal state and the widget keys
            st.session_state["temp"] = result["temp"]
            st.session_state["wind"] = result["wind"]
            st.session_state["press"] = result["press"]
            st.session_state["slider_temp"] = result["temp"]
            st.session_state["slider_wind"] = result["wind"]
            st.session_state["slider_press"] = result["press"]
            
            st.success(f"Loaded live data: {result['temp']}°C, {result['wind']} km/h, {result['press']} hPa")
            st.rerun()
        else:
            st.error(f"❌ Live data failed: {result['error']}. Using manual sliders.")

    # Group 1 — Weather
    st.markdown("**Weather Conditions (Day 1)**")

    temp_in = st.slider(
        "Temperature (°C)",
        -5.0, 35.0,
        value=float(st.session_state["temp"]),
        step=0.5,
        key="slider_temp"
    )

    wind_in = st.slider(
        "Wind speed (km/h)",
        0.0, 80.0,
        value=float(st.session_state["wind"]),
        step=1.0,
        key="slider_wind"
    )

    press_in = st.slider(
        "Pressure (hPa)",
        950.0, 1045.0,
        value=float(st.session_state["press"]),
        step=1.0,
        key="slider_press"
    )

    st.divider()

    # Run prediction button
    run_pred = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.info("💡 **Note on Sensitivity:** The BiLSTM model is highly autoregressive, prioritizing the 60-day historical trend. Large weather changes will shift the forecast, but the model maintains stability based on recent air quality momentum.")

with col_result:
    if run_pred:
        # Generate 7-day forecast
        outlook = generate_forecast(df, bundle, model, temp_in, wind_in, press_in)
        predicted = outlook[0]
        who_label, who_color = who_classify(predicted)

        # Result card
        st.markdown(f"""
        <div style="background:#0F2035; border-radius:12px; padding:24px; border:2px solid {who_color}; text-align:center; margin-bottom:16px;">
            <div style="color:#94A3B8; font-size:13px; margin-bottom:8px;">Predicted PM2.5 (Day 1)</div>
            <div style="color:{who_color}; font-size:42px; font-weight:700;">{predicted} µg/m³</div>
            <div style="display:inline-block; background:{who_color}22; color:{who_color}; padding:6px 16px; border-radius:20px; font-size:13px; font-weight:600; margin-top:8px;">{who_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # 7-day outlook
        st.markdown('<div class="chart-container"><div class="chart-title">7-Day Outlook</div>', unsafe_allow_html=True)
       
        day_labels = [f"Day {i+1}" for i in range(7)]
        outlook_colors = ["#EF4444" if v > 5 else "#0D9488" for v in outlook]

        fig_outlook = go.Figure()
        fig_outlook.add_trace(go.Bar(
            x=day_labels, y=outlook,
            marker_color=outlook_colors,
            text=[f"{v}" for v in outlook],
            textposition="outside",
            textfont=dict(color="#E2E8F0", size=11),
            hovertemplate="Day %{x}<br>PM2.5: %{y:.1f}<extra></extra>",
        ))
        fig_outlook.add_hline(y=5, line_dash="dash", line_color="#EF4444", line_width=1.5,
                              annotation_text="WHO 5 µg/m³",
                              annotation_position="top left",
                              annotation_font_color="#EF4444",
                              annotation_font_size=11)
        fig_outlook.update_layout(height=260, showlegend=False, yaxis_title="PM2.5 (µg/m³)")
        make_fig(fig_outlook)
        st.plotly_chart(fig_outlook, width="stretch", config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#0F2035; border-radius:12px; padding:40px; border:1px solid #1E3A5F; text-align:center; margin-bottom:16px;">
            <div style="color:#64748B; font-size:36px; margin-bottom:12px;">🔮</div>
            <div style="color:#94A3B8; font-size:15px;">Configure conditions and click <b>Run Prediction</b></div>
        </div>
        """, unsafe_allow_html=True)

# ─── Model RMSE Comparison Chart ────────────────────────────────────
st.markdown('<div class="chart-container"><div class="chart-title">Model RMSE Comparison</div>', unsafe_allow_html=True)

model_names = ["AttentionBiLSTM v2 ★", "Random Forest", "XGBoost (tuned)", "XGBoost untuned", "RF (ensemble run)", "LSTM original", "Stacked Ensemble", "ARIMA (5,1,2)", "SARIMA"]
rmse_values = [3.18, 5.22, 5.42, 5.46, 6.77, 6.96, 7.26, 7.61, 7.63]
model_names = model_names[::-1]
rmse_values = rmse_values[::-1]

bar_colors_rmse = ["#1E3A5F"] * len(model_names)
bar_colors_rmse[-1] = "#0D394D"

fig_rmse = go.Figure()
fig_rmse.add_trace(go.Bar(
    y=model_names, x=rmse_values,
    orientation="h",
    marker_color=bar_colors_rmse,
    text=[f"{v}" for v in rmse_values],
    textposition="outside",
    textfont=dict(color="#E2E8F0", size=11),
    hovertemplate="%{y}<br>RMSE: %{x}<extra></extra>",
))
fig_rmse.add_vline(x=7.61, line_dash="dash", line_color="#EF4444", line_width=1.5,
                   annotation_text="ARIMA baseline (7.61)",
                   annotation_position="top",
                   annotation_font_color="#EF4444",
                   annotation_font_size=11)
fig_rmse.update_layout(height=350, showlegend=False, xaxis_title="RMSE (µg/m³)")
make_fig(fig_rmse)
st.plotly_chart(fig_rmse, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)
