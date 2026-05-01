import streamlit as st
st.set_page_config(
    page_title="London Air Quality Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys


# ─── Data Loader ───────────────────────────────────────────────────────
@st.cache_data
def get_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "london_pollutants_weather_data_10Yrs.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "❌ Dataset not found. Please place london_pollutants_weather_data_10Yrs.csv inside /data folder"
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df.sort_values("date").dropna().reset_index(drop=True)
    return df


def get_metrics():
    """Return metrics dict — from pickle bundle if available, else corrected defaults."""
    metrics = _default_metrics()
    base = os.path.dirname(os.path.abspath(__file__))
    bundle_path = os.path.join(base, "lstm_bundle_v2.pkl")
    if os.path.exists(bundle_path):
        import joblib
        bundle = joblib.load(bundle_path)
        b_metrics = bundle.get("metrics", {})
        if "mae" in b_metrics:
            metrics["bilstm_mae"] = b_metrics["mae"]
        if "rmse" in b_metrics:
            metrics["bilstm_rmse"] = b_metrics["rmse"]
    return metrics


def _default_metrics():
    # Corrected values from final model run on corrected dataset
    return {
        "bilstm_mae":    2.34,
        "bilstm_rmse":   3.18,
        "ensemble_mae":  4.69,
        "ensemble_rmse": 7.26,
        "rf_rmse":       5.22,
        "arima_rmse":    7.61,
    }


# ─── Shared make_fig helper ────────────────────────────────────────────
def make_fig(fig):
    fig.update_layout(
        paper_bgcolor="#0F2035",
        plot_bgcolor="#0F2035",
        font=dict(color="#E2E8F0", size=12),
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
        yaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
    )
    return fig


# ─── Navigation Bar ────────────────────────────────────────────────────
def render_nav(active="Overview"):
    pages = [
        ("Overview", "/"),
        ("Trends",   "/Trends"),
        ("Drivers",  "/Drivers"),
        ("Forecast", "/Forecast"),
    ]
    links = ""
    for label, href in pages:
        cls = "nav-link active" if label == active else "nav-link"
        links += f'<a href="{href}" class="{cls}" target="_self">{label}</a>'

    st.markdown(f"""
    <style>
        .nav-bar {{
            display: flex; gap: 8px; background: #0F2035;
            padding: 10px 24px; border-radius: 12px;
            margin-bottom: 24px; border: 1px solid #1E3A5F;
        }}
        .nav-link {{
            color: #94A3B8; text-decoration: none;
            padding: 8px 20px; border-radius: 8px;
            font-weight: 500; font-size: 14px; transition: all 0.2s;
        }}
        .nav-link:hover {{ color: #E2E8F0; background: rgba(13,148,136,0.15); }}
        .nav-link.active {{ color: #0D9488; background: rgba(13,148,136,0.15); font-weight: 600; }}
        .stat-card {{ background: #0F2035; border-radius: 12px; padding: 20px; border: 1px solid #1E3A5F; text-align: center; }}
        .stat-label {{ color: #94A3B8; font-size: 13px; font-weight: 500; margin-bottom: 6px; }}
        .stat-value {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
        .stat-sub {{ color: #64748B; font-size: 12px; }}
        .page-body {{ padding: 24px 32px; }}
        .chart-container {{ background: #0F2035; border-radius: 12px; padding: 20px; border: 1px solid #1E3A5F; margin-bottom: 16px; }}
        .chart-title {{ color: #E2E8F0; font-size: 15px; font-weight: 600; margin-bottom: 12px; }}
        .finding-card {{ background: #0F2035; border-radius: 10px; padding: 14px 18px; border: 1px solid #1E3A5F; display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .finding-label {{ color: #E2E8F0; font-size: 13px; font-weight: 500; }}
        .finding-desc {{ color: #94A3B8; font-size: 12px; margin-top: 2px; }}
        .finding-value {{ font-size: 14px; font-weight: 700; white-space: nowrap; }}
    </style>
    <div class="nav-bar">{links}</div>
    """, unsafe_allow_html=True)


def stat_card(label, value, sub="", colour="#0D9488"):
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value" style="color:{colour}">{value}</div>
        <div class="stat-sub">{sub}</div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════
# OVERVIEW PAGE
# ═══════════════════════════════════════════════════════════════════════
render_nav("Overview")
st.markdown('<div class="page-body">', unsafe_allow_html=True)

df = get_data()
metrics = get_metrics()

# ── Stat Cards ─────────────────────────────────────────────────────────
improvement = round((1 - metrics["bilstm_rmse"] / metrics["arima_rmse"]) * 100)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(stat_card("Dataset size", "3,628", "daily observations · 22 vars", "#E2E8F0"), unsafe_allow_html=True)
with c2:
    st.markdown(stat_card("Best model RMSE", f"{metrics['bilstm_rmse']:.2f}", "AttentionBiLSTM v2 µg/m³", "#0D9488"), unsafe_allow_html=True)
with c3:
    st.markdown(stat_card("vs ARIMA baseline", f"−{improvement}%", "improvement over baseline", "#22C55E"), unsafe_allow_html=True)
with c4:
    st.markdown(stat_card("WHO exceedance", "~70%", "days > 5 µg/m³ in 2025", "#EF4444"), unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Annual Mean PM2.5 + Seasonal ────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="chart-container"><div class="chart-title">Annual Mean PM2.5</div>', unsafe_allow_html=True)
    annual = df.groupby("year")["pm25_mean"].mean().reset_index()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=annual["year"], y=annual["pm25_mean"],
        marker_color="#0D9488",
        hovertemplate="Year %{x}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
    ))
    # WHO 2021 revised annual guideline: 5 µg/m³
    fig_trend.add_hline(y=5, line_dash="dash", line_color="#EF4444", line_width=1.5,
                        annotation_text="WHO guideline (5 µg/m³, 2021 revised)",
                        annotation_position="top left",
                        annotation_font_color="#EF4444",
                        annotation_font_size=11)
    fig_trend.update_layout(height=240, showlegend=False, yaxis_title="PM2.5 (µg/m³)")
    make_fig(fig_trend)
    st.plotly_chart(fig_trend, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="chart-container"><div class="chart-title">Seasonal Pattern (Monthly)</div>', unsafe_allow_html=True)
    monthly = df.groupby("month")["pm25_mean"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    winter_months = {1, 2, 10, 11, 12}
    bar_colors = ["#EF4444" if m in winter_months else "#0D9488" for m in monthly["month"]]
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Bar(
        x=[month_names[m-1] for m in monthly["month"]],
        y=monthly["pm25_mean"],
        marker_color=bar_colors,
        hovertemplate="%{x}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
    ))
    fig_seasonal.update_layout(height=240, showlegend=False)
    make_fig(fig_seasonal)
    st.plotly_chart(fig_seasonal, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ── Model Comparison Table ──────────────────────────────────────────────
st.markdown('<div class="chart-container"><div class="chart-title">Model Performance Comparison</div>', unsafe_allow_html=True)

# Corrected values from final model run on corrected dataset
models    = ["ARIMA (baseline)", "SARIMA", "LSTM original", "XGBoost", "XGBoost (tuned)",
             "Random Forest", "RF (ensemble run)", "Stacked Ensemble", "AttentionBiLSTM v2 ★"]
mae_vals  = [5.28, 5.02, 4.62, 3.24, 3.19, 3.07, 4.36, 4.69, 2.34]
rmse_vals = [7.61, 7.63, 6.96, 5.46, 5.42, 5.22, 6.77, 7.26, 3.18]
vs_base   = []
for r in rmse_vals:
    pct = round((1 - r / 7.61) * 100)
    if pct > 0:
        vs_base.append(f"−{pct}%")
    elif pct == 0:
        vs_base.append("baseline")
    else:
        vs_base.append(f"+{abs(pct)}% (worse)")

row_fills       = ["#0F2035"] * len(models)
row_fills[-1]   = "#0D394D"   # highlight BiLSTM row
row_font_colors = ["#E2E8F0"] * len(models)
row_font_colors[-1] = "#5EEAAA"

vs_base_colors = ["#E2E8F0"] * len(models)
vs_base_colors[-1] = "#5EEAAA"

fig_table = go.Figure(data=[go.Table(
    header=dict(
        values=["<b>Model</b>", "<b>MAE</b>", "<b>RMSE</b>", "<b>vs baseline</b>"],
        fill_color="#1E3A5F",
        font=dict(color="#E2E8F0", size=13),
        align="left", height=32,
    ),
    cells=dict(
        values=[models, mae_vals, rmse_vals, vs_base],
        fill_color=[row_fills],
        font=dict(
            color=[row_font_colors, row_font_colors, row_font_colors, vs_base_colors],
            size=12,
        ),
        align="left", height=28,
    ),
)])
fig_table.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
make_fig(fig_table)
st.plotly_chart(fig_table, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ── Feature Importance Chart ────────────────────────────────────────────
st.markdown('<div class="chart-container"><div class="chart-title">Feature Importance (Random Forest — PS2)</div>', unsafe_allow_html=True)

# Corrected PS2 RF feature importance (no lag features, raw variables only)
fi_features    = ["CO", "SO₂", "PM10", "NO₂", "o3", "Wind speed", "Pressure", "Temperature"]
fi_importances = [6.1, 6.2, 6.4, 6.7, 15.5, 17.8, 19.4, 21.8]
met_features   = {"Wind speed", "Pressure", "Temperature"}
bar_colors_fi  = ["#475569" if f in met_features else "#0D9488" for f in fi_features]

fig_fi = go.Figure()
fig_fi.add_trace(go.Bar(
    y=fi_features, x=fi_importances,
    orientation="h",
    marker_color=bar_colors_fi,
    text=[f"{v}%" for v in fi_importances],
    textposition="outside",
    textfont=dict(color="#E2E8F0", size=11),
    hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
))
fig_fi.update_layout(height=280, showlegend=False, xaxis_title="Importance (%)")
make_fig(fig_fi)
st.plotly_chart(fig_fi, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)