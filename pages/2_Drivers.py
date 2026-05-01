import streamlit as st
st.set_page_config(
    page_title="Drivers · London Air Quality",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import get_data, make_fig, render_nav, stat_card
import plotly.graph_objects as go
import numpy as np

# ─── Page Setup ─────────────────────────────────────────────────────
render_nav("Drivers")
st.markdown('<div class="page-body">', unsafe_allow_html=True)

df = get_data()

# ─── Correlation Heatmap ───────────────────────────────────────────
st.markdown('<div class="chart-container"><div class="chart-title">Pollutant & Weather Correlation Matrix</div>', unsafe_allow_html=True)
corr_cols = ["pm25_mean", "no2_mean", "so2_mean", "co_mean", "pm10_mean", "temp_avg", "wind_speed", "pressure"]
display_labels = ["PM2.5", "NO₂", "SO₂", "CO", "PM10", "Temp", "Wind", "Pressure"]
corr_matrix = df[corr_cols].corr().values

fig_hm = go.Figure(data=go.Heatmap(
    z=corr_matrix,
    x=display_labels,
    y=display_labels,
    colorscale=[[0, "#1E3A5F"], [0.5, "#0F2035"], [1, "#0D9488"]],
    zmid=0, zmin=-1, zmax=1,
    text=np.round(corr_matrix, 2),
    texttemplate="%{text}",
    textfont=dict(size=11, color="#E2E8F0"),
    hovertemplate="%{x} vs %{y}<br>r = %{z:.2f}<extra></extra>",
    colorbar=dict(tickfont=dict(color="#E2E8F0")),
))
fig_hm.update_layout(height=320, xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"))
make_fig(fig_hm)
st.plotly_chart(fig_hm, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ─── Interactive Scatter Plot ──────────────────────────────────────
st.markdown('<div class="chart-container"><div class="chart-title">PM2.5 vs Environmental Factor</div>', unsafe_allow_html=True)

var_map = {
    "NO₂": "no2_mean",
    "SO₂": "so2_mean",
    "CO": "co_mean",
    "PM10": "pm10_mean",
    "Temperature": "temp_avg",
    "Wind speed": "wind_speed",
    "Pressure": "pressure",
}
selected_var = st.selectbox("Select variable", list(var_map.keys()))
col_name = var_map[selected_var]

sample = df.sample(n=min(600, len(df)), random_state=42)
x_vals = sample[col_name].values
y_vals = sample["pm25_mean"].values

# Trendline
coeffs = np.polyfit(x_vals, y_vals, 1)
trend_x = np.linspace(x_vals.min(), x_vals.max(), 100)
trend_y = np.polyval(coeffs, trend_x)
r = np.corrcoef(x_vals, y_vals)[0, 1]

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=x_vals, y=y_vals,
    mode="markers",
    marker=dict(color="#0D9488", size=4, opacity=0.5),
    name=f"r = {r:.2f}",
    hovertemplate=f"{selected_var}: %{{x:.1f}}<br>PM2.5: %{{y:.1f}}<extra></extra>",
))
fig_scatter.add_trace(go.Scatter(
    x=trend_x, y=trend_y,
    mode="lines",
    line=dict(color="#F59E0B", width=2),
    name="Trendline",
))
fig_scatter.update_layout(
    height=320, showlegend=True,
    xaxis_title=selected_var,
    yaxis_title="PM2.5 (µg/m³)",
    legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0),
)
make_fig(fig_scatter)
st.plotly_chart(fig_scatter, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ─── Feature Importance Chart ─────────────────────────────────────
st.markdown('<div class="chart-container"><div class="chart-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)

features = ["CO", "SO₂", "PM10", "NO₂", "o3", "Wind speed", "Pressure", "Temperature"]
importances = [6.1, 6.2, 6.4, 6.7, 15.5, 17.8, 19.4, 21.8]
met_features = {"Pressure", "Temperature", "Wind speed"}
bar_colors_fi = ["#475569" if f in met_features else "#0D9488" for f in features]

fig_fi = go.Figure()
fig_fi.add_trace(go.Bar(
    y=features, x=importances,
    orientation="h",
    marker_color=bar_colors_fi,
    text=[f"{v}%" for v in importances],
    textposition="outside",
    textfont=dict(color="#E2E8F0", size=11),
    hovertemplate="%{y}: %{x}%<extra></extra>",
))
fig_fi.update_layout(height=320, showlegend=False, xaxis_title="Importance (%)")
make_fig(fig_fi)
st.plotly_chart(fig_fi, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ─── Key Findings Summary ─────────────────────────────────────────
st.markdown('<div class="chart-title" style="margin-top:16px">Key Findings</div>', unsafe_allow_html=True)

findings = [
    ("Co-pollutant correlation", "NO₂, SO₂, CO all positively correlated with PM2.5", "r ≈ 0.21", "#EF4444"),
    ("Wind speed effect", "Higher wind disperses particulates, reducing PM2.5", "r = −0.20", "#0D9488"),
    ("OLS regression fit", "Linear model explains ~11% of PM2.5 variance", "R² = 0.108", "#3B82F6"),
    ("Partial correlation", "NO₂ remains significant after controlling for weather", "r ≈ 0.167", "#22C55E"),
    ("Precipitation", "Weak washout effect, not statistically significant", "NS", "#94A3B8"),
    ("High pressure", "Atmospheric inversions trap pollutants near ground", "r ≈ +0.20", "#F59E0B"),
]

for label, desc, value, colour in findings:
    st.markdown(f"""
    <div class="finding-card">
        <div>
            <div class="finding-label">{label}</div>
            <div class="finding-desc">{desc}</div>
        </div>
        <div class="finding-value" style="color:{colour}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
