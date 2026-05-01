import streamlit as st
st.set_page_config(
    page_title="Trends · London Air Quality",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import get_data, make_fig, render_nav, stat_card
import plotly.graph_objects as go
import numpy as np

# ─── Page Setup ─────────────────────────────────────────────────────
render_nav("Trends")
st.markdown('<div class="page-body">', unsafe_allow_html=True)

df = get_data()

# ─── Controls Row ───────────────────────────────────────────────────
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
with ctrl1:
    year_range = st.slider("Year range", 2017, 2025, (2017, 2025))
with ctrl2:
    smoothing = st.slider("Rolling average (days)", 7, 90, 30)
with ctrl3:
    who_toggle = st.toggle("WHO threshold", value=True)
st.markdown('</div>', unsafe_allow_html=True)

# Filter data
mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
filt = df.loc[mask].copy().sort_values("date")

# ─── Time Series Chart ─────────────────────────────────────────────
st.markdown('<div class="chart-container"><div class="chart-title">Daily PM2.5 with Rolling Average</div>', unsafe_allow_html=True)
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=filt["date"], y=filt["pm25_mean"],
    mode="lines", name="Daily PM2.5",
    line=dict(color="#0D9488", width=1),
    opacity=0.4,
    hovertemplate="%{x|%Y-%m-%d}<br>PM2.5: %{y:.1f}<extra></extra>",
))
rolling = filt["pm25_mean"].rolling(smoothing, min_periods=1).mean()
fig_ts.add_trace(go.Scatter(
    x=filt["date"], y=rolling,
    mode="lines", name=f"{smoothing}-day avg",
    line=dict(color="#F59E0B", width=2),
    hovertemplate="%{x|%Y-%m-%d}<br>Avg: %{y:.1f}<extra></extra>",
))
if who_toggle:
    fig_ts.add_hline(y=5, line_dash="dash", line_color="#EF4444", line_width=1.5,
                     annotation_text="WHO 5 µg/m³",
                     annotation_position="top left",
                     annotation_font_color="#EF4444",
                     annotation_font_size=11)
fig_ts.update_layout(height=300, showlegend=True,
                     legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0))
make_fig(fig_ts)
st.plotly_chart(fig_ts, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ─── STL Decomposition ─────────────────────────────────────────────
st.markdown('<div class="chart-container"><div class="chart-title">STL Decomposition</div>', unsafe_allow_html=True)
if len(filt) > 730:
    from statsmodels.tsa.seasonal import STL
    series = filt.set_index("date")["pm25_mean"].asfreq("D")
    series = series.interpolate()
    stl = STL(series, period=365, robust=True).fit()

    fig_stl = go.Figure()
    fig_stl.add_trace(go.Scatter(x=stl.trend.index, y=stl.trend, mode="lines",
                                  name="Trend", line=dict(color="#0D9488", width=2)))
    fig_stl.add_trace(go.Scatter(x=stl.seasonal.index, y=stl.seasonal, mode="lines",
                                  name="Seasonal", line=dict(color="#F59E0B", width=1.5)))
    fig_stl.add_trace(go.Scatter(x=stl.resid.index, y=stl.resid, mode="lines",
                                  name="Residual", line=dict(color="#94A3B8", width=1), opacity=0.6))
    fig_stl.update_layout(height=260, legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0))
    make_fig(fig_stl)
    st.plotly_chart(fig_stl, width="stretch", config={"displayModeBar": False})
else:
    st.info("Select a date range covering at least 2 years (730 days) to display STL decomposition.")
st.markdown('</div>', unsafe_allow_html=True)

# ─── WHO Compliance Chart ──────────────────────────────────────────
st.markdown('<div class="chart-container"><div class="chart-title">WHO Guideline Exceedance by Year</div>', unsafe_allow_html=True)
yearly_exc = filt.groupby("year").apply(
    lambda g: round((g["pm25_mean"] > 5).mean() * 100, 1)
).reset_index()
yearly_exc.columns = ["year", "pct"]
exc_colors = ["#EF4444" if p > 40 else "#0D9488" for p in yearly_exc["pct"]]

fig_who = go.Figure()
fig_who.add_trace(go.Bar(
    x=yearly_exc["year"], y=yearly_exc["pct"],
    marker_color=exc_colors,
    text=[f"{v}%" for v in yearly_exc["pct"]],
    textposition="outside",
    textfont=dict(color="#E2E8F0", size=11),
    hovertemplate="Year %{x}<br>Exceedance: %{y:.1f}%<extra></extra>",
))
fig_who.update_layout(height=260, showlegend=False, yaxis_title="Days exceeding 5 µg/m³ (%)")
make_fig(fig_who)
st.plotly_chart(fig_who, width="stretch", config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ─── Mann-Kendall Callout Cards ────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
mk1, mk2, mk3, mk4 = st.columns(4)
with mk1:
    st.markdown(stat_card("Test result", "Inconclusive", "p = 0.065", "#94A3B8"), unsafe_allow_html=True)
with mk2:
    st.markdown(stat_card("Trend direction", "Inconclusive", "Mann-Kendall", "#94A3B8"), unsafe_allow_html=True)
with mk3:
    st.markdown(stat_card("Seasonal cycle", "Annual", "365-day period", "#0D9488"), unsafe_allow_html=True)
with mk4:
    st.markdown(stat_card("WHO exceedance trend", "Worsening", "2022–2025", "#EF4444"), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
