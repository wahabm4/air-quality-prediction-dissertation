# London Air Quality Dashboard 🌬️

AI-Driven PM2.5 Analysis & Forecasting · 2016–2025

A Streamlit web application presenting the research findings of the CSI-6-CSP dissertation project. The dashboard provides interactive visualisations of London's air quality trends, environmental drivers, and a stacked ensemble model for PM2.5 forecasting.

## Features

- **Overview** — Key metrics, annual trends, seasonal patterns, model comparison
- **Trends** — Time series analysis with STL decomposition, WHO compliance tracking
- **Drivers** — Correlation heatmaps, scatter plots, feature importance analysis
- **Forecast** — Live prediction with stacked ensemble (RF + XGBoost + LR meta-learner), 7-day outlook

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs on synthetic data by default. Drop in the real `ensemble_bundle.pkl` and `london_pollutants_weather_data_10Yrs.csv` to activate real data and predictions — no code changes required.

## Project Structure

```
air_quality_app/
├── app.py                        ← entry point, Overview page
├── ensemble_bundle.pkl           ← model file (provided separately)
├── london_pollutants_weather_data_10Yrs.csv  ← real data (optional)
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml               ← theme configuration
└── pages/
    ├── 1_Trends.py               ← PS1 deep dive
    ├── 2_Drivers.py              ← PS2 deep dive
    └── 3_Forecast.py             ← PS3 + live prediction
```

## Tech Stack

- Python 3.11+
- Streamlit >= 1.32.0
- Plotly >= 5.18.0
- scikit-learn, XGBoost, statsmodels
- Open-Meteo API (free, no key required)

## Deployment

Push to GitHub, then deploy via [Streamlit Community Cloud](https://share.streamlit.io).

---

CSI-6-CSP Honours Computer Science Project · London South Bank University
