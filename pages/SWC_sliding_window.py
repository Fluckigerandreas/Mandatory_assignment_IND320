# SWC_sliding_window.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import requests_cache
from retry_requests import retry
import openmeteo_requests

from Data_loader import load_production, load_consumption


# ======================================================
# STREAMLIT SETTINGS
# ======================================================
st.set_page_config(layout="wide")
st.title("Sliding Window Correlation: Meteorology vs Energy")


# ======================================================
# ERA5 WEATHER DATA DOWNLOAD (Open-Meteo)
# ======================================================
@st.cache_data(show_spinner="Downloading ERA5 ERA5 weather data...")
def download_era5_openmeteo(lat, lon, year, timezone="Europe/Oslo"):
    """Download ERA5 hourly data using Open-Meteo with retries & caching."""

    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
        ],
        "models": "era5",
        "timezone": timezone,
    }

    response = client.weather_api(url, params=params)[0]
    hourly = response.Hourly()

    # ---- Extract timestamps (seconds → datetime) ----
    timestamps = pd.to_datetime(hourly.Time(), unit="s", utc=True)

    # ---- Extract all variables ----
    df = pd.DataFrame({
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "precipitation": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
        "wind_gusts_10m": hourly.Variables(3).ValuesAsNumpy(),
        "wind_direction_10m": hourly.Variables(4).ValuesAsNumpy(),
    }, index=timestamps)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]

    return df


# ======================================================
# FIXED LOCATION + YEAR
# ======================================================
LAT, LON = 59.91, 10.75
YEAR = 2023


# ======================================================
# LOAD ENERGY DATA
# ======================================================
prod_df = load_production()
cons_df = load_consumption()


# ======================================================
# SIDEBAR SETTINGS
# ======================================================
st.sidebar.header("Settings")

variable_weather = st.sidebar.selectbox(
    "Select meteorological variable",
    ["temperature_2m", "precipitation", "wind_speed_10m",
     "wind_direction_10m", "wind_gusts_10m"]
)

energy_type = st.sidebar.radio("Select energy type", ["Production", "Consumption"])

if energy_type == "Production":
    energy_df = prod_df.reset_index()
    group_col = "productiongroup"
else:
    energy_df = cons_df.reset_index()
    group_col = "consumptiongroup"

groups = sorted(energy_df[group_col].dropna().unique())
selected_group = st.sidebar.selectbox("Select group", groups)

# Controls
lag = st.sidebar.slider("Lag (hours)", 0, 200, 0)
window = st.sidebar.slider("Sliding window size", 5, 200, 48)
max_lag = st.sidebar.slider("Max lag for cross-correlation", 0, 200, 50)


# ======================================================
# PREPARE ENERGY SERIES
# ======================================================
energy_series = (
    energy_df[energy_df[group_col] == selected_group]
    .assign(quantitykwh=lambda df: pd.to_numeric(df["quantitykwh"], errors="coerce"))
    .dropna(subset=["quantitykwh"])
    .groupby("starttime")["quantitykwh"]
    .sum()
)


# ======================================================
# LOAD WEATHER DATA
# ======================================================
weather_df = download_era5_openmeteo(LAT, LON, YEAR)


# ======================================================
# ALIGN WEATHER + ENERGY
# ======================================================
merged = pd.concat(
    [weather_df[variable_weather], energy_series],
    axis=1, join="inner"
).dropna()

if merged.empty:
    st.error("Merged dataset is empty — no overlapping timestamps found.")
    st.stop()

x = merged[variable_weather]
y = merged["quantitykwh"]


# ======================================================
# SLIDING WINDOW CORRELATION
# ======================================================
def sliding_window_corr(x, y, lag, window):
    x_shifted = x.shift(lag)
    df = pd.concat([x_shifted, y], axis=1).dropna()
    return df.iloc[:, 1].rolling(window, center=True).corr(df.iloc[:, 0])


swc = sliding_window_corr(x, y, lag, window)

if lag > 0:
    corr_value = np.corrcoef(y[lag:], x[:-lag])[0, 1]
else:
    corr_value = np.corrcoef(y, x)[0, 1]


# ======================================================
# FIGURE 1: WEATHER SERIES
# ======================================================
st.subheader(f"Weather variable: {variable_weather}")

fig_weather = go.Figure()
fig_weather.add_trace(go.Scatter(x=x.index, y=x, mode="lines", name=variable_weather))
fig_weather.update_layout(height=300, yaxis_title=variable_weather)
st.plotly_chart(fig_weather, width="stretch")


# ======================================================
# FIGURE 2: ENERGY SERIES
# ======================================================
st.subheader(f"Energy series: {selected_group}")

fig_energy = go.Figure()
fig_energy.add_trace(go.Scatter(x=y.index, y=y, mode="lines", name="kWh"))
fig_energy.update_layout(height=300, yaxis_title="kWh")
st.plotly_chart(fig_energy, width="stretch")


# ======================================================
# FIGURE 3: SLIDING WINDOW CORRELATION
# ======================================================
st.subheader("Sliding Window Correlation")

fig_swc = go.Figure()
fig_swc.add_trace(go.Scatter(x=swc.index, y=swc, mode="lines", name="SWC"))
fig_swc.update_layout(
    height=350,
    xaxis_title="Time",
    yaxis_title="Correlation",
    title=f"SWC (lag={lag}, window={window}) — Corr={corr_value:.3f}"
)
st.plotly_chart(fig_swc, width="stretch")


# ======================================================
# FIGURE 4: CORRELATION VS LAG
# ======================================================
st.header("Correlation vs Lag")

lags = range(-max_lag, max_lag + 1)
corrs = []

for L in lags:
    if L > 0:
        corr = np.corrcoef(y[L:], x[:-L])[0, 1]
    elif L < 0:
        corr = np.corrcoef(y[:L], x[-L:])[0, 1]
    else:
        corr = np.corrcoef(y, x)[0, 1]
    corrs.append(corr)

fig_lag = go.Figure()
fig_lag.add_trace(go.Scatter(x=list(lags), y=corrs, mode="lines+markers"))
fig_lag.update_layout(
    height=350,
    xaxis_title="Lag (hours)",
    yaxis_title="Correlation",
    title="Correlation vs Lag"
)
st.plotly_chart(fig_lag, width="stretch")
