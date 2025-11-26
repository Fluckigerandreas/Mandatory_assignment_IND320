# SWC_sliding_window.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Data_loader import load_production, load_consumption
import requests

st.set_page_config(layout="wide")
st.title("Sliding Window Correlation: Meteorology vs Energy")

# -------------------------
# Fixed location and year
# -------------------------
LAT, LON = 59.91, 10.75
YEAR = 2023

# -------------------------
# Download ERA5 weather (cached)
# -------------------------
@st.cache_data(show_spinner="Downloading ERA5 weather...")
def download_era5(lat, lon, year, timezone="Europe/Oslo"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "timezone": timezone,
        "models": "era5",
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m",
                   "wind_direction_10m", "wind_gusts_10m"]
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    
    numeric_cols = ["temperature_2m", "precipitation", "wind_speed_10m",
                    "wind_direction_10m", "wind_gusts_10m"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]
    return df

# -------------------------
# Load energy data (cached)
# -------------------------
prod_df = load_production()
cons_df = load_consumption()

# -------------------------
# Sidebar selectors
# -------------------------
st.sidebar.header("Settings")
variable_weather = st.sidebar.selectbox("Select meteorological variable", 
                                        ["temperature_2m", "precipitation", "wind_speed_10m", 
                                         "wind_direction_10m", "wind_gusts_10m"])
variable_energy_type = st.sidebar.radio("Select energy type", ["Production", "Consumption"])

if variable_energy_type == "Production":
    energy_df = prod_df.reset_index()
    group_col = "productiongroup"
else:
    energy_df = cons_df.reset_index()
    group_col = "consumptiongroup"

# Select group
groups = sorted(energy_df[group_col].dropna().unique())
selected_group = st.sidebar.selectbox("Select group", groups)

lag = st.sidebar.slider("Lag (hours)", 0, 100, 0)
window = st.sidebar.slider("Sliding window length", 1, 60, 45)
center = st.sidebar.slider("Center index for window", window//2, 1000, 500)

# -------------------------
# Prepare energy series
# -------------------------
energy_series = energy_df[energy_df[group_col]==selected_group].copy()
energy_series["quantitykwh"] = pd.to_numeric(energy_series["quantitykwh"], errors="coerce")
energy_series = energy_series.dropna(subset=["quantitykwh"])
energy_series = energy_series.groupby("starttime")["quantitykwh"].sum()  # Aggregate duplicates

# -------------------------
# Download weather
# -------------------------
weather_df = download_era5(LAT, LON, YEAR)

# -------------------------
# Align data
# -------------------------
combined_df = pd.concat([weather_df[variable_weather], energy_series], axis=1, join="inner")
combined_df = combined_df.apply(pd.to_numeric, errors="coerce").dropna()
x = combined_df[variable_weather]
y = combined_df["quantitykwh"]

# -------------------------
# Sliding window correlation
# -------------------------
def sliding_window_corr(x, y, lag=0, window=45):
    x_shifted = x.shift(lag)
    combined = pd.concat([x_shifted, y], axis=1).dropna()
    swc = combined.iloc[:,1].rolling(window, center=True).corr(combined.iloc[:,0])
    return swc

swc = sliding_window_corr(x, y, lag=lag, window=window)

if lag > 0:
    corr_value = np.corrcoef(y[lag:].values, x[:-lag].values)[0,1]
else:
    corr_value = np.corrcoef(y.values, x.values)[0,1]

# -------------------------
# Plot SWC with Plotly
# -------------------------
fig = go.Figure()
highlight_start = max(center - window//2, 0)
highlight_end = min(center + window//2, len(y))

# Energy
fig.add_trace(go.Scatter(y=y, x=y.index, mode="lines", name=f"{selected_group}"))
fig.add_trace(go.Scatter(y=y.iloc[highlight_start:highlight_end],
                         x=y.index[highlight_start:highlight_end],
                         mode="lines", line=dict(color="red"), name="Highlighted"))

# Weather
fig.add_trace(go.Scatter(y=x, x=x.index, mode="lines", name=f"{variable_weather}"))
fig.add_trace(go.Scatter(y=x.iloc[highlight_start:highlight_end],
                         x=x.index[highlight_start:highlight_end],
                         mode="lines", line=dict(color="red"), name="Highlighted"))

# SWC
fig.add_trace(go.Scatter(y=swc, x=swc.index, mode="lines", name="SWC"))

fig.update_layout(height=800, xaxis_title="Time", yaxis_title="Values / Correlation",
                  title=f"Sliding Window Correlation\nlag={lag}, window={window}, correlation={corr_value:.3f}")

st.plotly_chart(fig, use_container_width=True)
st.write(f"Correlation between **{selected_group}** and **{variable_weather}** lagged {lag} timepoints: **{corr_value:.3f}**")

# -------------------------
# Cross-correlation vs lag plot
# -------------------------
st.write("---")
st.header("Correlation vs Lag")

max_lag = st.slider("Max lag for correlation plot (hours)", 0, 100, 50)

corrs = []
lags = range(-max_lag, max_lag+1)
for l in lags:
    if l > 0:
        corr = np.corrcoef(y[l:].values, x[:-l].values)[0,1]
    elif l < 0:
        corr = np.corrcoef(y[:l].values, x[-l:].values)[0,1]
    else:
        corr = np.corrcoef(y.values, x.values)[0,1]
    corrs.append(corr)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=list(lags), y=corrs, mode="lines+markers", name="Correlation"))
fig2.update_layout(xaxis_title="Lag (hours)", yaxis_title="Correlation",
                   title=f"Correlation between **{selected_group}** and **{variable_weather}** vs lag")
st.plotly_chart(fig2, use_container_width=True)
