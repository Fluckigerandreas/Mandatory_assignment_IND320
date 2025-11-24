import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pymongo import MongoClient
import certifi
import requests

# -------------------------
# Data Loading (cached)
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
    df["season"] = df.index.to_series().apply(lambda dt: dt.year if dt.month >= 7 else dt.year - 1)
    
    # Ensure numeric
    numeric_cols = ["temperature_2m", "precipitation", "wind_speed_10m",
                    "wind_direction_10m", "wind_gusts_10m"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    return df

@st.cache_data(show_spinner="Loading production data...")
def load_production():
    client = MongoClient(st.secrets["mongo"]["uri"], tls=True, tlsCAFile=certifi.where())
    db = client["Elhub"]
    df = pd.DataFrame(list(db["Data"].find()))
    if df.empty:
        return df
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce", utc=True)
    df = df.dropna(subset=["starttime"])
    if "pricearea" in df.columns:
        df["pricearea"] = df["pricearea"].apply(lambda x: x if x else "NO")
    
    df["quantitykwh"] = pd.to_numeric(df["quantitykwh"], errors="coerce")
    df = df.dropna(subset=["quantitykwh"])
    
    df = df.groupby(["pricearea", "productiongroup", "starttime"], as_index=False).agg({"quantitykwh": "sum"})
    df.set_index("starttime", inplace=True)
    
    # Aggregate duplicates
    df = df.groupby(df.index).sum()
    return df

@st.cache_data(show_spinner="Loading consumption data...")
def load_consumption():
    client = MongoClient(st.secrets["mongo"]["uri"], tls=True, tlsCAFile=certifi.where())
    db = client["Consumption_Elhub"]
    df = pd.DataFrame(list(db["Data"].find()))
    if df.empty:
        return df
    df["starttime"] = pd.to_datetime(df["starttime"], utc=True)
    if "pricearea" in df.columns:
        df["pricearea"] = df["pricearea"].apply(lambda x: x if x else "NO")
    
    df["quantitykwh"] = pd.to_numeric(df["quantitykwh"], errors="coerce")
    df = df.dropna(subset=["quantitykwh"])
    
    df = df.groupby(["pricearea", "consumptiongroup", "starttime"], as_index=False).agg({"quantitykwh": "sum"})
    df.set_index("starttime", inplace=True)
    
    # Aggregate duplicates
    df = df.groupby(df.index).sum()
    return df

# -------------------------
# Load Data
# -------------------------
st.title("Sliding Window Correlation: Meteorology vs Energy")

weather_df = download_era5(lat=59.91, lon=10.75, year=2023)
prod_df = load_production()
cons_df = load_consumption()

# -------------------------
# Sidebar selectors
# -------------------------
st.sidebar.header("Settings")
variable_weather = st.sidebar.selectbox("Select meteorological variable", weather_df.columns)
variable_energy_type = st.sidebar.radio("Select energy type", ["Production", "Consumption"])
if variable_energy_type == "Production":
    variable_energy = st.sidebar.selectbox("Select production group", prod_df.columns)
    energy_df = prod_df
else:
    variable_energy = st.sidebar.selectbox("Select consumption group", cons_df.columns)
    energy_df = cons_df

lag = st.sidebar.slider("Lag (hours)", 0, 100, 0)
window = st.sidebar.slider("Sliding window length", 1, 60, 45)
center = st.sidebar.slider("Center index for window", window//2, len(weather_df)-window//2, len(weather_df)//2)

# -------------------------
# Align Data
# -------------------------
combined_df = pd.concat([weather_df[variable_weather], energy_df[variable_energy]], axis=1, join="inner")
combined_df = combined_df.apply(pd.to_numeric, errors="coerce").dropna()
x = combined_df[variable_weather]
y = combined_df[variable_energy]

# -------------------------
# Sliding Window Correlation
# -------------------------
def sliding_window_corr(x, y, lag=0, window=45):
    if lag > 0:
        x_lagged = x.shift(lag).iloc[window-1:]
        y_trunc = y.iloc[window-1:]
    else:
        x_lagged = x.iloc[window-1:]
        y_trunc = y.iloc[window-1:]
    swc = y_trunc.rolling(window, center=True).corr(x_lagged)
    return swc

swc = sliding_window_corr(x, y, lag=lag, window=window)
corr_value = np.corrcoef(y[lag:].values, x[:-lag].values if lag>0 else x.values)[0,1]

# -------------------------
# Plot with Plotly
# -------------------------
fig = go.Figure()

# Top plot: Energy
fig.add_trace(go.Scatter(y=y, x=y.index, mode="lines", name=f"{variable_energy}"))
highlight_start = max(center-window//2, 0)
highlight_end = min(center+window//2, len(y))
fig.add_trace(go.Scatter(y=y.iloc[highlight_start:highlight_end],
                         x=y.index[highlight_start:highlight_end],
                         mode="lines", line=dict(color="red"), name="Highlighted"))

# Middle plot: Weather
fig.add_trace(go.Scatter(y=x, x=x.index, mode="lines", name=f"{variable_weather}"))
fig.add_trace(go.Scatter(y=x.iloc[highlight_start:highlight_end],
                         x=x.index[highlight_start:highlight_end],
                         mode="lines", line=dict(color="red"), name="Highlighted"))

# Bottom plot: Sliding window correlation
fig.add_trace(go.Scatter(y=swc, x=swc.index, mode="lines", name="SWC"))

# Safely highlight center of SWC
center_swc = min(center, len(swc)-1)
if len(swc) > 0:
    fig.add_trace(go.Scatter(y=[swc.iloc[center_swc]],
                             x=[swc.index[center_swc]],
                             mode="markers",
                             marker=dict(color="red", size=10)))

fig.update_layout(height=800, xaxis_title="Time", yaxis_title="Values / Correlation",
                  title=f"Sliding Window Correlation (lag={lag}, window={window})\nCorrelation={corr_value:.3f}")

st.plotly_chart(fig, use_container_width=True)

st.write(f"Correlation between **{variable_energy}** and **{variable_weather}** lagged {lag} timepoints: **{corr_value:.3f}**")
