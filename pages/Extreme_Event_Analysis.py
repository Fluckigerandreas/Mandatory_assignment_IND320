# ======================================================
# NewB_single_file_plotly.py — Streamlit page
# ======================================================
import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.neighbors import LocalOutlierFactor
import requests_cache
from retry_requests import retry
import openmeteo_requests
import plotly.graph_objects as go

# ======================================================
# PRICE AREAS (CITIES)
# ======================================================
price_areas = [
    {"price_area": "NO1", "city": "Oslo", "latitude": 59.9139, "longitude": 10.7522},
    {"price_area": "NO2", "city": "Kristiansand", "latitude": 58.1467, "longitude": 7.9956},
    {"price_area": "NO3", "city": "Trondheim", "latitude": 63.4305, "longitude": 10.3951},
    {"price_area": "NO4", "city": "Tromsø", "latitude": 69.6492, "longitude": 18.9553},
    {"price_area": "NO5", "city": "Bergen", "latitude": 60.3913, "longitude": 5.3221},
]
cities_df = pd.DataFrame(price_areas)

# ======================================================
# ERA5 WEATHER DATA DOWNLOAD
# ======================================================
@st.cache_data(show_spinner="Downloading weather data...")
def download_era5_openmeteo(lat, lon, year, timezone="Europe/Oslo"):
    """Download ERA5 hourly weather data from Open-Meteo."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
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

    df = pd.DataFrame(
        {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "precipitation": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
            "wind_gusts_10m": hourly.Variables(3).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(4).ValuesAsNumpy(),
        }
    )
    df["time"] = df["time"].dt.tz_convert(timezone)
    df.set_index("time", inplace=True)
    return df

# ======================================================
# TEMPERATURE OUTLIERS (Highpass–Lowpass Filter + Trend SPC)
# ======================================================
def detect_temperature_outliers_filter(df, temp_col="temperature_2m", cutoff_hours=400,
                                       sample_rate_hours=1, n_std=2.0):
    s = df[temp_col].dropna().sort_index()
    x = s.values.astype(float)

    nyquist = 0.5 / sample_rate_hours
    cutoff_freq = 1 / cutoff_hours
    normal_cutoff = cutoff_freq / nyquist

    b, a = butter(N=4, Wn=normal_cutoff, btype="low")
    trend = filtfilt(b, a, x)

    residual = x - trend
    sigma_hat = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    upper = trend + n_std * sigma_hat
    lower = trend - n_std * sigma_hat
    mask = (x > upper) | (x < lower)
    outliers = pd.DataFrame({"temperature": x[mask]}, index=s.index[mask])

    # --- Plotly Interactive ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=x, mode="lines", name="Temperature (°C)", line=dict(color="blue", width=1.5)))
    fig.add_trace(go.Scatter(x=s.index, y=trend, mode="lines", name="Trend", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=s.index, y=upper, mode="lines", line=dict(color="orange", width=0.5), name="SPC upper"))
    fig.add_trace(go.Scatter(x=s.index, y=lower, mode="lines", line=dict(color="orange", width=0.5), name="SPC lower", fill='tonexty', fillcolor='rgba(255,165,0,0.2)'))
    fig.add_trace(go.Scatter(x=outliers.index, y=outliers["temperature"], mode="markers", name=f"Outliers ({len(outliers)})", marker=dict(color="red", size=6)))

    fig.update_layout(title="Temperature Outliers (Highpass–Lowpass + SPC)",
                      xaxis_title="Time", yaxis_title="Temperature (°C)",
                      template="plotly_white", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

    return outliers

# ======================================================
# PRECIPITATION LOF ANOMALIES
# ======================================================
def detect_precipitation_lof(df, precip_col="precipitation", contamination=0.01):
    p = df[precip_col].fillna(0).sort_index()
    nonzero_mask = p.values > 0
    X_nonzero = np.log1p(p.values[nonzero_mask]).reshape(-1, 1)

    if len(X_nonzero) == 0:
        st.warning("No non-zero precipitation values to analyze.")
        return pd.DataFrame(columns=[precip_col])

    n_neighbors = min(len(X_nonzero) - 1, 20)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(X_nonzero)

    outliers = pd.DataFrame({precip_col: p.values[nonzero_mask][y_pred == -1]},
                            index=p.index[nonzero_mask][y_pred == -1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p.values, mode="lines", name="Precipitation (mm)", line=dict(color="blue", width=1.2)))
    fig.add_trace(go.Scatter(x=outliers.index, y=outliers[precip_col], mode="markers",
                             name=f"LOF anomalies ({len(outliers)})",
                             marker=dict(color="red", size=6, line=dict(width=1, color="darkred"))))
    fig.update_layout(title=f"Precipitation Anomalies (LOF, contamination={contamination:.3f})",
                      xaxis_title="Time", yaxis_title="Precipitation (mm)",
                      template="plotly_white", hovermode="x unified", height=450)
    st.plotly_chart(fig, use_container_width=True)

    return outliers

# ======================================================
# STREAMLIT PAGE
# ======================================================
st.title("Outlier & Anomaly Analysis (Weather Data)")

city_name = st.selectbox("Select city", [c["city"] for c in price_areas])
city_info = next(c for c in price_areas if c["city"] == city_name)
year = st.number_input("Select year", min_value=2000, max_value=2025, value=2021)

weather_df = download_era5_openmeteo(city_info["latitude"], city_info["longitude"], year)
st.write(f"✅ Loaded weather data for {city_name} ({len(weather_df)} rows)")

# Tabs
tab1, tab2 = st.tabs(["Temperature Outliers (SPC)", "Precipitation Anomalies (LOF)"])

with tab1:
    st.header("Temperature Outliers")
    n_std = st.number_input("Number of standard deviations", min_value=0.1, value=2.0, step=0.1)
    cutoff_hours = st.number_input("Cutoff hours for smoothing", min_value=1, value=400, step=1)
    temp_outliers = detect_temperature_outliers_filter(weather_df, cutoff_hours=cutoff_hours, n_std=n_std)
    st.write(f"Total outliers detected: {len(temp_outliers)}")
    st.dataframe(temp_outliers.head(20))

with tab2:
    st.header("Precipitation Anomalies")
    contamination = st.slider("Proportion of anomalies (contamination)", min_value=0.001, max_value=0.1, value=0.01, step=0.01)
    precip_outliers = detect_precipitation_lof(weather_df, contamination=contamination)
    st.write(f"Total anomalies detected: {len(precip_outliers)}")
    st.dataframe(precip_outliers.head(20))
