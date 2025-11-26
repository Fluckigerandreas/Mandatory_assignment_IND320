import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

st.set_page_config(page_title="Weather Data Plot", page_icon="📈")
st.title("📊 Weather Data Visualization")

# --- City definitions ---
cities = [
    {"city": "Oslo", "lat": 59.9139, "lon": 10.7522},
    {"city": "Kristiansand", "lat": 58.1467, "lon": 7.9956},
    {"city": "Trondheim", "lat": 63.4305, "lon": 10.3951},
    {"city": "Tromsø", "lat": 69.6492, "lon": 18.9553},
    {"city": "Bergen", "lat": 60.3913, "lon": 5.3221},
]

# --- Robust API fetch with retries ---
def fetch_era5(lat, lon, year=2021, timezone="Europe/Oslo"):
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
            "wind_direction_10m"
        ],
        "models": "era5",
        "timezone": timezone
    }
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    r = session.get(url, params=params, timeout=20)  # shorter timeout
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.to_period("M")
    return df

# --- Cache data per city ---
@st.cache_data(show_spinner=True)
def load_city_data(city_name):
    city = next(c for c in cities if c["city"] == city_name)
    return fetch_era5(city["lat"], city["lon"])

# --- Controls ---
st.header("Controls")

city_option = st.selectbox("Select city:", [c["city"] for c in cities])

try:
    df = load_city_data(city_option)
except requests.exceptions.RequestException as e:
    st.error(f"Failed to load data for {city_option}: {e}")
    st.stop()

if df.empty:
    st.warning("No data available for this city.")
    st.stop()

columns = ["All"] + list(df.columns[1:-1])
selected_column = st.selectbox("Select variable:", columns)

unique_months = df['month'].unique().astype(str).tolist()
month_range = st.select_slider(
    "Select months:",
    options=unique_months,
    value=(unique_months[0], unique_months[-1])
)
start, end = pd.Period(month_range[0]), pd.Period(month_range[1])

filtered_df = df[(df['month'] >= start) & (df['month'] <= end)]
if selected_column != "All":
    filtered_df = filtered_df[["time", selected_column]]

st.subheader("Weather Data Plot")

if selected_column == "All":
    chart_data = filtered_df.melt(
        id_vars=["time"], 
        value_vars=df.columns[1:-1], 
        var_name="Variable", 
        value_name="Value"
    )
    fig = px.line(
        chart_data,
        x="time",
        y="Value",
        color="Variable",
        labels={"time": "Time", "Value": "Value"},
        title="All Weather Variables"
    )
else:
    fig = px.line(
        filtered_df,
        x="time",
        y=selected_column,
        labels={"time": "Time", selected_column: selected_column},
        title=f"{selected_column} over Time"
    )
    fig.update_traces(mode="lines+markers", hovertemplate="%{x}<br>%{y}")

fig.update_layout(width=900, height=500, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
