# ======================================================
# Page3.py — Streamlit with Plotly
# ======================================================
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

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

# --- Function to fetch ERA5 weather data ---
@st.cache_data
def load_data_api(lat, lon, year=2021, timezone="Europe/Oslo"):
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
    r = requests.get(url, params=params)
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.to_period("M")
    return df

# --- Page controls ---
st.header("Controls")

# City selection
city_option = st.selectbox("Select city:", [c["city"] for c in cities])
selected_city = next(c for c in cities if c["city"] == city_option)

# Load data for column and month options
df_sample = load_data_api(selected_city["lat"], selected_city["lon"])

# Variable selection
columns = ["All"] + list(df_sample.columns[1:-1])  # skip 'time' and 'month'
selected_column = st.selectbox("Select variable:", columns)

# Month range slider
unique_months = df_sample['month'].unique().astype(str).tolist()
month_range = st.select_slider(
    "Select months:",
    options=unique_months,
    value=(unique_months[0], unique_months[-1])
)
start, end = pd.Period(month_range[0]), pd.Period(month_range[1])

# --- Load filtered data ---
df = load_data_api(selected_city["lat"], selected_city["lon"])
filtered_df = df[(df['month'] >= start) & (df['month'] <= end)]

# --- Plotting ---
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

fig.update_layout(
    width=900,
    height=500,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

