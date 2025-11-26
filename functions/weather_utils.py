import streamlit as st
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests

# Functions for weather data handling

# Price area ↔ city mapping
areas_df = pd.DataFrame({
    "price_area": ["NO1", "NO2", "NO3", "NO4", "NO5"],
    "city": ["Oslo", "Kristiansand", "Trondheim", "Tromsø", "Bergen"],
    "latitude": [59.91, 58.15, 63.43, 69.65, 60.39],
    "longitude": [10.75, 7.99, 10.40, 18.96, 5.32]
})


# Helper functions
def get_city_from_area(price_area: str):
    """Return (city, latitude, longitude) for a given price area code."""
    row = areas_df.loc[areas_df["price_area"] == price_area]
    if row.empty:
        raise ValueError(f"Price area '{price_area}' not found.")
    return (
        row["city"].values[0],
        row["latitude"].values[0],
        row["longitude"].values[0]
    )


# Download ERA5 data
@st.cache_data(show_spinner=True)
def download_era5_data(latitude: float, longitude: float, year: int) -> pd.DataFrame:
    """Download ERA5 data from Open-Meteo for a given location and year."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # ERA5 API parameters
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m"
        ],
        "timezone": "Europe/Oslo"
    }

    # Fetch data
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    variables = params["hourly"]

    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    for i, var in enumerate(variables):
        hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(hourly_data)

    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("Europe/Oslo")
    df = df[df["time"].dt.year == year]


    return df