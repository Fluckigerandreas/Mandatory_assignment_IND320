import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions.weather_utils import download_era5_data


# --- Tabler (2003) components ---

def compute_Qupot(hourly_wind_speeds, dt=3600):
    """Potential wind-driven transport (Qupot) [kg/m] via u^3.8."""
    return sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847


def sector_index(direction):
    """Map wind direction (deg) to 16-sector index."""
    return int(((direction + 11.25) % 360) // 22.5)


def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    """Cumulative transport per 16 wind sectors (kg/m)."""
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        idx = sector_index(d)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors


def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    """Tabler components with controlling transport and annual Qt."""
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe
    Srwe = theta * Swe
    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
    else:
        Qinf = Qupot
    Qt = Qinf * (1 - 0.14 ** (F / T))
    return Qt


# --- Public API ---

def calculate_snow_drift(lat: float, lon: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """
    Download ERA5 data for the given coordinates and seasonal year, compute Qt (kg/m).
    The date range should represent one seasonal year (July 1 to June 30).
    """
    years = {start_date.year, end_date.year}
    dfs = [download_era5_data(lat, lon, y) for y in years]
    df = pd.concat(dfs).sort_values("time")

    # Ensure dataframe times are tz-aware in Europe/Oslo
    df["time"] = df["time"].dt.tz_convert("Europe/Oslo")

    # Make bounds tz-aware
    start_date = pd.Timestamp(start_date).tz_localize("Europe/Oslo") if start_date.tzinfo is None else start_date
    end_date = pd.Timestamp(end_date).tz_localize("Europe/Oslo") if end_date.tzinfo is None else end_date

    df_period = df[(df["time"] >= start_date) & (df["time"] <= end_date)].copy()
    if df_period.empty:
        return float("nan")

    # Hourly Swe: precipitation when temp < +1°C
    df_period["Swe_hourly"] = df_period.apply(
        lambda row: row["precipitation"] if row["temperature_2m"] < 1 else 0, axis=1
    )
    Swe_total = df_period["Swe_hourly"].sum()
    wind_speeds = df_period["wind_speed_10m"].tolist()

    T, F, theta = 3000, 30000, 0.5
    Qt = compute_snow_transport(T, F, theta, Swe_total, wind_speeds)
    return float(Qt)


def plot_wind_rose(lat: float, lon: float, start_year: int, end_year: int):
    """
    Download ERA5 data for the given coordinates and year range, plot wind rose.
    Returns a matplotlib Figure.
    """
    dfs = [download_era5_data(lat, lon, y) for y in range(start_year, end_year + 1)]
    df = pd.concat(dfs).sort_values("time")
    if df.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "No data in selected range", ha="center", va="center")
        plt.axis("off")
        return fig

    # Ensure dataframe times are tz-aware in Europe/Oslo
    df["time"] = df["time"].dt.tz_convert("Europe/Oslo")

    # Compute Swe and sector transports
    df["Swe_hourly"] = df.apply(
        lambda row: row["precipitation"] if row["temperature_2m"] < 1 else 0, axis=1
    )
    ws = df["wind_speed_10m"].tolist()
    wdir = df["wind_direction_10m"].tolist()
    sectors = compute_sector_transport(ws, wdir)

    # Overall Qt average
    Swe_total = df["Swe_hourly"].sum()
    Qt = compute_snow_transport(3000, 30000, 0.5, Swe_total, ws)

    # Plot rose
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    angles = np.deg2rad(np.arange(0, 360, 360 / 16))
    ax.bar(angles, np.array(sectors) / 1000.0, width=np.deg2rad(22.5), edgecolor="black")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    directions = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                  'S','SSW','SW','WSW','W','WNW','NW','NNW']
    ax.set_xticks(angles)
    ax.set_xticklabels(directions)
    ax.set_title(f"Wind rose {start_year}-{end_year}\nAvg Qt: {Qt/1000:.1f} tonnes/m")
    return fig