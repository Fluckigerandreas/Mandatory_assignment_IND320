import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from shapely.geometry import shape, Point, Polygon, MultiPolygon
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests_cache

# ------------------ Streamlit page ------------------
st.set_page_config(layout="wide")
st.title("❄ Snow Drift Analyzer – ERA5 + Map Selection")

# ------------------ Load GeoJSON ------------------
geojson_path = "file.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

def extract_geojson_area(feature):
    props = feature.get("properties", {})
    for key in ["ElSpotOmr","Elspot_omr","ELSPOT_OMR"]:
        if key in props:
            return str(props[key])
    return None

# ------------------ Session state ------------------
if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = (59.663, 10.762)  # default NMBU Ås

# ------------------ Cached ERA5 downloader ------------------
session = requests_cache.CachedSession(".cache", expire_after=86400)

@st.cache_data(show_spinner="Downloading ERA5 weather...", persist=True)
def download_era5(lat, lon, start_year, end_year, timezone="Europe/Oslo"):
    dfs = []
    for year in range(start_year, end_year + 1):
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
        r = session.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        df["season"] = df.index.to_series().apply(lambda dt: dt.year if dt.month >= 7 else dt.year-1)
        dfs.append(df)
    return pd.concat(dfs) if dfs else pd.DataFrame()

# ------------------ Snow transport functions ------------------
def compute_Qupot(hourly_wind_speeds, dt=3600):
    return sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847

def sector_index(direction):
    return int(((direction + 11.25) % 360) // 22.5)

def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        idx = sector_index(d)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors

def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    Qupot = compute_Qupot(hourly_wind_speeds)
    Qspot = 0.5 * T * Swe
    Srwe = theta * Swe
    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
        control = "Snowfall controlled"
    else:
        Qinf = Qupot
        control = "Wind controlled"
    Qt = Qinf * (1 - 0.14 ** (F / T))
    return {"Qupot": Qupot, "Qspot": Qspot, "Srwe": Srwe, "Qinf": Qinf, "Qt": Qt, "Control": control}

def compute_yearly_results(df, T, F, theta):
    seasons = sorted(df['season'].unique())
    results_list = []
    for s in seasons:
        df_season = df[df['season']==s].copy()
        df_season["Swe_hourly"] = df_season.apply(lambda r: r["precipitation"] if r["temperature_2m"] < 1 else 0, axis=1)
        total_Swe = df_season["Swe_hourly"].sum()
        ws = df_season["wind_speed_10m"].tolist()
        result = compute_snow_transport(T,F,theta,total_Swe,ws)
        result["season"] = f"{s}-{s+1}"
        results_list.append(result)
    return pd.DataFrame(results_list)

def compute_monthly_results(df, T, F, theta):
    df = df.copy()
    df["Swe_hourly"] = df.apply(lambda r: r["precipitation"] if r["temperature_2m"] < 1 else 0, axis=1)
    monthly_groups = df.groupby([df.index.year, df.index.month])
    out = []
    for (year, month), g in monthly_groups:
        Swe = g["Swe_hourly"].sum()
        ws = g["wind_speed_10m"].tolist()
        if len(ws) == 0:
            continue
        result = compute_snow_transport(T,F,theta,Swe,ws)
        result["year"] = year
        result["month"] = month
        out.append(result)
    return pd.DataFrame(out)

def compute_average_sector(df):
    sectors_list = []
    for s, group in df.groupby('season'):
        group = group.copy()
        group["Swe_hourly"] = group.apply(lambda r: r["precipitation"] if r["temperature_2m"] < 1 else 0, axis=1)
        ws = group["wind_speed_10m"].tolist()
        wd = group["wind_direction_10m"].tolist()
        sectors_list.append(compute_sector_transport(ws, wd))
    return np.mean(sectors_list, axis=0)

def plot_wind_rose(avg_sector_values, overall_avg):
    dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW']
    theta = np.linspace(0,360,16,endpoint=False)
    r = np.array(avg_sector_values)/1000
    fig = go.Figure(go.Barpolar(
        r=r, theta=theta, width=[22.5]*16,
        marker_color=r, marker_line_color="black", marker_line_width=1
    ))
    fig.update_layout(polar=dict(
        radialaxis=dict(title="Qt (tonnes/m)"),
        angularaxis=dict(direction="clockwise", rotation=90, ticktext=dirs, tickvals=theta)
    ))
    st.plotly_chart(fig)

# ------------------ Map ------------------
m = folium.Map(location=st.session_state.clicked_point, zoom_start=6)
for feat in geojson_data.get("features", []):
    folium.GeoJson(feat, style_function=lambda f: {"fillColor":"blue","color":"blue","weight":1,"fillOpacity":0.2}).add_to(m)
folium.Marker(st.session_state.clicked_point, icon=folium.Icon(color="red")).add_to(m)
map_data = st_folium(m, width=900, height=500)

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.session_state.clicked_point = (lat, lon)

# ------------------ User Inputs ------------------
start_year = st.number_input("Start Year", 2021, 2024, 2021)
end_year = st.number_input("End Year", start_year, 2024, 2022)

T = 3000
F = 30000
theta = 0.5

# ------------------ Download ERA5 ------------------
df_all = download_era5(*st.session_state.clicked_point, start_year, end_year)

if df_all.empty:
    st.warning("No ERA5 data available for this location/year range.")
else:
    # --- Yearly Qt ---
    yearly_df = compute_yearly_results(df_all, T,F,theta)
    st.subheader("📘 Yearly Snow Drift Qt")
    if not yearly_df.empty:
        yearly_df["Qt (tonnes/m)"] = yearly_df["Qt"]/1000
        st.dataframe(yearly_df[["season","Qt (tonnes/m)","Control"]])
        fig_year = go.Figure(go.Bar(x=yearly_df["season"], y=yearly_df["Qt (tonnes/m)"], marker_color="skyblue"))
        fig_year.update_layout(title="Yearly Snow Drift Qt", yaxis_title="Qt (tonnes/m)")
        st.plotly_chart(fig_year)

    # --- Monthly Qt ---
    monthly_df = compute_monthly_results(df_all, T,F,theta)
    if not monthly_df.empty:
        monthly_df["Qt (tonnes/m)"] = monthly_df["Qt"]/1000
        monthly_df["month_str"] = monthly_df["year"].astype(str)+"-"+monthly_df["month"].astype(str).str.zfill(2)
        st.subheader("📗 Monthly Snow Drift Qt")
        st.dataframe(monthly_df[["month_str","Qt (tonnes/m)","Control"]])
        fig_month = go.Figure(go.Scatter(x=monthly_df["month_str"], y=monthly_df["Qt (tonnes/m)"],
                                         mode="lines+markers", name="Monthly Qt"))
        fig_month.update_layout(title="Monthly Qt", yaxis_title="Qt (tonnes/m)")
        st.plotly_chart(fig_month)

        # --- Combined Monthly + Yearly Qt ---
        fig_comb = go.Figure()
        fig_comb.add_trace(go.Scatter(x=monthly_df["month_str"], y=monthly_df["Qt (tonnes/m)"],
                                      mode="lines+markers", name="Monthly"))
        if not yearly_df.empty:
            fig_comb.add_trace(go.Bar(x=yearly_df["season"], y=yearly_df["Qt (tonnes/m)"],
                                      name="Yearly", opacity=0.35))
        fig_comb.update_layout(title="Combined Monthly + Yearly Qt", yaxis_title="Qt (tonnes/m)")
        st.plotly_chart(fig_comb)

    # --- Wind Rose ---
    st.subheader("🟣 Wind Rose")
    avg_sectors = compute_average_sector(df_all)
    overall_avg = yearly_df["Qt"].mean() if not yearly_df.empty else 0
    plot_wind_rose(avg_sectors, overall_avg)

