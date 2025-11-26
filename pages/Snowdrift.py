import re
import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from shapely.geometry import shape, Point, Polygon, MultiPolygon
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests_cache
from retry_requests import retry
import openmeteo_requests
import branca

# ------------------ Streamlit page ------------------
st.set_page_config(layout="wide")
st.title("Norway Price Areas Map + Snow Drift Analyzer")

# ================== Load GeoJSON ===================
geojson_path = "file.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ================== Normalization helper ===================
def normalize_to_NO(code):
    if code is None:
        return None
    if isinstance(code, int):
        return f"NO{code}"
    s = str(code).strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    for pattern in [r"^N0?([1-9])$", r"^NO0?([1-9])$", r"^0?([1-9])$"]:
        m = re.match(pattern, s)
        if m:
            return f"NO{m.group(1)}"
    return None

def extract_geojson_area(feature):
    props = feature.get("properties", {})
    for k in ["ElSpotOmr","Elspot_omr","ELSPOT_OMR","ElSpotOmråde","ELSPOT_OMRADE"]:
        if k in props:
            return normalize_to_NO(props[k])
    for v in props.values():
        if isinstance(v, (str,int)) and normalize_to_NO(v):
            return normalize_to_NO(v)
    return None

# ================== Session state ===================
if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = (59.663, 10.762)  # default NMBU Ås
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None

# ================== Data loader (example placeholders) ===================
# Replace these with your actual Data_loader functions
def load_production(): return pd.DataFrame()
def load_consumption(): return pd.DataFrame()

prod_df = load_production()
cons_df = load_consumption()

# ================== Layout: Left (Map) + Right (Snow Drift) ===================
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("🗺 Norway Price Areas Map")
    m = folium.Map(location=st.session_state.clicked_point, zoom_start=6)

    # Compute mean per area for coloring
    data_type = st.radio("Select data type:", ["Production", "Consumption"], horizontal=True)
    df = prod_df if data_type=="Production" else cons_df
    group_col = "productiongroup" if data_type=="Production" else "consumptiongroup"

    if not df.empty and group_col in df.columns:
        selected_group = st.selectbox("Select group:", sorted(df[group_col].dropna().unique()))
        selected_year = st.selectbox("Select year:", [2021,2022,2023,2024])
        df_group = df[df[group_col]==selected_group].copy()
        df_group.index = pd.to_datetime(df_group.index)
        df_group["pricearea"] = df_group["pricearea"].apply(normalize_to_NO)
        area_means = df_group.groupby("pricearea")["quantitykwh"].mean().to_dict()
        vals = list(area_means.values())
        if vals:
            colormap = branca.colormap.LinearColormap(
                colors=["#d73027", "#fee08b", "#1a9850"],
                vmin=min(vals), vmax=max(vals),
                caption=f"Mean quantity kWh for {selected_group} ({selected_year})"
            )
        else:
            area_means = {}

    # Add polygons
    for feat in geojson_data.get("features", []):
        area = extract_geojson_area(feat)
        fill = "#dddddd"
        if area in area_means:
            fill = colormap(area_means[area])
        if st.session_state.selected_area == area:
            style = {"fillColor": fill, "color": "red", "weight": 3, "fillOpacity":0.65}
        else:
            style = {"fillColor": fill, "color": "#3333cc", "weight":1, "fillOpacity":0.55}
        folium.GeoJson(feat, style_function=lambda f, s=style: s).add_to(m)

    folium.Marker(st.session_state.clicked_point, icon=folium.Icon(color="red")).add_to(m)
    if 'colormap' in locals(): colormap.add_to(m)
    map_data = st_folium(m, width=600, height=700)

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.session_state.clicked_point = (lat, lon)

with col2:
    st.subheader("❄ Snow Drift Analyzer")
    start_year = st.number_input("Start Year", 2020, 2024, 2021)
    end_year = st.number_input("End Year", start_year, 2024, 2022)
    T = 3000
    F = 30000
    theta = 0.5

    @st.cache_data(show_spinner="Downloading ERA5 weather...", persist=True)
    def download_era5(lat, lon, start_year, end_year, timezone="Europe/Oslo"):
        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        client = openmeteo_requests.Client(session=retry_session)
        all_years = []
        for year in range(start_year, end_year+1):
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
                "hourly":["temperature_2m","precipitation","wind_speed_10m","wind_gusts_10m","wind_direction_10m"],
                "models":"era5","timezone":timezone
            }
            response = client.weather_api(url, params=params)[0]
            hourly = response.Hourly()
            times = np.atleast_1d(hourly.Time())
            if len(times)==0: continue
            start_ts = pd.to_datetime(times[0], unit="s", utc=True)
            end_ts = pd.to_datetime(np.atleast_1d(hourly.TimeEnd())[0], unit="s", utc=True)
            interval_seconds = hourly.Interval()
            df = pd.DataFrame({
                "time": pd.date_range(start=start_ts, end=end_ts, freq=pd.Timedelta(seconds=interval_seconds), inclusive="left"),
                "temperature_2m": np.atleast_1d(hourly.Variables(0).ValuesAsNumpy()),
                "precipitation": np.atleast_1d(hourly.Variables(1).ValuesAsNumpy()),
                "wind_speed_10m": np.atleast_1d(hourly.Variables(2).ValuesAsNumpy()),
                "wind_gusts_10m": np.atleast_1d(hourly.Variables(3).ValuesAsNumpy()),
                "wind_direction_10m": np.atleast_1d(hourly.Variables(4).ValuesAsNumpy()),
            })
            df["time"] = df["time"].dt.tz_convert(timezone)
            df.set_index("time", inplace=True)
            df["season"] = df.index.to_series().apply(lambda dt: dt.year if dt.month>=7 else dt.year-1)
            all_years.append(df)
        return pd.concat(all_years) if all_years else pd.DataFrame()

    df_all = download_era5(*st.session_state.clicked_point, start_year, end_year)

    if df_all.empty:
        st.warning("No ERA5 data available for this location/year range.")
    else:
        # --- Yearly Qt ---
        def compute_Qupot(hourly_wind_speeds, dt=3600):
            return sum((u**3.8)*dt for u in hourly_wind_speeds)/233847

        def sector_index(direction):
            return int(((direction+11.25)%360)//22.5)

        def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
            sectors = [0.0]*16
            for u,d in zip(hourly_wind_speeds,hourly_wind_dirs):
                sectors[sector_index(d)] += ((u**3.8)*dt)/233847
            return sectors

        def compute_snow_transport(T,F,theta,Swe, hourly_wind_speeds, dt=3600):
            Qupot = compute_Qupot(hourly_wind_speeds)
            Qspot = 0.5*T*Swe
            Srwe = theta*Swe
            if Qupot > Qspot:
                Qinf = 0.5*T*Srwe
                control = "Snowfall controlled"
            else:
                Qinf = Qupot
                control = "Wind controlled"
            Qt = Qinf*(1-0.14**(F/T))
            return {"Qupot":Qupot,"Qspot":Qspot,"Srwe":Srwe,"Qinf":Qinf,"Qt":Qt,"Control":control}

        def compute_yearly_results(df,T,F,theta):
            seasons = sorted(df['season'].unique())
            results_list = []
            for s in seasons:
                df_s = df[df['season']==s].copy()
                df_s["Swe_hourly"] = df_s.apply(lambda r: r["precipitation"] if r["temperature_2m"]<1 else 0, axis=1)
                total_Swe = df_s["Swe_hourly"].sum()
                ws = df_s["wind_speed_10m"].tolist()
                result = compute_snow_transport(T,F,theta,total_Swe,ws)
                result["season"] = f"{s}-{s+1}"
                results_list.append(result)
            return pd.DataFrame(results_list)

        yearly_df = compute_yearly_results(df_all,T,F,theta)
        st.subheader("📘 Yearly Snow Drift Qt")
        if not yearly_df.empty:
            yearly_df["Qt (tonnes/m)"] = yearly_df["Qt"]/1000
            st.dataframe(yearly_df[["season","Qt (tonnes/m)","Control"]])
