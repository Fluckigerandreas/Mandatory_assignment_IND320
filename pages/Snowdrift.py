import re
import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests_cache
from retry_requests import retry
import openmeteo_requests

# ------------------ Streamlit page ------------------
st.set_page_config(layout="wide")
st.title("Norway Map + Snow Drift Analyzer with Trend & Wind Rose")

# ------------------ Load GeoJSON ------------------
geojson_path = "file.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ------------------ Session state ------------------
if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = (59.663, 10.762)  # default NMBU Ås

# ------------------ Layout: Map | Snow Drift ------------------
col1, col2 = st.columns([1,1])

# ------------------ LEFT: MAP ------------------
with col1:
    st.subheader("🗺 Norway Map")
    m = folium.Map(location=st.session_state.clicked_point, zoom_start=6)
    for feat in geojson_data.get("features", []):
        folium.GeoJson(feat, style_function=lambda f: {
            "fillColor": "#dddddd", "color": "#3333cc", "weight": 1, "fillOpacity": 0.2
        }).add_to(m)
    folium.Marker(st.session_state.clicked_point, icon=folium.Icon(color="red")).add_to(m)
    map_data = st_folium(m, width=600, height=700)

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.session_state.clicked_point = (lat, lon)

    st.write(f"Clicked coordinates: {st.session_state.clicked_point}")

# ------------------ RIGHT: SNOW DRIFT ------------------
with col2:
    st.subheader("❄ Snow Drift Analyzer")
    start_year = st.number_input("Start Year", 2020, 2024, 2021)
    end_year = st.number_input("End Year", start_year, 2024, 2022)
    T, F, theta = 3000, 30000, 0.5

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
                "start_date": f"{year}-01-01",
                "end_date": f"{year}-12-31",
                "hourly": ["temperature_2m","precipitation","wind_speed_10m",
                           "wind_gusts_10m","wind_direction_10m"],
                "models":"era5","timezone": timezone
            }
            response = client.weather_api(url, params=params)[0]
            hourly = response.Hourly()
            times = np.atleast_1d(hourly.Time())
            if len(times)==0: continue

            var_arrays = []
            for i in range(5):
                arr = np.atleast_1d(hourly.Variables(i).ValuesAsNumpy())
                # pad with NaN if length mismatch
                if len(arr) < len(times):
                    arr = np.pad(arr, (0,len(times)-len(arr)), constant_values=np.nan)
                var_arrays.append(arr[:len(times)])

            df = pd.DataFrame({
                "time": pd.to_datetime(times, unit="s", utc=True),
                "temperature_2m": var_arrays[0],
                "precipitation": var_arrays[1],
                "wind_speed_10m": var_arrays[2],
                "wind_gusts_10m": var_arrays[3],
                "wind_direction_10m": var_arrays[4],
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
        # ------------------ Snow Drift Calculations ------------------
        def compute_Qupot(ws, dt=3600): return sum(u**3.8*dt for u in ws)/233847
        def sector_index(d): return int(((d+11.25)%360)//22.5)
        def compute_sector_transport(ws, wd, dt=3600):
            sectors = [0.0]*16
            for u,d in zip(ws, wd):
                sectors[sector_index(d)] += (u**3.8*dt)/233847
            return sectors
        def compute_snow_transport(T,F,theta,Swe, ws): 
            Qupot = compute_Qupot(ws)
            Qspot = 0.5*T*Swe
            Srwe = theta*Swe
            Qinf = 0.5*T*Srwe if Qupot>Qspot else Qupot
            control = "Snowfall controlled" if Qupot>Qspot else "Wind controlled"
            Qt = Qinf*(1-0.14**(F/T))
            return {"Qupot":Qupot,"Qspot":Qspot,"Srwe":Srwe,"Qinf":Qinf,"Qt":Qt,"Control":control}

        # Yearly
        def compute_yearly(df):
            results=[]
            for s in sorted(df['season'].unique()):
                df_s = df[df['season']==s].copy()
                df_s["Swe_hourly"]=df_s.apply(lambda r:r["precipitation"] if r["temperature_2m"]<1 else 0, axis=1)
                total_Swe=df_s["Swe_hourly"].sum()
                ws=df_s["wind_speed_10m"].tolist()
                res=compute_snow_transport(T,F,theta,total_Swe,ws)
                res["season"]=f"{s}-{s+1}"
                results.append(res)
            return pd.DataFrame(results)

        # Monthly
        def compute_monthly(df):
            df = df.copy()
            df["Swe_hourly"]=df.apply(lambda r:r["precipitation"] if r["temperature_2m"]<1 else 0, axis=1)
            out=[]
            for (y,m), g in df.groupby([df.index.year, df.index.month]):
                Swe=g["Swe_hourly"].sum()
                ws=g["wind_speed_10m"].tolist()
                if len(ws)==0: continue
                res=compute_snow_transport(T,F,theta,Swe,ws)
                res["year"]=y
                res["month"]=m
                out.append(res)
            return pd.DataFrame(out)

        yearly_df = compute_yearly(df_all)
        monthly_df = compute_monthly(df_all)

        # ------------------ Display Yearly ------------------
        st.subheader("📘 Yearly Qt")
        if not yearly_df.empty:
            yearly_df["Qt (tonnes/m)"]=yearly_df["Qt"]/1000
            st.dataframe(yearly_df[["season","Qt (tonnes/m)","Control"]])

        # ------------------ Display Monthly + Trend Line ------------------
        st.subheader("📗 Monthly Qt")
        if not monthly_df.empty:
            monthly_df["Qt (tonnes/m)"]=monthly_df["Qt"]/1000
            monthly_df["month_str"]=monthly_df["year"].astype(str)+"-"+monthly_df["month"].astype(str).str.zfill(2)
            st.dataframe(monthly_df[["month_str","Qt (tonnes/m)","Control"]])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_df["month_str"], y=monthly_df["Qt (tonnes/m)"], 
                                     mode="lines+markers", name="Monthly Qt", line=dict(color="skyblue")))
            fig.update_layout(title="Monthly Snow Drift Qt Trend", yaxis_title="Qt (tonnes/m)")
            st.plotly_chart(fig)

            # ------------------ Wind Rose ------------------
            st.subheader("🟣 Wind Rose")
            sectors_list=[]
            for s, g in df_all.groupby('season'):
                g["Swe_hourly"]=g.apply(lambda r:r["precipitation"] if r["temperature_2m"]<1 else 0, axis=1)
                ws=g["wind_speed_10m"].tolist()
                wd=g["wind_direction_10m"].tolist()
                sectors_list.append(compute_sector_transport(ws, wd))
            avg_sectors=np.mean(sectors_list, axis=0)
            dirs=['N','NNE','NE','ENE','E','ESE','SE','SSE',
                  'S','SSW','SW','WSW','W','WNW','NW','NNW']
            theta=np.linspace(0,360,16,endpoint=False)
            fig_wind=go.Figure(go.Barpolar(
                r=avg_sectors/1000, theta=theta, width=[22.5]*16,
                marker_color=avg_sectors/1000, marker_line_color="black", marker_line_width=1
            ))
            fig_wind.update_layout(polar=dict(
                radialaxis=dict(title="Qt (tonnes/m)"),
                angularaxis=dict(direction="clockwise", rotation=90, ticktext=dirs, tickvals=theta)
            ))
            st.plotly_chart(fig_wind)
