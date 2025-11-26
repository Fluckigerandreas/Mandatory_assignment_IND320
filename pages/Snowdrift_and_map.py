# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
from Data_loader import load_production, load_consumption
from functions.snow_drift import calculate_snow_drift, plot_wind_rose
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Energy Map & Snow Drift Explorer")

# ==============================================================================
# Normalize function for GeoJSON and dataframe keys
# ==============================================================================
def normalize_area_name(name):
    if isinstance(name, str):
        name = name.replace(" ", "")
        if name.startswith("N0") and len(name) == 3:
            return "NO" + name[-1]
    return name

# ==============================================================================
# Load GeoJSON
# ==============================================================================
geojson_path = Path("file.geojson")
if not geojson_path.exists():
    st.error(f"GeoJSON file not found at {geojson_path}")
    st.stop()

with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

for feature in geojson_data["features"]:
    raw_name = feature["properties"].get("ElSpotOmr")
    feature["properties"]["ElSpotOmrNorm"] = normalize_area_name(raw_name)

# ==============================================================================
# Session state init
# ==============================================================================
if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = None
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None

# ==============================================================================
# UI – choose Production / Consumption
# ==============================================================================
data_type = st.radio("Select data type:", ["Production", "Consumption"], horizontal=True)

if data_type == "Production":
    df = load_production()
    group_col = "productiongroup"
else:
    df = load_consumption()
    group_col = "consumptiongroup"

if df.empty:
    st.warning(f"No {data_type} data loaded.")
    st.stop()

# --------------------------------------------------------------------------
# FIX: reset index so 'starttime' is a column (loaders set it as index)
# --------------------------------------------------------------------------
df = df.reset_index()

# Ensure required columns exist
required_cols = {"starttime", "pricearea", "quantitykwh", group_col}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Dataframe is missing required columns: {missing}")
    st.stop()

# ==============================================================================
# UI – group selection and time slider
# ==============================================================================
groups = sorted(df[group_col].dropna().unique().tolist())
group = st.selectbox("Select group:", groups)
days = st.slider("Time interval (days):", 1, 30, 7)

# ==============================================================================
# Filter dataframe
# ==============================================================================
df = df.copy()
df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
df["pricearea"] = df["pricearea"].apply(normalize_area_name)

end_time = df["starttime"].max()
start_time = end_time - timedelta(days=days)

df_period = df[
    (df["starttime"] >= start_time) &
    (df["starttime"] <= end_time) &
    (df[group_col] == group)
]

if df_period.empty:
    st.warning("No data available for this selection.")
    st.stop()

means_df = df_period.groupby("pricearea", as_index=False)["quantitykwh"].mean()
means_dict = dict(zip(means_df["pricearea"], means_df["quantitykwh"]))

# ==============================================================================
# Map
# ==============================================================================
m = folium.Map(location=[63.0, 10.5], zoom_start=5.5)

vmin = means_df["quantitykwh"].min()
vmax = means_df["quantitykwh"].max()
thresholds = np.linspace(vmin, vmax, 6).tolist() if not np.isclose(vmin, vmax) else [vmin-1e-6, vmin, vmax+1e-6]

folium.Choropleth(
    geo_data=geojson_data,
    name="choropleth",
    data=means_df,
    columns=["pricearea", "quantitykwh"],
    key_on="feature.properties.ElSpotOmrNorm",
    fill_color="YlGnBu",
    fill_opacity=0.6,
    line_opacity=0.3,
    line_color="black",
    legend_name=f"{data_type} mean quantity (kWh)",
    threshold_scale=thresholds,
    nan_fill_color="lightgray"
).add_to(m)

folium.GeoJson(
    geojson_data,
    name="tooltips",
    tooltip=folium.GeoJsonTooltip(
        fields=["ElSpotOmrNorm"],
        aliases=["Price area:"],
        labels=True,
        sticky=True
    ),
    style_function=lambda _: {"color": "transparent", "weight": 0, "fillOpacity": 0}
).add_to(m)

# Highlight clicked point and selected area
if st.session_state.clicked_point:
    folium.Marker(
        location=st.session_state.clicked_point,
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

if st.session_state.selected_area:
    def highlight_style(feat):
        return {"color": "#d62728", "weight": 4, "fillOpacity": 0} \
            if feat["properties"].get("ElSpotOmrNorm") == st.session_state.selected_area \
            else {"color": "transparent", "weight": 0, "fillOpacity": 0}

    folium.GeoJson(
        geojson_data,
        name="selected_highlight",
        style_function=highlight_style,
        tooltip=None,
    ).add_to(m)

map_data = st_folium(m, width=950, height=630)

# Capture click events
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.session_state.clicked_point = (lat, lon)

    point = Point(lon, lat)
    clicked_area = None
    for feature in geojson_data["features"]:
        geom = shape(feature["geometry"])
        if isinstance(geom, (Polygon, MultiPolygon)) and geom.contains(point):
            clicked_area = feature["properties"].get("ElSpotOmrNorm")
            break
    st.session_state.selected_area = clicked_area

# ==============================================================================
# Display values
# ==============================================================================
st.write("### Mean quantity (kWh) per NO area:")
st.dataframe(means_df)

if st.session_state.selected_area:
    val = means_dict.get(st.session_state.selected_area, None)
    if val is not None and not pd.isna(val):
        st.success(f"Selected area: **{st.session_state.selected_area}** → {val:.2f} kWh")
    else:
        st.success(f"Selected area: **{st.session_state.selected_area}** (no data)")

if st.session_state.clicked_point:
    st.write(f"Clicked coordinates: {st.session_state.clicked_point}")

# ==============================================================================
# Snow Drift Section
# ==============================================================================
st.write("---")
st.header("❄️ Snow Drift Explorer")

if st.session_state.clicked_point:
    lat, lon = st.session_state.clicked_point
    st.write(f"Using coordinates: {lat:.3f}, {lon:.3f}")

    start_year, end_year = st.slider(
        "Select seasonal year range (July–June)",
        min_value=2021, max_value=2023,
        value=(2021, 2022)
    )

    years = range(start_year, end_year + 1)
    results = []
    for y in years:
        start_date = pd.Timestamp(year=y, month=7, day=1)
        end_date = pd.Timestamp(year=y+1, month=6, day=30, hour=23, minute=59, second=59)
        try:
            drift = calculate_snow_drift(lat, lon, start_date, end_date)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        results.append({"year": f"{y}-{y+1}", "snow_drift_kgm": drift})

    df_drift = pd.DataFrame(results)
    df_drift["snow_drift_tonnesm"] = df_drift["snow_drift_kgm"] / 1000.0

    st.write("### Annual snow drift (July–June)")
    st.bar_chart(df_drift.set_index("year")["snow_drift_tonnesm"])

    st.write("### Wind rose")
    try:
        fig = plot_wind_rose(lat, lon, start_year, end_year)
        fig.set_size_inches(4,4)
        st.pyplot(fig)
    except FileNotFoundError as e:
        st.error(str(e))
else:
    st.warning("No coordinates selected on the map above. Please click a location.")
