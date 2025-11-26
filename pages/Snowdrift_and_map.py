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
import matplotlib.pyplot as plt

# Your domain functions (must exist and be importable)
from functions.snow_drift import calculate_snow_drift, plot_wind_rose
from Data_loader import load_production, load_consumption

st.set_page_config(layout="wide")
st.title("Energy Map & Snow Drift Explorer")

# ==============================================================================
# Normalize function for GeoJSON and dataframe keys
# ==============================================================================
def normalize_area_name(name):
    """Normalize area names: remove whitespace, uppercase, fix common typos like 'N0' -> 'NO'."""
    if not isinstance(name, str):
        return name
    n = name.strip().upper().replace(" ", "")
    # fix a common typo where zero is used instead of letter O: N0 -> NO
    if n.startswith("N0") and len(n) >= 3:
        n = "NO" + n[2:]
    return n

# ==============================================================================
# Load GeoJSON
# ==============================================================================
geojson_path = "/workspaces/blank-app/file.geojson"
if not geojson_path.exists():
    st.error(f"GeoJSON file not found at {geojson_path}")
    st.stop()

with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# Add normalized property used for keying the choropleth
for feature in geojson_data.get("features", []):
    raw_name = feature.get("properties", {}).get("ElSpotOmr") or feature.get("properties", {}).get("ElSpotOmrNorm")
    feature["properties"]["ElSpotOmrNorm"] = normalize_area_name(raw_name)

# ==============================================================================
# Session state init
# ==============================================================================
if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = None
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None

# ==============================================================================
# UI – choose Production / Consumption and time window
# ==============================================================================
data_type = st.radio("Select data type:", ["Production", "Consumption"], horizontal=True)

days = st.slider("Days to include (most recent)", min_value=1, max_value=365, value=30, help="Number of days back from latest timestamp to include in the averages")

# ==============================================================================
# Load data using Data_loader
# ==============================================================================
try:
    prod_df = load_production()
except Exception as e:
    st.error(f"Failed to load production data: {e}")
    prod_df = pd.DataFrame()

try:
    cons_df = load_consumption()
except Exception as e:
    st.error(f"Failed to load consumption data: {e}")
    cons_df = pd.DataFrame()

# Choose active dataframe
df = prod_df if data_type == "Production" else cons_df

# Validate required columns
required_cols = {"starttime", "pricearea", "quantitykwh"}
if df.empty:
    st.error(f"No data loaded for {data_type}. Check Data_loader functions.")
    st.stop()
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Dataframe is missing required columns: {missing}")
    st.write("Columns present:", list(df.columns))
    st.stop()

# ==============================================================================
# Prepare dataframe
# ==============================================================================
df = df.copy()
df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
if df["starttime"].isna().all():
    st.error("All 'starttime' values could not be parsed as datetimes.")
    st.stop()

df["pricearea"] = df["pricearea"].apply(normalize_area_name)

end_time = df["starttime"].max()
start_time = end_time - timedelta(days=int(days))

df_period = df[(df["starttime"] >= start_time) & (df["starttime"] <= end_time)]

if df_period.empty:
    st.warning("No data available for the selected time window.")
    st.stop()

# Compute mean quantitykwh per pricearea
means_df = df_period.groupby("pricearea", as_index=False)["quantitykwh"].mean().rename(columns={"quantitykwh": "quantitykwh_mean"})

if means_df.empty:
    st.warning("No aggregated data available for this selection.")
    st.stop()

means_dict = dict(zip(means_df["pricearea"], means_df["quantitykwh_mean"]))

# ==============================================================================
# Map
# ==============================================================================
m = folium.Map(location=[63.0, 10.5], zoom_start=5.5)

vmin = means_df["quantitykwh_mean"].min()
vmax = means_df["quantitykwh_mean"].max()
# Create thresholds for Choropleth (ensure at least two different values)
if np.isclose(vmin, vmax):
    thresholds = [vmin - 1e-6, vmin, vmax + 1e-6]
else:
    thresholds = np.linspace(vmin, vmax, 6).tolist()

folium.Choropleth(
    geo_data=geojson_data,
    name="choropleth",
    data=means_df,
    columns=["pricearea", "quantitykwh_mean"],
    key_on="feature.properties.ElSpotOmrNorm",
    fill_color="YlGnBu",
    fill_opacity=0.6,
    line_opacity=0.3,
    line_color="black",
    legend_name=f"{data_type} mean quantity (kWh)",
    threshold_scale=thresholds,
    nan_fill_color="lightgray"
).add_to(m)

# Add tooltip GeoJson (transparent fill so underlying choropleth shows)
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

# Highlight selected point and area if present
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

# Render map and capture click events
map_data = st_folium(m, width=950, height=630)

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.session_state.clicked_point = (lat, lon)

    # Determine which pricearea contains the clicked point
    point = Point(lon, lat)  # shapely uses (x=lon, y=lat)
    clicked_area = None
    for feature in geojson_data.get("features", []):
        geom = shape(feature["geometry"])
        if isinstance(geom, (Polygon, MultiPolygon)) and geom.contains(point):
            clicked_area = feature["properties"].get("ElSpotOmrNorm")
            break
    st.session_state.selected_area = clicked_area

# ==============================================================================
# Display values
# ==============================================================================
st.write("### Mean quantity (kWh) per NO area:")
st.dataframe(means_df.rename(columns={"quantitykwh_mean": "mean_kWh"}))

if st.session_state.selected_area:
    val = means_dict.get(st.session_state.selected_area, None)
    if val is not None and not pd.isna(val):
        st.success(f"Selected area: **{st.session_state.selected_area}** → {val:.2f} kWh (mean over last {days} days)")
    else:
        st.success(f"Selected area: **{st.session_state.selected_area}** (no data for the chosen time window)")

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
        min_value=2000, max_value=2025,
        value=(2015, 2020)
    )

    years = range(start_year, end_year + 1)
    results = []
    for y in years:
        start_date = pd.Timestamp(year=y, month=7, day=1)
        end_date = pd.Timestamp(year=y + 1, month=6, day=30, hour=23, minute=59, second=59)
        try:
            drift = calculate_snow_drift(lat, lon, start_date, end_date)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Error calculating snow drift for season {y}-{y+1}: {e}")
            drift = np.nan
        results.append({"year": f"{y}-{y+1}", "snow_drift_kgm": drift})

    df_drift = pd.DataFrame(results)
    if not df_drift.empty:
        df_drift["snow_drift_tonnesm"] = df_drift["snow_drift_kgm"] / 1000.0
        st.write("### Annual snow drift (July–June)")
        # Use altair or st.bar_chart; simple bar_chart is ok
        st.bar_chart(df_drift.set_index("year")["snow_drift_tonnesm"])
    else:
        st.info("No snow drift results to display.")

    st.write("### Wind rose")
    try:
        fig = plot_wind_rose(lat, lon, start_year, end_year)
        # If the plot_wind_rose returns a Matplotlib figure, show it
        if hasattr(fig, "set_size_inches"):
            fig.set_size_inches(4, 4)
            st.pyplot(fig)
        else:
            # handle other return types (e.g., plotly) — try to display directly
            st.write(fig)
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Failed to create wind rose: {e}")
else:
    st.warning("No coordinates selected on the map above. Please click a location.")
