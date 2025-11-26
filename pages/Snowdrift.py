import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from functions.Snow_drift import calculate_snow_drift, plot_wind_rose

st.title("❄️ Snow Drift Explorer")

# Debug view (optional; comment out if noisy)
st.write("Clicked point:", st.session_state.get("clicked_point"))

# Guard: require coordinates selected on the map page
if "clicked_point" not in st.session_state or st.session_state.clicked_point is None:
    st.warning("No coordinates selected on the map page. Please go back and click a location.")
    st.stop()

lat, lon = st.session_state.clicked_point
st.write(f"Using coordinates: {lat:.3f}, {lon:.3f}")

# Year range selector (seasonal years)
start_year, end_year = st.slider(
    "Select seasonal year range (July–June)",
    min_value=2000, max_value=2025,
    value=(2015, 2020)
)

# Compute seasonal drifts for each year in range
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

df = pd.DataFrame(results)
df["snow_drift_tonnesm"] = df["snow_drift_kgm"] / 1000.0

# Plot snow drift per year
st.write("### Annual snow drift (July–June)")
st.bar_chart(df.set_index("year")["snow_drift_tonnesm"])

# Plot wind rose
st.write("### Wind rose")
try:
    fig = plot_wind_rose(lat, lon, start_year, end_year)
    st.pyplot(fig)
except FileNotFoundError as e:
    st.error(str(e))
