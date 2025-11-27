import streamlit as st

st.title("⚡ Energy & Weather Dashboard")
st.markdown("""
Welcome! This dashboard provides interactive analysis of energy production and weather data.  
Navigate through the pages using the sidebar:

1. **Analysis of Elhub data**  
   Explore hourly electricity production by price area and production group:  
   - Pie chart: Total production per price area.  
   - Line chart: Hourly production trends by group and month.  

2. **Beautiful STL and Spectrogram**  
   Analyze time series patterns in energy production:  
   - STL Decomposition: Trend, seasonal, and residual components.  
   - Spectrogram: Frequency content over time.  

3. **Data Visualization Dashboard**  
   Flexible plotting of ERA5 weather data:  
   - Select variables and months.  
   - Plot multiple variables or a single variable over time.  

4. **Extreme Event Analysis**  
   Detect unusual weather events for selected cities and years:  
   - Temperature: Outliers via DCT + SPC.  
   - Precipitation: Anomalies via Local Outlier Factor (LOF).  

5. **Interactive Electricity Production & Consumption Map (Elhub NO1–NO5)**  
   Explore average quantities per price area:  
   - Select data type, group, and year.  
   - Inspect area-specific values by clicking on map regions.  

6. **Interactive SARIMAX Forecasting**  
   Forecast electricity production or consumption:  
   - Select type, group, and price area.  
   - Choose training data period for model fitting.  
   - Configure SARIMAX parameters (AR, MA, seasonal terms).  
   - Generate forecasts with confidence intervals.  

7. **Energy Data & Snow Drift Explorer**  
   Study energy and snow drift impacts interactively:  
   - Energy Map: Visualize production or consumption per NO price area over selected days.  
   - Click map locations to inspect area-level values.  
   - Snow Drift Explorer: Calculate and visualize seasonal and monthly snow drift at chosen coordinates.  
   - Generate wind rose plots for wind-driven snow distribution analysis.  

8. **Sliding Window Correlation: Energy vs Weather (Daily)**  
   Analyze relationships between energy and weather:  
   - Compare production or consumption with ERA5 meteorological variables.  
   - Apply lag to see leading/lagging effects.  
   - Use movable sliding windows to explore correlations over time.  
   - Visualize energy, weather, and correlation series interactively.
""")

# Added a nice picture (Chose a nice night sky)
st.image(
    "https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1200&q=80",
    use_container_width=True
)

st.markdown("---")
st.info("Use the sidebar to navigate between different pages of the dashboard :)")

