# ======================================================
# NewA.py — Streamlit page (Plotly + Global Loader)
# ======================================================
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import global data loader
from your_global_loader import load_production  # <-- replace with actual import path

# ======================================================
# 1) STL decomposition
# ======================================================
def stl_decompose_series(series, period=24*7, title="STL Decomposition"):
    """Perform STL decomposition on a time series and plot with Plotly."""
    # Ensure datetime index
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.sort_index()

    # Remove duplicate timestamps
    series = series.groupby(series.index).sum()

    # Regularize to hourly frequency
    series = series.asfreq("h")
    series = series.interpolate(method="time")

    # STL
    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    # Plot with Plotly
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    fig.add_trace(go.Scatter(x=series.index, y=result.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=result.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=result.resid, name="Residual"), row=4, col=1)

    fig.update_layout(height=900, width=900, title_text=f"{title}: {series.name}")
    st.plotly_chart(fig, use_container_width=True)

    return result

# ======================================================
# 2) Spectrogram
# ======================================================
def plot_spectrogram(series, fs=1.0, nperseg=24*7, noverlap=None):
    """Plot the spectrogram of a time series using Plotly."""
    s = series.dropna().astype(float)
    noverlap = noverlap or nperseg // 2
    f, t, Sxx = signal.spectrogram(s.values, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig = go.Figure(data=go.Heatmap(
        z=Sxx_db,
        x=t,
        y=f,
        colorscale="Viridis",
        colorbar=dict(title="dB")
    ))
    fig.update_layout(
        title="Spectrogram (dB scale)",
        xaxis_title="Window index",
        yaxis_title="Frequency [cycles/hour]",
        height=600,
        width=900
    )
    st.plotly_chart(fig, use_container_width=True)

    return f, t, Sxx

# ======================================================
# 3) Streamlit UI
# ======================================================
st.title("STL & Spectrogram")

# Load data using global loader
df = load_production()
if df.empty:
    st.warning("No production data found.")
    st.stop()

# Checkbox: Use all data or filter
use_all = st.checkbox("Use all data (aggregate over price area and production group)", value=False)

if use_all:
    # Aggregate all data
    series = df.groupby(df.index)["quantitykwh"].sum()
else:
    # Select price area & production group
    priceareas = df["pricearea"].unique()
    prod_groups = df["productiongroup"].unique()

    selected_area = st.selectbox("Select price area", priceareas)
    selected_group = st.selectbox("Select production group", prod_groups)

    # Filter data
    df_area = df[(df["pricearea"] == selected_area) & (df["productiongroup"] == selected_group)]
    series = df_area["quantitykwh"]

# Tabs for analysis
tab1, tab2 = st.tabs(["STL Decomposition", "Spectrogram"])

with tab1:
    st.header("STL Decomposition")
    period = st.number_input("STL period (hours)", min_value=1, value=24*7)
    stl_res = stl_decompose_series(series, period=period)

with tab2:
    st.header("Spectrogram")
    nperseg = st.number_input("Window size (nperseg)", min_value=1, value=24*7)
    plot_spectrogram(series, nperseg=nperseg)

