# ======================================================
# NewA.py — Streamlit page
# ======================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy import signal
from pymongo.mongo_client import MongoClient
import certifi

# ======================================================
# 1) Load data from MongoDB (cached)
# ======================================================
@st.cache_data(show_spinner="Loading data from MongoDB...")
def load_data():
    """Load data from MongoDB with caching, aggregate duplicates, produce time series."""
    uri = st.secrets["mongo"]["uri"]
    ca = certifi.where()
    client = MongoClient(uri, tls=True, tlsCAFile=ca)
    db = client['Elhub']
    collection = db['Data']

    data = list(collection.find())
    if not data:
        return pd.DataFrame()  # Empty DataFrame fallback

    df = pd.DataFrame(data)
    df["starttime"] = pd.to_datetime(df["starttime"])

    # Aggregate duplicates by summing quantities
    df = df.groupby(["pricearea", "productiongroup", "starttime"], as_index=False).agg({"quantitykwh": "sum"})

    # Set datetime index for time series
    df.set_index("starttime", inplace=True)

    return df

# ======================================================
# 2) STL decomposition
# ======================================================
def stl_decompose_series(series, period=24*7, title="STL Decomposition"):
    """Perform STL decomposition on a time series."""
    # Ensure datetime index
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.sort_index()

    # Timezone handling (correct)
    if series.index.tz is None:
        series = series.tz_localize("UTC")
    else:
        series = series.tz_convert("UTC")

    # Regularize to hourly frequency
    series = series.asfreq("h")
    series = series.interpolate(method="time")

    # STL
    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    # Plot
    fig = result.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle(f"{title}\n{series.name}", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    return result

# ======================================================
# 3) Spectrogram
# ======================================================
def plot_spectrogram(series, fs=1.0, nperseg=24*7, noverlap=None):
    """Plot the spectrogram of a time series."""
    s = series.dropna().astype(float)
    noverlap = noverlap or nperseg // 2
    f, t, Sxx = signal.spectrogram(s.values, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    ax.set_title("Spectrogram (dB scale)")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Frequency [cycles/hour]")
    plt.colorbar(pcm, ax=ax, label="dB")
    plt.tight_layout()
    st.pyplot(fig)

    return f, t, Sxx

# ======================================================
# 4) Streamlit UI
# ======================================================
st.title("NewA Analysis: STL & Spectrogram")

# Load data
df = load_data()
if df.empty:
    st.warning("No data found in MongoDB.")
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
