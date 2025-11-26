import streamlit as st
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import plotly.graph_objects as go

# -------------------------
# Use your Data_loader
# -------------------------
from Data_loader import load_production, load_consumption

st.set_page_config(layout="wide")
st.title("Energy Production/Consumption Forecast")

# -------------------------
# Series selection
# -------------------------
def select_series(df, kind):
    if df.empty:
        return pd.Series(dtype=float), {}

    key_col = "productiongroup" if kind == "production" else "consumptiongroup"
    groups = sorted(df[key_col].dropna().unique())
    group = st.selectbox(f"Select {key_col}", options=groups)

    priceareas = sorted(df["pricearea"].dropna().unique())
    pa = st.selectbox("Select price area", options=np.append(["ALL"], priceareas))

    if pa == "ALL":
        filtered = df[df[key_col] == group].copy()
    else:
        filtered = df[(df[key_col] == group) & (df["pricearea"] == pa)].copy()

    if filtered.empty:
        return pd.Series(dtype=float), {"pricearea": pa, "group": group}

    series = (
        filtered.groupby(filtered.index)
        .agg({"quantitykwh": "sum"})
        .sort_index()["quantitykwh"]
    )

    # Daily resampling
    series = series.resample("D").sum().fillna(0)

    try:
        inferred = series.index.inferred_freq
        if inferred:
            series = series.asfreq(inferred)
    except Exception:
        pass

    return series, {"pricearea": pa, "group": group}

# -------------------------
# SARIMAX fitting (cached)
# -------------------------
@st.cache_resource(show_spinner=True)
def fit_sarimax(series, order, seasonal_order, exog=None):
    model = sm.tsa.SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

# -------------------------
# Load data using Data_loader
# -------------------------
prod_df = load_production()
cons_df = load_consumption()

# -------------------------
# Sidebar selections
# -------------------------
kind = st.sidebar.radio("Select type", ["production", "consumption"])
series, meta = select_series(prod_df if kind=="production" else cons_df, kind)

if series.empty:
    st.warning("No data available for selected options.")
    st.stop()

# -------------------------
# Training data period selection
# -------------------------
st.header("Training Data Period Selection")
min_date = series.index.min().date()
max_date = series.index.max().date()
start_date, end_date = st.date_input(
    "Select training data period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Convert date_input to tz-aware datetime to match series index
start_dt = pd.Timestamp(datetime.datetime.combine(start_date, datetime.time(0, 0)), tz="UTC")
end_dt = pd.Timestamp(datetime.datetime.combine(end_date, datetime.time(23, 59, 59)), tz="UTC")
series = series.loc[start_dt:end_dt]

# -------------------------
# Show series using Plotly
# -------------------------
st.subheader(f"Selected series: {meta}")
fig_series = go.Figure()
fig_series.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Observed"))
fig_series.update_layout(title="Observed Series", xaxis_title="Date", yaxis_title="Quantity kWh")
st.plotly_chart(fig_series, use_container_width=True)

# -------------------------
# Forecast parameters
# -------------------------
st.sidebar.subheader("SARIMAX parameters")
p = st.sidebar.number_input("AR term (p)", 0, 5, 1)
d = st.sidebar.number_input("Differencing term (d)", 0, 2, 1)
q = st.sidebar.number_input("MA term (q)", 0, 5, 1)

P = st.sidebar.number_input("Seasonal AR term (P)", 0, 2, 0)
D = st.sidebar.number_input("Seasonal differencing (D)", 0, 1, 0)
Q = st.sidebar.number_input("Seasonal MA term (Q)", 0, 2, 0)
m = st.sidebar.number_input("Seasonal period (m)", 1, 168, 24)

steps = st.sidebar.number_input("Forecast horizon (steps)", 1, 168, 24)
forecast_button = st.sidebar.button("Run Forecast")

# -------------------------
# Run Forecast
# -------------------------
if forecast_button:
    with st.spinner("Fitting SARIMAX model..."):
        model_fit = fit_sarimax(series, order=(p, d, q), seasonal_order=(P, D, Q, m))
        forecast_res = model_fit.get_forecast(steps=steps)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

    # Plot forecast with Plotly
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Observed"))
    fig_forecast.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean.values, mode="lines", name="Forecast", line=dict(color="red")))
    fig_forecast.add_trace(go.Scatter(
        x=conf_int.index.tolist() + conf_int.index[::-1].tolist(),
        y=conf_int.iloc[:,0].tolist() + conf_int.iloc[:,1][::-1].tolist(),
        fill="toself", fillcolor="rgba(255,182,193,0.3)",
        line=dict(color="rgba(255,255,255,0)"), showlegend=True, name="Confidence Interval"
    ))
    fig_forecast.update_layout(title="SARIMAX Forecast", xaxis_title="Date", yaxis_title="Quantity kWh")
    st.plotly_chart(fig_forecast, use_container_width=True)

