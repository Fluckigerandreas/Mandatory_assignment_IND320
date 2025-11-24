import streamlit as st
import pandas as pd
import numpy as np
import certifi
from pymongo import MongoClient
import statsmodels.api as sm

# -------------------------
# Data loading (cached)
# -------------------------
@st.cache_data(show_spinner="Loading production data...")
def load_production():
    client = MongoClient(st.secrets["mongo"]["uri"], tls=True, tlsCAFile=certifi.where())
    db = client["Elhub"]
    df = pd.DataFrame(list(db["Data"].find()))
    if df.empty:
        return df
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce", utc=True)
    df = df.dropna(subset=["starttime"])
    if "pricearea" in df.columns:
        df["pricearea"] = df["pricearea"].apply(lambda x: x if x else "NO")
    df = df.groupby(["pricearea", "productiongroup", "starttime"], as_index=False).agg({"quantitykwh": "sum"})
    df.set_index("starttime", inplace=True)
    return df

@st.cache_data(show_spinner="Loading consumption data...")
def load_consumption():
    client = MongoClient(st.secrets["mongo"]["uri"], tls=True, tlsCAFile=certifi.where())
    db = client["Consumption_Elhub"]
    df = pd.DataFrame(list(db["Data"].find()))
    if df.empty:
        return df
    df["starttime"] = pd.to_datetime(df["starttime"], utc=True)
    if "pricearea" in df.columns:
        df["pricearea"] = df["pricearea"].apply(lambda x: x if x else "NO")
    df = df.groupby(["pricearea", "consumptiongroup", "starttime"], as_index=False).agg({"quantitykwh": "sum"})
    df.set_index("starttime", inplace=True)
    return df

prod_df = load_production()
cons_df = load_consumption()

# -------------------------
# Series selection (NO CACHE!)
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

    # -------------------------
    # NEW: Daily resampling
    # -------------------------
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
# Streamlit UI
# -------------------------
st.title("Energy Production/Consumption Forecast")

# Sidebar selections
kind = st.sidebar.radio("Select type", ["production", "consumption"])
series, meta = select_series(prod_df if kind=="production" else cons_df, kind)

if series.empty:
    st.warning("No data available for selected options.")
else:
    st.subheader(f"Selected series: {meta}")
    st.line_chart(series)

    # Forecast parameters
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

    if forecast_button:
        with st.spinner("Fitting SARIMAX model..."):
            model_fit = fit_sarimax(series, order=(p, d, q), seasonal_order=(P, D, Q, m))
            forecast_res = model_fit.get_forecast(steps=steps)
            forecast_mean = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()

        st.subheader("Forecast")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series.index, series.values, label="Observed")
        ax.plot(forecast_mean.index, forecast_mean.values, label="Forecast", color="red")
        ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
        ax.legend()
        st.pyplot(fig)
