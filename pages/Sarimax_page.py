import streamlit as st
import pandas as pd
import numpy as np
import certifi
from pymongo import MongoClient
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
    # Daily resampling
    # -------------------------
    series = series.resample("D").sum().fillna(0)
