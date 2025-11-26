import streamlit as st
from pymongo import MongoClient
import certifi
import pandas as pd

# -------------------------
# Shared MongoDB Client (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def get_mongo_client():
    return MongoClient(st.secrets["mongo"]["uri"], tls=True, tlsCAFile=certifi.where())


# -------------------------
# Production Data Loader
# -------------------------
@st.cache_data(show_spinner="Loading production data...")
def load_production():
    client = get_mongo_client()
    db = client["Elhub"]
    df = pd.DataFrame(list(db["Data"].find()))

    if df.empty:
        return df

    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce", utc=True)
    df = df.dropna(subset=["starttime"])

    if "pricearea" in df.columns:
        df["pricearea"] = df["pricearea"].apply(lambda x: x if x else "NO")

    df = df.groupby(["pricearea", "productiongroup", "starttime"], as_index=False).agg(
        {"quantitykwh": "sum"}
    )
    df.set_index("starttime", inplace=True)

    return df


# -------------------------
# Consumption Data Loader
# -------------------------
@st.cache_data(show_spinner="Loading consumption data...")
def load_consumption():
    client = get_mongo_client()
    db = client["Consumption_Elhub"]
    df = pd.DataFrame(list(db["Data"].find()))

    if df.empty:
        return df

    df["starttime"] = pd.to_datetime(df["starttime"], utc=True)

    if "pricearea" in df.columns:
        df["pricearea"] = df["pricearea"].apply(lambda x: x if x else "NO")

    df = df.groupby(["pricearea", "consumptiongroup", "starttime"], as_index=False).agg(
        {"quantitykwh": "sum"}
    )
    df.set_index("starttime", inplace=True)

    return df
