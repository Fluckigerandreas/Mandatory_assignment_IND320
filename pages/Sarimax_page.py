import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import certifi
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import io

# -----------------------------
# Helper: provided data loaders
# -----------------------------
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
        # cheap normalizer: lowercase and strip
        df["pricearea"] = df["pricearea"].str.strip().str.upper()

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
        df["pricearea"] = df["pricearea"].str.strip().str.upper()
    df = df.groupby(["pricearea", "consumptiongroup", "starttime"], as_index=False).agg({"quantitykwh": "sum"})
    df.set_index("starttime", inplace=True)
    return df

# -----------------------------
# Utilities
# -----------------------------

def select_series(df, kind):
    # kind = 'production' or 'consumption'
    if df.empty:
        return None, None
    if kind == "production":
        groups = df["productiongroup"].unique()
        key_col = "productiongroup"
    else:
        groups = df["consumptiongroup"].unique()
        key_col = "consumptiongroup"

    priceareas = df["pricearea"].unique()
    pa = st.selectbox("Select price area", options=np.append(["ALL"], sorted(priceareas)))
    group = st.selectbox(f"Select {key_col}", options=sorted(groups))

    if pa == "ALL":
        series = df[df[key_col] == group].groupby(df.index).agg({"quantitykwh": "sum"}).sort_index()
    else:
        series = df[(df[key_col] == group) & (df["pricearea"] == pa)][["quantitykwh"]].sort_index()

    series = series["quantitykwh"].asfreq(series.index.inferred_freq or None)
    return series, (pa, group)


def build_exog_matrix(df, selected_exogs, freq, start, end):
    # selected_exogs: list of tuples (pricearea, group, kind)
    # This returns exog aligned to freq index between start and end.
    if not selected_exogs:
        return None
    frames = []
    for kind, pa, group in selected_exogs:
        if kind == "production":
            base = prod_df
            key = "productiongroup"
        else:
            base = cons_df
            key = "consumptiongroup"
        sub = base[(base[key] == group) & ((base["pricearea"] == pa) | (pa == "ALL"))]
        s = sub.groupby(sub.index).agg({"quantitykwh": "sum"})["quantitykwh"].rename(f"{kind}_{pa}_{group}")
        s = s.asfreq(freq)
        frames.append(s)
    if not frames:
        return None
    exog = pd.concat(frames, axis=1).loc[start:end]
    return exog


def make_future_exog(train_exog, horizon, method="repeat_last"):
    # Simple: repeat last observed value for forecast horizon
    if train_exog is None:
        return None
    last = train_exog.iloc[-1:]
    rep = pd.concat([last]*horizon, ignore_index=True)
    rep.index = pd.date_range(start=train_exog.index[-1] + (train_exog.index[1]-train_exog.index[0]) if len(train_exog.index)>1 else pd.Timedelta("1H"), periods=horizon, freq=train_exog.index.inferred_freq or None)
    rep.columns = train_exog.columns
    return rep

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide", page_title="SARIMAX energy forecasting")
st.title("SARIMAX — Energy production & consumption forecasting")

prod_df = load_production()
cons_df = load_consumption()

left, right = st.columns([2,1])
with left:
    data_choice = st.radio("Dataset", options=["Production", "Consumption"]) 
    kind = "production" if data_choice == "Production" else "consumption"

    series, meta = select_series(prod_df if kind=="production" else cons_df, kind)
    if series is None or series.empty:
        st.warning("No data available for that selection — check price area / group or the database connection.")
        st.stop()

    # Resampling frequency
    freq = st.selectbox("Resample frequency (if uncertain choose H or D)", options=[None, 'H','D','W','M'], index=1)
    if freq:
        series = series.resample(freq).sum()

    st.markdown("**Time range for training**")
    min_date = series.index.min().date()
    max_date = series.index.max().date()
    start_date = st.date_input("Train start date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("Train end date", value=max_date, min_value=min_date, max_value=max_date)
    if start_date >= end_date:
        st.error("Start date must be before end date")
        st.stop()

    train_start = pd.to_datetime(start_date).tz_localize(series.index.tz) if series.index.tz else pd.to_datetime(start_date)
    train_end = pd.to_datetime(end_date).tz_localize(series.index.tz) if series.index.tz else pd.to_datetime(end_date)

    train_series = series.loc[train_start:train_end]

    horizon = st.number_input("Forecast horizon (periods)", min_value=1, max_value=8760, value=48)

    # Exogenous variables selection
    st.markdown("**Exogenous variables**")
    exog_candidates = []
    if not prod_df.empty:
        for pg in sorted(prod_df['productiongroup'].unique()[:50]):
            exog_candidates.append(("production", "ALL", pg))
    if not cons_df.empty:
        for cg in sorted(cons_df['consumptiongroup'].unique()[:50]):
            exog_candidates.append(("consumption", "ALL", cg))

    # Present a multiselect with readable labels
    exog_map = {f"{k}_{pa}_{g}": (k, pa, g) for (k, pa, g) in exog_candidates}
    exog_labels = list(exog_map.keys())
    selected_labels = st.multiselect("Select exogenous series (will repeat last observed value for forecast) — choose none for univariate", options=exog_labels)
    selected_exogs = [exog_map[l] for l in selected_labels]

    # SARIMAX params
    st.markdown("**SARIMAX parameters**")
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.number_input("p (AR order)", min_value=0, max_value=5, value=1)
        d = st.number_input("d (difference order)", min_value=0, max_value=2, value=0)
        q = st.number_input("q (MA order)", min_value=0, max_value=5, value=1)
    with col2:
        P = st.number_input("P (seasonal AR)", min_value=0, max_value=3, value=0)
        D = st.number_input("D (seasonal diff)", min_value=0, max_value=2, value=0)
        Q = st.number_input("Q (seasonal MA)", min_value=0, max_value=3, value=0)
    with col3:
        s = st.number_input("s (seasonal period - e.g. 24 for daily hourly, 168 for weekly hourly)", min_value=0, max_value=8760, value=24)
        trend = st.selectbox("trend", options=[None,'n','c','t','ct'], index=1)

    dynamic = st.checkbox("Use dynamic forecasting (start dynamic=True after n steps)", value=False)
    dynamic_start = None
    if dynamic:
        dynamic_start = st.number_input("Dynamic start (use integer index within train, or 0 to start forecasting from first forecast step)", min_value=0, max_value=len(train_series)-1, value=0)

    fit_button = st.button("Fit SARIMAX & Forecast")

with right:
    st.markdown("## Model & Output")
    info = st.empty()

# -----------------------------
# Modeling
# -----------------------------
if fit_button:
    with st.spinner("Fitting model — this may take a while depending on data length and orders..."):
        # Prepare exog for training
        exog_train = build_exog_matrix(pd.concat([prod_df, cons_df], keys=['production','consumption'], names=['kind']), selected_exogs, freq or series.index.inferred_freq, train_start, train_end) if selected_exogs else None

        # If exog present, build future exog as repeated last
        if exog_train is not None and not exog_train.empty:
            exog_future = make_future_exog(exog_train, horizon)
            # align indices
            future_index = exog_future.index
        else:
            exog_future = None
            future_index = pd.date_range(start=train_series.index[-1] + (train_series.index[1]-train_series.index[0]) if len(train_series.index)>1 else pd.Timedelta("1H"), periods=horizon, freq=train_series.index.inferred_freq or None)

        # Fit SARIMAX
        warnings.simplefilter('ignore', ConvergenceWarning)
        try:
            model = SARIMAX(train_series, order=(p,d,q), seasonal_order=(P,D,Q,int(s) if s>0 else 0), trend=trend, exog=exog_train)
            res = model.fit(disp=False)
        except Exception as e:
            st.error(f"Model fitting failed: {e}")
            st.stop()

        info.write(f"Model fitted. AIC: {res.aic:.2f}, BIC: {res.bic:.2f}")

        # Forecast
        try:
            # get prediction with conf_int
            if exog_future is not None:
                pred = res.get_forecast(steps=horizon, exog=exog_future)
            else:
                pred = res.get_forecast(steps=horizon)

            mean_forecast = pred.predicted_mean
            conf_int = pred.conf_int(alpha=0.05)

            # Create dataframe for display and download
            fc_df = pd.DataFrame({
                'forecast': mean_forecast,
                'lower_ci': conf_int.iloc[:,0],
                'upper_ci': conf_int.iloc[:,1]
            }, index=future_index)

            # Combine with history
            history = train_series

            # Plot
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(history.index, history.values, label='history')
            ax.plot(fc_df.index, fc_df['forecast'], label='forecast')
            ax.fill_between(fc_df.index, fc_df['lower_ci'], fc_df['upper_ci'], alpha=0.3, label='95% CI')
            ax.set_title(f"SARIMAX forecast for {meta}")
            ax.legend()
            st.pyplot(fig)

            st.markdown("### Forecast head")
            st.dataframe(fc_df.head(20))

            # Download
            buf = io.BytesIO()
            fc_df.to_csv(buf)
            buf.seek(0)
            st.download_button("Download forecast CSV", data=buf, file_name="sarimax_forecast.csv", mime='text/csv')

        except Exception as e:
            st.error(f"Forecasting failed: {e}")
            st.stop()

    
    st.success("Done")

else:
    st.info("Choose parameters and press 'Fit SARIMAX & Forecast' to run the model.")

# -----------------------------
# End
# -----------------------------
