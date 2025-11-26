import streamlit as st
import pandas as pd
import plotly.express as px

# ✔ Import the shared cached loader
from data_loader import load_production

# -------------------------------
# LOAD DATA (cached globally)
# -------------------------------
df = load_production()

if df.empty:
    st.error("No production data found in MongoDB.")
    st.stop()

st.caption(f"✅ Loaded {len(df)} production records (cached across all pages).")


# -------------------------------
# DEFINE COLORS
# -------------------------------
group_colors = {
    "hydro": "blue",
    "wind": "lightblue",
    "solar": "yellow",
    "thermal": "green",
    "other": "black"
}

# Add fallback colors for unexpected groups
for group in df["productiongroup"].unique():
    if group not in group_colors:
        group_colors[group] = px.colors.qualitative.Pastel1[
            len(group_colors) % len(px.colors.qualitative.Pastel1)
        ]


# -------------------------------
# STREAMLIT LAYOUT
# -------------------------------
st.title("⚡ Energy Production Dashboard")
col1, col2 = st.columns(2)

# -------------------------------
# LEFT COLUMN: Price area + Pie Chart
# -------------------------------
with col1:
    st.header("Total Production per Price Area")
    st.subheader("Select Price Areas:")

    selected_areas = []

    # Arrange checkboxes horizontally
    price_areas = df["pricearea"].unique()
    n_cols = min(4, len(price_areas))
    rows = (len(price_areas) + n_cols - 1) // n_cols
    for r in range(rows):
        cols = st.columns(n_cols)
        for c, area_idx in enumerate(range(r * n_cols, min((r + 1) * n_cols, len(price_areas)))):
            area = price_areas[area_idx]
            if cols[c].checkbox(area, value=True, key=f"chk_{area}"):
                selected_areas.append(area)

    if not selected_areas:
        st.warning("Please select at least one price area.")
        st.stop()

    df_area = df[df["pricearea"].isin(selected_areas)]
    total_by_group = df_area.groupby("productiongroup")["quantitykwh"].sum().reset_index()

    # Pie chart
    fig_pie = px.pie(
        total_by_group,
        names="productiongroup",
        values="quantitykwh",
        color="productiongroup",
        color_discrete_map=group_colors,
        title="Total Production in Selected Price Area(s)",
        width=600,
        height=600
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)


# -------------------------------
# RIGHT COLUMN: Line Chart by Month and Group
# -------------------------------
with col2:
    st.header("Monthly Production Line Plot")

    # Production group selection (pills style)
    prod_groups_selected = st.multiselect(
        "Select production group(s):",
        df["productiongroup"].unique(),
        default=df["productiongroup"].unique()
    )

    # Month selection
    month = st.selectbox(
        "Select a month:",
        list(range(1, 13)),
        format_func=lambda x: pd.to_datetime(f"2021-{x}-01").strftime("%B")
    )

    # Filter and aggregate data
    df_filtered = df_area[
        (df_area["productiongroup"].isin(prod_groups_selected)) &
        (df_area["starttime"].dt.month == month)
    ]

    if df_filtered.empty:
        st.warning("No data for this selection.")
    else:
        # --- SUM across price areas ---
        df_sum = (
            df_filtered
            .groupby(["starttime", "productiongroup"], as_index=False)["quantitykwh"].sum()
            .sort_values("starttime")
        )

        # --- Create the line chart ---
        fig_line = px.line(
            df_sum,
            x="starttime",
            y="quantitykwh",
            color="productiongroup",
            markers=True,
            color_discrete_map=group_colors,
            title=f"Total Hourly Production ({pd.to_datetime(f'2021-{month}-01').strftime('%B')})",
            width=900,
            height=500
        )
        fig_line.update_traces(connectgaps=False)
        st.plotly_chart(fig_line, use_container_width=True)


# -------------------------------
# Data Source Info
# -------------------------------
with st.expander("ℹ️ Data Source"):
    st.write("""
    The data in this dashboard comes from the ELHUB API, showing hourly electricity
    production by price area and production group. It’s stored in MongoDB and visualized here interactively.
    """)

