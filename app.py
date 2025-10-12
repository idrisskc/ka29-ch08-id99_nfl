# =======================================================
# app.py
# 🏈 NFL Big Data Bowl 2026 - Analytics Dashboard
# ======================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime
from utils import compute_all_kpis_and_aggregate, load_data_from_kaggle_api

# ---------------------
# 🎨 NFL Color Palette
# ---------------------
COLOR_BG = "#0B0C10"
COLOR_PANEL = "#1B263B"
COLOR_ACCENT = "#C1121F"
COLOR_GOLD = "#FFD700"
COLOR_SILVER = "#A9A9A9"
TEXT_COLOR = "#E6EEF8"

# ---------------------
# ⚙️ Page Configuration
# ---------------------
st.set_page_config(page_title="NFL Big Data Bowl 2026", layout="wide", page_icon="🏈")

st.markdown(f"""
<style>
.stApp {{ background-color: {COLOR_BG}; color: {TEXT_COLOR}; }}
[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg, {COLOR_PANEL}, #0b0c10);
    border-right: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 1rem;
}}
.kpi-card {{
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; padding: 16px; margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
}}
.kpi-title {{ color: {COLOR_SILVER}; font-size:13px; margin-bottom:6px; }}
.kpi-value {{ font-size:28px; font-weight:700; color:{COLOR_GOLD}; }}
.kpi-sub {{ color: rgba(255,255,255,0.6); font-size:12px; }}
</style>
""", unsafe_allow_html=True)

st.title("🏈 NFL Big Data Bowl 2026 - Analytics Dashboard")
st.markdown("_Explore, visualize, and understand NFL tracking data from the 2026 Big Data Bowl._")

# ---------------------
# Sidebar Filters
# ---------------------
st.sidebar.header("⚙️ Dashboard Controls")

today = datetime.today().date()
start_default = today.replace(year=today.year - 1)
start_date = st.sidebar.date_input("Start date", start_default)
end_date = st.sidebar.date_input("End date", today)
timeframe = st.sidebar.selectbox("Select time frame", ["Daily", "Weekly", "Monthly"])
chart_type = st.sidebar.selectbox("Select a chart type", ["Bar", "Line", "Area"])

# ---------------------
# Data Source Selection
# ---------------------
source = st.sidebar.radio("Data Source:", ["Kaggle API", "Local ./data/"])
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

full_df = pd.DataFrame()
if source == "Kaggle API":
    kaggle_user = st.sidebar.text_input("Kaggle Username", value=st.secrets.get("KAGGLE_USERNAME", ""))
    kaggle_key = st.sidebar.text_input("Kaggle Key", type="password", value=st.secrets.get("KAGGLE_KEY", ""))
    if st.sidebar.button("🚀 Load Data from Kaggle"):
        with st.spinner("Loading data from Kaggle API..."):
            full_df = load_data_from_kaggle_api(username=kaggle_user, key=kaggle_key)
elif source == "Local ./data/":
    if st.sidebar.button("🚀 Load Data from Local"):
        from utils import load_data  # fallback to local loader
        with st.spinner("Loading local CSV data..."):
            full_df = load_data(use_kaggle=False, base_dir=data_dir)

if not full_df.empty:
    st.success(f"✅ {len(full_df):,} rows loaded successfully!")

    # ---------------------
    # KPIs & Charts
    # ---------------------
    kpis, df_pass_kpis = compute_all_kpis_and_aggregate(full_df)

    st.markdown("## 📊 Key Performance Indicators")
    cols = st.columns(4)
    for i, (k, v) in enumerate(kpis.items()):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{k}</div>
                <div class="kpi-value">{v:.2f}</div>
                <div class="kpi-sub">Aggregate Value</div>
            </div>
            """, unsafe_allow_html=True)

    # KPI Chart
    df_kpi = pd.DataFrame(list(kpis.items()), columns=["KPI", "Value"])
    if chart_type == "Bar":
        fig = px.bar(df_kpi, x="KPI", y="Value", color="KPI", template="plotly_dark", text="Value")
    elif chart_type == "Line":
        fig = px.line(df_kpi, x="KPI", y="Value", markers=True, template="plotly_dark")
    else:  # Area
        fig = px.area(df_kpi, x="KPI", y="Value", template="plotly_dark")

    fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, font_color=TEXT_COLOR)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("⬅️ Use the sidebar to load your data from Kaggle API or local CSV files.")
