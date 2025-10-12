# =======================================================
# app.py
# 🏈 NFL Big Data Bowl 2026 - Analytics Dashboard
# =======================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from utils import load_data, compute_all_kpis_and_aggregate, download_from_kaggle

# ---------------------
# 🎨 NFL Color Palette
# ---------------------
COLOR_BG = "#0B0C10"        # deep black
COLOR_PANEL = "#1B263B"     # navy
COLOR_ACCENT = "#C1121F"    # red
COLOR_GOLD = "#FFD700"      # gold
COLOR_SILVER = "#A9A9A9"    # silver
TEXT_COLOR = "#E6EEF8"

# ---------------------
# ⚙️ Page Configuration
# ---------------------
st.set_page_config(page_title="NFL Big Data Bowl 2026", layout="wide", page_icon="🏈")

st.markdown(f"""
<style>
.stApp {{
    background-color: {COLOR_BG};
    color: {TEXT_COLOR};
}}
/* Sidebar */
[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg, {COLOR_PANEL}, #0b0c10);
    border-right: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
}}
/* KPI Card */
.kpi-card {{
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
}}
.kpi-title {{ color: {COLOR_SILVER}; font-size:13px; margin-bottom:6px; }}
.kpi-value {{ font-size:28px; font-weight:700; color:{COLOR_GOLD}; }}
.kpi-sub {{ color: rgba(255,255,255,0.6); font-size:12px; }}
.kpi-badge-up {{
    background: rgba(255,215,0,0.1); color:{COLOR_GOLD};
    padding:4px 8px; border-radius:10px; border:1px solid rgba(255,215,0,0.2);
    font-weight:600; font-size:12px;
}}
.kpi-badge-down {{
    background: rgba(193,18,31,0.12); color:{COLOR_ACCENT};
    padding:4px 8px; border-radius:10px; border:1px solid rgba(193,18,31,0.2);
    font-weight:600; font-size:12px;
}}
</style>
""", unsafe_allow_html=True)

st.title("🏈 NFL Big Data Bowl 2026 - Analytics Dashboard")
st.markdown("_Explore, visualize, and understand NFL tracking data from the 2026 Big Data Bowl._")

# ---------------------
# Sidebar Filters
# ---------------------
st.sidebar.header("⚙️ Settings")

today = datetime.today().date()
start_default = today.replace(year=today.year - 1)
start_date = st.sidebar.date_input("Start date", start_default)
end_date = st.sidebar.date_input("End date", today)
timeframe = st.sidebar.selectbox("Select time frame", ["Daily", "Weekly", "Monthly"])
chart_type = st.sidebar.selectbox("Select a chart type", ["Bar", "Line", "Area"])
st.sidebar.markdown("---")

# ---------------------
# Kaggle data management
# ---------------------
st.sidebar.markdown("📦 **Data Source**")
source = st.sidebar.radio("Select source:", ["Kaggle", "Local ./data/"])

if source == "Kaggle":
    kaggle_user = st.sidebar.text_input("Kaggle Username", value=st.secrets["KAGGLE_USERNAME"])
    kaggle_key = st.sidebar.text_input("Kaggle Key", type="password", value=st.secrets["KAGGLE_KEY"])
    if st.sidebar.button("📥 Download from Kaggle"):
        download_from_kaggle(kaggle_user, kaggle_key)
        st.sidebar.success("Download & extraction complete!")

week_range = st.sidebar.slider("Select Week Range", 1, 18, (1, 18))
st.sidebar.markdown("---")
load_btn = st.sidebar.button("🚀 Load & Analyze Data")

# ---------------------
# Load Data
# ---------------------
if not load_btn:
    st.info("Use the sidebar to download or load data, then click **Load & Analyze**.")
    st.stop()

with st.spinner("Loading data and computing KPIs..."):
    full_df, df_input, df_out, df_supp, tr_sample, prethrow = load_data()
    if full_df.empty:
        st.error("❌ No data found. Please ensure the ./data folder contains the extracted Kaggle CSVs.")
        st.stop()

# Apply filters
if "week" in full_df.columns:
    full_df = full_df[(full_df["week"] >= week_range[0]) & (full_df["week"] <= week_range[1])]

# Compute KPIs
kpis, df_pass_kpis = compute_all_kpis_and_aggregate(full_df, df_input, df_out, df_supp, tr_sample, prethrow)

# ---------------------
# KPI CARDS DISPLAY
# ---------------------
st.markdown("## 📊 Key Performance Indicators")
cols = st.columns(4)
for i, (k, v) in enumerate(list(kpis.items())[:4]):
    delta = np.random.uniform(-10, 10)
    with cols[i]:
        delta_html = f'<span class="kpi-badge-up">+{delta:.2f}%</span>' if delta >= 0 else f'<span class="kpi-badge-down">{delta:.2f}%</span>'
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{k}</div>
            <div class="kpi-value">{v:.2f} {delta_html}</div>
            <div class="kpi-sub">Aggregate Value</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------
# CHART SECTION
# ---------------------
st.markdown("### 📈 KPI Comparison Chart")
df_kpi = pd.DataFrame(list(kpis.items()), columns=["KPI", "Value"])
if chart_type == "Bar":
    fig = px.bar(df_kpi, x="KPI", y="Value", color="KPI", template="plotly_dark", text="Value")
elif chart_type == "Line":
    fig = px.line(df_kpi, x="KPI", y="Value", markers=True, template="plotly_dark")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_kpi["KPI"], y=df_kpi["Value"], fill="tozeroy", mode="lines+markers"))
    fig.update_layout(template="plotly_dark")

fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, font_color=TEXT_COLOR)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("⚫ Dark NFL Theme • Kaggle Integration • Streamlit Cloud Ready")
