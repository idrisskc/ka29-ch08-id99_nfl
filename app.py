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
from datetime import datetime
from utils import load_data, compute_all_kpis_and_aggregate, download_from_kaggle

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
# ⚙️ Streamlit Config
# ---------------------
st.set_page_config(page_title="NFL Big Data Bowl 2026", layout="wide", page_icon="🏈")

st.markdown(f"""
<style>
.stApp {{
    background-color: {COLOR_BG};
    color: {TEXT_COLOR};
}}
[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg, {COLOR_PANEL}, #0b0c10);
    border-right: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
}}
.kpi-card {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 10px;
}}
.kpi-title {{ color: {COLOR_SILVER}; font-size:13px; margin-bottom:6px; }}
.kpi-value {{ font-size:28px; font-weight:700; color:{COLOR_GOLD}; }}
.kpi-sub {{ color: rgba(255,255,255,0.6); font-size:12px; }}
</style>
""", unsafe_allow_html=True)

st.title("🏈 NFL Big Data Bowl 2026 - Analytics Dashboard")
st.markdown("_Analyze and visualize player tracking data from the official Kaggle competition._")

# ---------------------
# Sidebar
# ---------------------
st.sidebar.header("⚙️ Dashboard Controls")

source = st.sidebar.radio("Data Source:", ["Kaggle", "Local ./data/"])
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

if source == "Kaggle":
    kaggle_user = st.sidebar.text_input("Kaggle Username", value=st.secrets.get("KAGGLE_USERNAME", ""))
    kaggle_key = st.sidebar.text_input("Kaggle Key", type="password", value=st.secrets.get("KAGGLE_KEY", ""))
    if st.sidebar.button("📦 Download from Kaggle"):
        with st.spinner("Downloading dataset from Kaggle..."):
            download_from_kaggle(kaggle_user, kaggle_key, data_dir)
        st.sidebar.success("✅ Download and extraction complete!")

if st.sidebar.button("🚀 Load Data"):
    with st.spinner("Loading data..."):
        full_df, df_input, df_out, df_supp, tr_sample, prethrow = load_data(data_dir)

        if full_df.empty:
            st.error("❌ No data found after extraction. Please check ./data folder.")
            st.stop()
        else:
            st.success(f"✅ {len(full_df)} rows loaded successfully!")

            # Display file overview
            st.sidebar.markdown("### 📄 Available Files:")
            for f in os.listdir(data_dir):
                st.sidebar.write(f"- {f}")

            # ---------------------
            # KPIs & Charts
            # ---------------------
            kpis, df_pass_kpis = compute_all_kpis_and_aggregate(full_df, df_input, df_out, df_supp, tr_sample, prethrow)

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

            # Chart
            df_kpi = pd.DataFrame(list(kpis.items()), columns=["KPI", "Value"])
            fig = px.bar(df_kpi, x="KPI", y="Value", color="KPI", template="plotly_dark", text="Value")
            fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, font_color=TEXT_COLOR)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("⬅️ Use the sidebar to download or load your data.")
