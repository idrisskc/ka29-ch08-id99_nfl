# =======================================================
# app.py - NFL Big Data Bowl 2026 Dashboard
# =======================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import functions with error handling
try:
    from utils import (
        load_data_from_kaggle, 
        compute_all_kpis, 
        load_local_data,
        get_column_info,
        detect_available_columns,
        get_data_summary
    )
except ImportError as e:
    st.error(f"❌ Error importing utils.py: {str(e)}")
    st.info("Make sure utils.py is in the same directory as app.py")
    st.stop()

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
st.set_page_config(
    page_title="NFL Big Data Bowl 2026", 
    layout="wide", 
    page_icon="🏈",
    initial_sidebar_state="expanded"
)

# ---------------------
# 🎨 Custom CSS
# ---------------------
st.markdown(f"""
<style>
.stApp {{ 
    background-color: {COLOR_BG}; 
    color: {TEXT_COLOR}; 
}}
[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg, {COLOR_PANEL}, #0b0c10);
    border-right: 1px solid rgba(255,255,255,0.1);
    padding: 1.5rem;
}}
.kpi-card {{
    background: linear-gradient(135deg, rgba(193,18,31,0.1), rgba(27,38,59,0.3));
    border: 1px solid rgba(255,215,0,0.2);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    transition: transform 0.2s;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    border-color: rgba(255,215,0,0.4);
}}
.kpi-title {{ 
    color: {COLOR_SILVER}; 
    font-size: 13px; 
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px; 
}}
.kpi-value {{ 
    font-size: 32px; 
    font-weight: 700; 
    color: {COLOR_GOLD}; 
    margin-bottom: 4px;
}}
.kpi-sub {{ 
    color: rgba(255,255,255,0.5); 
    font-size: 11px; 
}}
</style>
""", unsafe_allow_html=True)

# ---------------------
# 🏈 Header
# ---------------------
st.title("🏈 NFL Big Data Bowl 2026")
st.markdown("_Advanced analytics dashboard for NFL tracking data_")
st.markdown("---")

# ---------------------
# 📊 Initialize Session State
# ---------------------
if 'full_df' not in st.session_state:
    st.session_state.full_df = pd.DataFrame()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# ---------------------
# 📊 Sidebar Controls
# ---------------------
st.sidebar.header("⚙️ Dashboard Controls")
st.sidebar.subheader("📥 Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Kaggle API", "Upload CSV", "Local Directory"],
    help="Load data from Kaggle competition, upload files, or use local directory"
)

# ---------------------
# 🔐 Kaggle API Option
# ---------------------
if data_source == "Kaggle API":
    st.sidebar.markdown("#### Kaggle Credentials")
    
    # Try secrets first, then manual input
    try:
        default_username = st.secrets.get("KAGGLE_USERNAME", "")
        default_key = st.secrets.get("KAGGLE_KEY", "")
    except:
        default_username = ""
        default_key = ""
    
    kaggle_username = st.sidebar.text_input(
        "Username", 
        value=default_username,
        help="Your Kaggle username"
    )
    kaggle_key = st.sidebar.text_input(
        "API Key", 
        type="password",
        value=default_key,
        help="Your Kaggle API key"
    )
    
    competition_name = st.sidebar.text_input(
        "Competition Name",
        value="nfl-big-data-bowl-2025",
        help="Kaggle competition slug"
    )
    
    if st.sidebar.button("🚀 Load Data from Kaggle", type="primary"):
        if kaggle_username and kaggle_key:
            try:
                st.session_state.full_df = load_data_from_kaggle(
                    username=kaggle_username,
                    key=kaggle_key,
                    competition=competition_name
                )
                st.session_state.data_loaded = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.data_loaded = False
        else:
            st.sidebar.warning("⚠️ Please enter both username and API key")

# ---------------------
# 📤 Upload CSV Option
# ---------------------
elif data_source == "Upload CSV":
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files from the NFL dataset"
    )
    
    if st.sidebar.button("🚀 Load Uploaded Files", type="primary"):
        if uploaded_files:
            dfs = []
            for file in uploaded_files:
                try:
                    df = pd.read_csv(file, low_memory=False)
                    dfs.append(df)
                    st.sidebar.success(f"✓ {file.name}: {len(df):,} rows")
                except Exception as e:
                    st.sidebar.error(f"⚠️ Error with {file.name}: {str(e)}")
            
            if dfs:
                st.session_state.full_df = pd.concat(dfs, ignore_index=True)
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Total: {len(st.session_state.full_df):,} rows!")
                st.rerun()
            else:
                st.sidebar.error("❌ No files could be loaded")
        else:
            st.sidebar.warning("⚠️ Please upload at least one CSV file")

# ---------------------
# 📂 Local Directory Option
# ---------------------
else:
    data_dir = st.sidebar.text_input(
        "Data Directory Path",
        value="./data",
        help="Path to directory containing CSV files"
    )
    
    if st.sidebar.button("🚀 Load from Local Directory", type="primary"):
        try:
            st.session_state.full_df = load_local_data(data_dir)
            st.session_state.data_loaded = True
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.data_loaded = False

# ---------------------
# 📈 Visualization Controls (only show if data loaded)
# ---------------------
if st.session_state.data_loaded:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Visualization Settings")
    
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["Bar", "Line", "Area", "Scatter"]
    )
    
    show_top_n = st.sidebar.slider(
        "Show Top N KPIs",
        min_value=5,
        max_value=20,
        value=12
    )
    
    show_data_info = st.sidebar.checkbox(
        "Show Data Information",
        value=True
    )

# ---------------------
# 🎯 Main Dashboard Content
# ---------------------
if st.session_state.data_loaded and not st.session_state.full_df.empty:
    df = st.session_state.full_df
    
    # Data Summary
    if show_data_info:
        st.markdown("## 📋 Dataset Overview")
        
        summary = get_data_summary(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{summary['Total Rows']:,}")
        with col2:
            st.metric("Total Columns", f"{summary['Total Columns']:,}")
        with col3:
            st.metric("Memory Usage", f"{summary['Memory Usage (MB)']:.1f} MB")
        with col4:
            st.metric("Missing Values", f"{summary['Missing %']:.1f}%")
        
        with st.expander("📊 Available Data Columns"):
            available_cols = detect_available_columns(df)
            if available_cols:
                for category, cols in available_cols.items():
                    st.markdown(f"**{category}:** {', '.join(cols)}")
            else:
                st.write("All columns:", df.columns.tolist())
        
        st.markdown("---")
    
    # Compute KPIs
    kpis = compute_all_kpis(df)
    
    if not kpis:
        st.warning("⚠️ No KPIs could be computed. Check your data columns.")
    else:
        # KPI Cards
        st.markdown("## 📊 Key Performance Indicators")
        
        sorted_kpis = dict(sorted(kpis.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)[:show_top_n])
        
        cols_per_row = 4
        kpi_items = list(sorted_kpis.items())
        
        for i in range(0, len(kpi_items), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, (kpi_name, kpi_value) in enumerate(kpi_items[i:i+cols_per_row]):
                with cols[j]:
                    display_value = f"{kpi_value:.2f}" if not np.isnan(kpi_value) else "N/A"
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">{kpi_name}</div>
                        <div class="kpi-value">{display_value}</div>
                        <div class="kpi-sub">Computed Value</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # KPI Chart
        st.markdown("## 📈 KPI Visualization")
        
        df_kpi = pd.DataFrame([
            {"KPI": k, "Value": v} 
            for k, v in sorted_kpis.items() 
            if not np.isnan(v)
        ])
        
        if not df_kpi.empty:
            if chart_type == "Bar":
                fig = px.bar(df_kpi, x="KPI", y="Value", color="Value", 
                           color_continuous_scale="Viridis", template="plotly_dark")
            elif chart_type == "Line":
                fig = px.line(df_kpi, x="KPI", y="Value", markers=True, template="plotly_dark")
            elif chart_type == "Area":
                fig = px.area(df_kpi, x="KPI", y="Value", template="plotly_dark")
            else:
                fig = px.scatter(df_kpi, x="KPI", y="Value", size="Value", color="Value", 
                               color_continuous_scale="Plasma", template="plotly_dark")
            
            fig.update_layout(
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_PANEL,
                font_color=TEXT_COLOR,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Data Preview
        st.markdown("---")
        st.markdown("## 📋 Data Explorer")
        
        with st.expander("🔍 View Sample Data"):
            st.dataframe(df.head(100), use_container_width=True, height=400)
        
        with st.expander("📊 Column Statistics"):
            st.dataframe(get_column_info(df), use_container_width=True, height=400)

else:
    # Welcome Screen
    st.info("👈 **Get Started:** Use the sidebar to load your NFL data")
    
    st.markdown("""
    ### 🏈 Welcome to NFL Big Data Bowl 2026 Dashboard
    
    #### 📥 How to Load Data:
    
    **Option 1: Kaggle API**
    - Enter your Kaggle username and API key
    - Specify the competition name
    - Click "Load Data from Kaggle"
    
    **Option 2: Upload CSV**
    - Select "Upload CSV" option
    - Upload your CSV files
    - Click "Load Uploaded Files"
    
    **Option 3: Local Directory**
    - Enter the path to your data folder
    - Click "Load from Local Directory"
    
    #### 📊 Features:
    - **14+ KPIs** computed automatically
    - **Interactive visualizations** (Bar, Line, Area, Scatter)
    - **Data exploration** tools
    - **Column statistics** and analysis
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**NFL Big Data Bowl 2026**")
st.sidebar.caption("v1.0 - Analytics Dashboard")
