# =======================================================
# app.py - NFL Big Data Bowl 2026 Dashboard
# =======================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils import load_data_from_kaggle, compute_all_kpis

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
    page_icon="🏈"
)

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
.metric-description {{
    color: rgba(255,255,255,0.4);
    font-size: 10px;
    font-style: italic;
    margin-top: 4px;
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
# 📊 Sidebar Controls
# ---------------------
st.sidebar.header("⚙️ Dashboard Controls")

# Data source selection
st.sidebar.subheader("📥 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Kaggle API", "Upload CSV"],
    help="Load data from Kaggle competition or upload your own files"
)

# Initialize session state for data
if 'full_df' not in st.session_state:
    st.session_state.full_df = pd.DataFrame()

# ---------------------
# 🔐 Kaggle Authentication
# ---------------------
if data_source == "Kaggle API":
    st.sidebar.markdown("#### Kaggle Credentials")
    
    # Try to get from secrets first, then allow manual input
    default_username = st.secrets.get("KAGGLE_USERNAME", "")
    default_key = st.secrets.get("KAGGLE_KEY", "")
    
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
        help="Name of the Kaggle competition"
    )
    
    if st.sidebar.button("🚀 Load Data from Kaggle", type="primary"):
        if kaggle_username and kaggle_key:
            with st.spinner("🏈 Loading NFL data from Kaggle..."):
                try:
                    st.session_state.full_df = load_data_from_kaggle(
                        username=kaggle_username,
                        key=kaggle_key,
                        competition=competition_name
                    )
                    st.sidebar.success(f"✅ Loaded {len(st.session_state.full_df):,} rows!")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Please enter both username and API key")

# ---------------------
# 📤 File Upload
# ---------------------
else:  # Upload CSV
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files from the NFL dataset"
    )
    
    if st.sidebar.button("🚀 Load Uploaded Files", type="primary"):
        if uploaded_files:
            with st.spinner("📂 Loading uploaded files..."):
                dfs = []
                for file in uploaded_files:
                    df = pd.read_csv(file, low_memory=False)
                    dfs.append(df)
                    st.sidebar.info(f"✓ {file.name}: {len(df):,} rows")
                
                st.session_state.full_df = pd.concat(dfs, ignore_index=True)
                st.sidebar.success(f"✅ Total: {len(st.session_state.full_df):,} rows loaded!")
        else:
            st.sidebar.warning("⚠️ Please upload at least one CSV file")

# ---------------------
# 📈 Visualization Controls
# ---------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Visualization Settings")

chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Bar", "Line", "Area", "Scatter"],
    help="Select visualization type for KPIs"
)

show_top_n = st.sidebar.slider(
    "Show Top N KPIs",
    min_value=5,
    max_value=14,
    value=10,
    help="Display top performing metrics"
)

# ---------------------
# 🎯 Main Dashboard
# ---------------------
if not st.session_state.full_df.empty:
    df = st.session_state.full_df
    
    # Compute KPIs
    with st.spinner("🧮 Computing KPIs..."):
        kpis = compute_all_kpis(df)
    
    # ---------------------
    # 📊 KPI Cards Display
    # ---------------------
    st.markdown("## 📊 Key Performance Indicators")
    
    # Sort KPIs by value and take top N
    sorted_kpis = dict(sorted(kpis.items(), key=lambda x: x[1], reverse=True)[:show_top_n])
    
    # Display in responsive columns
    cols_per_row = 4
    kpi_items = list(sorted_kpis.items())
    
    for i in range(0, len(kpi_items), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (kpi_name, kpi_value) in enumerate(kpi_items[i:i+cols_per_row]):
            with cols[j]:
                # Handle NaN values
                display_value = f"{kpi_value:.2f}" if not np.isnan(kpi_value) else "N/A"
                
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">{kpi_name}</div>
                    <div class="kpi-value">{display_value}</div>
                    <div class="kpi-sub">Average Value</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---------------------
    # 📈 KPI Visualization
    # ---------------------
    st.markdown("## 📈 KPI Trends")
    
    # Prepare data for visualization
    df_kpi = pd.DataFrame([
        {"KPI": k, "Value": v} 
        for k, v in sorted_kpis.items() 
        if not np.isnan(v)
    ])
    
    if not df_kpi.empty:
        # Create visualization based on selection
        if chart_type == "Bar":
            fig = px.bar(
                df_kpi, 
                x="KPI", 
                y="Value",
                color="Value",
                color_continuous_scale="Viridis",
                template="plotly_dark",
                title="KPI Overview"
            )
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
            
        elif chart_type == "Line":
            fig = px.line(
                df_kpi, 
                x="KPI", 
                y="Value",
                markers=True,
                template="plotly_dark",
                title="KPI Trends"
            )
            
        elif chart_type == "Area":
            fig = px.area(
                df_kpi, 
                x="KPI", 
                y="Value",
                template="plotly_dark",
                title="KPI Distribution"
            )
            
        else:  # Scatter
            fig = px.scatter(
                df_kpi, 
                x="KPI", 
                y="Value",
                size="Value",
                color="Value",
                color_continuous_scale="Plasma",
                template="plotly_dark",
                title="KPI Analysis"
            )
        
        # Update layout
        fig.update_layout(
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_PANEL,
            font_color=TEXT_COLOR,
            title_font_size=20,
            xaxis_title="Metric",
            yaxis_title="Value",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ---------------------
    # 📋 Data Preview
    # ---------------------
    st.markdown("---")
    st.markdown("## 📋 Data Preview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", f"{len(df.columns):,}")
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Show sample data
    with st.expander("🔍 View Sample Data", expanded=False):
        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=400
        )
    
    # Column statistics
    with st.expander("📊 Column Statistics", expanded=False):
        st.dataframe(
            df.describe(),
            use_container_width=True
        )

else:
    # ---------------------
    # 🎯 Welcome Screen
    # ---------------------
    st.info("👈 **Get Started:** Use the sidebar to load your NFL data")
    
    st.markdown("""
    ### 🏈 Welcome to NFL Big Data Bowl 2026 Dashboard
    
    This dashboard provides comprehensive analytics for NFL tracking data. Here's how to get started:
    
    #### 📥 Loading Data
    
    **Option 1: Kaggle API**
    1. Enter your Kaggle credentials in the sidebar
    2. Click "Load Data from Kaggle"
    3. Wait for data to load
    
    **Option 2: Upload CSV**
    1. Select "Upload CSV" in the sidebar
    2. Upload one or more CSV files
    3. Click "Load Uploaded Files"
    
    #### 📊 Available KPIs
    
    - **PPE** - Yards Gained
    - **CBR** - Completion Probability
    - **ADY** - Distance Metrics
    - **TDR** - Time Analysis
    - **CWE** - Closest Defender Distance
    - **EDS** - End Speed
    - **VMC** - Velocity Metrics
    - And 7 more advanced metrics...
    
    #### 🎨 Visualization Options
    
    Choose from multiple chart types and customize your view using the sidebar controls.
    """)
    
    st.markdown("---")
    st.markdown("_Built with Streamlit • Powered by NFL Big Data Bowl 2025_")

# ---------------------
# 📍 Footer
# ---------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info("""
**NFL Big Data Bowl 2026**  
Advanced Analytics Dashboard  
Version 1.0
""")
