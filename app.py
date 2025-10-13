# =======================================================
# app.py - NFL Big Data Bowl 2026 Dashboard (FIXED)
# =======================================================

import sys
import streamlit as st

# =======================================================
# 🔍 SECTION DEBUG - À AFFICHER EN PREMIER
# =======================================================
st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("🔍 Show Debug Info", value=False)

if debug_mode:
    with st.expander("🔍 DEBUG INFORMATION", expanded=True):
        st.write("### 🖥️ System Information")
        st.write(f"**Python version:** {sys.version}")
        st.write(f"**Streamlit version:** {st.__version__}")
        st.write(f"**Working directory:** {sys.path[0]}")
        
        st.write("---")
        st.write("### 📦 Testing Imports...")
        
        # Test pandas
        try:
            import pandas as pd
            st.success(f"✅ pandas {pd.__version__}")
        except Exception as e:
            st.error(f"❌ pandas: {e}")
            st.stop()
        
        # Test numpy
        try:
            import numpy as np
            st.success(f"✅ numpy {np.__version__}")
        except Exception as e:
            st.error(f"❌ numpy: {e}")
            st.stop()
        
        # Test plotly
        try:
            import plotly
            import plotly.express as px
            st.success(f"✅ plotly {plotly.__version__}")
        except Exception as e:
            st.error(f"❌ plotly: {e}")
            st.stop()
        
        # Test datetime
        try:
            from datetime import datetime
            st.success(f"✅ datetime")
        except Exception as e:
            st.error(f"❌ datetime: {e}")
            st.stop()
        
        st.write("---")
        st.write("### 📄 Checking Files...")
        
        # List files
        try:
            import os
            files = os.listdir('.')
            st.write(f"**Files found:** {', '.join(files)}")
            
            if 'utils.py' in files:
                st.success("✅ utils.py found")
            else:
                st.error("❌ utils.py NOT FOUND")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error listing files: {e}")
        
        st.write("---")
        st.write("### 🔧 Testing utils.py...")
        
        # Test utils.py import
        try:
            import utils
            st.success("✅ utils.py imported")
            
            required_functions = [
                'load_data_from_kaggle',
                'compute_all_kpis',
                'load_local_data',
                'get_column_info',
                'detect_available_columns',
                'get_data_summary'
            ]
            
            st.write("**Function checks:**")
            all_ok = True
            for func in required_functions:
                if hasattr(utils, func):
                    st.success(f"✅ {func}")
                else:
                    st.error(f"❌ {func} MISSING")
                    all_ok = False
            
            if not all_ok:
                st.error("⚠️ Some functions are missing!")
                st.stop()
            else:
                st.success("🎉 All functions available!")
                
        except ImportError as e:
            st.error(f"❌ Cannot import utils.py: {e}")
            st.info("**Solutions:**")
            st.info("1. Verify utils.py is in the same folder as app.py")
            st.info("2. Check for syntax errors in utils.py")
            st.info("3. Ensure all dependencies are installed")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

# =======================================================
# 📦 IMPORTS PRINCIPAUX
# =======================================================
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
    st.info("Enable 'Show Debug Info' in sidebar for more details")
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
        value="nfl-big-data-bowl-2026-analytics",
        help="Kaggle competition slug"
    )
    
    if st.sidebar.button("🚀 Load Data from Kaggle", type="primary"):
        if kaggle_username and kaggle_key:
            try:
                with st.spinner("🏈 Downloading NFL data from Kaggle... This may take a few minutes..."):
                    df_result = load_data_from_kaggle(
                        username=kaggle_username,
                        key=kaggle_key,
                        competition=competition_name
                    )
                    
                    # Vérifier que les données ont été chargées
                    if df_result is not None and not df_result.empty:
                        st.session_state.full_df = df_result
                        st.session_state.data_loaded = True
                        st.success(f"✅ Successfully loaded {len(st.session_state.full_df):,} rows!")
                        st.info("📊 Scroll down to see the visualizations!")
                    else:
                        st.error("❌ No data was loaded from Kaggle")
                        st.session_state.data_loaded = False
                        
            except Exception as e:
                st.error(f"❌ Kaggle API Error: {str(e)}")
                with st.expander("🔍 Error Details & Solutions"):
                    st.code(str(e), language="python")
                    st.markdown("""
                    **Common Solutions:**
                    1. ✅ Verify your Kaggle credentials are correct
                    2. ✅ Accept the competition rules at: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/rules
                    3. ✅ Check competition name is: `nfl-big-data-bowl-2026-analytics`
                    4. ✅ Ensure kaggle package is installed: `pip install kaggle`
                    5. ✅ Check your API quota hasn't been exceeded
                    """)
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
            with st.spinner("📂 Loading uploaded files..."):
                dfs = []
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    try:
                        df = pd.read_csv(file, low_memory=False)
                        dfs.append(df)
                        st.sidebar.success(f"✓ {file.name}: {len(df):,} rows")
                    except Exception as e:
                        st.sidebar.error(f"⚠️ Error with {file.name}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                if dfs:
                    st.session_state.full_df = pd.concat(dfs, ignore_index=True)
                    st.session_state.data_loaded = True
                    st.sidebar.success(f"✅ Total: {len(st.session_state.full_df):,} rows!")
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
            with st.spinner(f"📂 Loading CSV files from {data_dir}..."):
                df_result = load_local_data(data_dir)
                if df_result is not None and not df_result.empty:
                    st.session_state.full_df = df_result
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(st.session_state.full_df):,} rows")
                else:
                    st.error("❌ No data loaded")
                    st.session_state.data_loaded = False
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(str(e), language="python")
            st.session_state.data_loaded = False

# ---------------------
# 📈 Visualization Controls (only show if data loaded)
# ---------------------
if st.session_state.data_loaded and not st.session_state.full_df.empty:
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
        
        try:
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
        except Exception as e:
            st.warning(f"⚠️ Could not compute summary: {str(e)}")
            if debug_mode:
                st.code(str(e), language="python")
        
        with st.expander("📊 Available Data Columns"):
            try:
                available_cols = detect_available_columns(df)
                if available_cols:
                    for category, cols in available_cols.items():
                        st.markdown(f"**{category}:** {', '.join(cols)}")
                else:
                    st.write("**All columns:**")
                    col_list = df.columns.tolist()
                    st.write(f"Total: {len(col_list)} columns")
                    st.write(", ".join(col_list[:20]))
                    if len(col_list) > 20:
                        st.write(f"... and {len(col_list) - 20} more")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if debug_mode:
                    st.code(str(e), language="python")
        
        st.markdown("---")
    
    # Compute KPIs
    try:
        with st.spinner("🧮 Computing KPIs..."):
            kpis = compute_all_kpis(df)
    except Exception as e:
        st.error(f"❌ Error computing KPIs: {str(e)}")
        if debug_mode:
            st.code(str(e), language="python")
        kpis = {}
    
    if not kpis:
        st.warning("⚠️ No KPIs could be computed from the data.")
        with st.expander("🔍 Dataset Columns"):
            st.write(df.columns.tolist())
    else:
        # KPI Cards
        st.markdown("## 📊 Key Performance Indicators")
        
        try:
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
        except Exception as e:
            st.error(f"❌ Error displaying KPIs: {str(e)}")
            if debug_mode:
                st.code(str(e), language="python")
        
        st.markdown("---")
        
        # KPI Chart
        st.markdown("## 📈 KPI Visualization")
        
        try:
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
            else:
                st.info("No data available for visualization")
        except Exception as e:
            st.error(f"❌ Chart error: {str(e)}")
            if debug_mode:
                st.code(str(e), language="python")
        
        # Data Preview
        st.markdown("---")
        st.markdown("## 📋 Data Explorer")
        
        with st.expander("🔍 View Sample Data"):
            try:
                st.dataframe(df.head(100), use_container_width=True, height=400)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if debug_mode:
                    st.code(str(e), language="python")
        
        with st.expander("📊 Column Statistics"):
            try:
                col_info = get_column_info(df)
                st.dataframe(col_info, use_container_width=True, height=400)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if debug_mode:
                    st.code(str(e), language="python")

else:
    # Welcome Screen
    st.info("👈 **Get Started:** Use the sidebar to load your NFL data")
    
    st.markdown("""
    ### 🏈 Welcome to NFL Big Data Bowl 2026 Dashboard
    
    #### 📥 How to Load Data:
    
    **Option 1: Kaggle API** (Recommended)
    - Enter your Kaggle username and API key
    - Competition: `nfl-big-data-bowl-2026-analytics`
    - Click "Load Data from Kaggle"
    - ⏱️ First load may take 2-5 minutes
    
    **Option 2: Upload CSV**
    - Select "Upload CSV" option
    - Upload your CSV files from Kaggle
    - Click "Load Uploaded Files"
    
    **Option 3: Local Directory**
    - Enter the path to your data folder
    - Click "Load from Local Directory"
    
    #### 📊 Features:
    - **14+ KPIs** computed automatically
    - **Interactive visualizations** (Bar, Line, Area, Scatter)
    - **Data exploration** tools
    - **Column statistics** and analysis
    
    #### 🔍 Troubleshooting:
    - Check "Show Debug Info" in sidebar for diagnostics
    - Verify all files are in the correct directory
    - Ensure all dependencies are installed
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**NFL Big Data Bowl 2026**")
st.sidebar.caption("v1.0 - Analytics Dashboard")
