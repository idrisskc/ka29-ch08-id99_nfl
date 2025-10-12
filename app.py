# =======================================================
# app.py - NFL Big Data Bowl 2026 Dashboard (WITH DEBUG)
# =======================================================
import sys
import streamlit as st

# =======================================================
# 🔍 DEBUG MODE - Show import status
# =======================================================
st.sidebar.markdown("### 🔍 Debug Info")

try:
    st.sidebar.success(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
except:
    pass

# Test imports step by step
try:
    import pandas as pd
    st.sidebar.success("✅ pandas")
except Exception as e:
    st.sidebar.error(f"❌ pandas: {e}")
    st.stop()

try:
    import numpy as np
    st.sidebar.success("✅ numpy")
except Exception as e:
    st.sidebar.error(f"❌ numpy: {e}")
    st.stop()

try:
    import plotly.express as px
    import plotly.graph_objects as go
    st.sidebar.success("✅ plotly")
except Exception as e:
    st.sidebar.error(f"❌ plotly: {e}")
    st.stop()

try:
    from datetime import datetime
    st.sidebar.success("✅ datetime")
except Exception as e:
    st.sidebar.error(f"❌ datetime: {e}")
    st.stop()

# Test utils.py import
try:
    from utils import (
        load_data_from_kaggle, 
        compute_all_kpis, 
        load_local_data,
        get_column_info,
        detect_available_columns,
        get_data_summary
    )
    st.sidebar.success("✅ utils.py imported")
except ImportError as e:
    st.sidebar.error(f"❌ utils.py: {str(e)}")
    st.error("**ERROR:** utils.py not found or has import errors")
    st.info("Make sure utils.py is in the same directory as app.py")
    st.code(str(e), language="python")
    st.stop()
except Exception as e:
    st.sidebar.error(f"❌ utils.py error: {str(e)}")
    st.error(f"**ERROR in utils.py:** {str(e)}")
    st.stop()

st.sidebar.success("🎉 All imports OK!")
st.sidebar.markdown("---")

# =======================================================
# ⚙️ Page Configuration
# =======================================================
st.set_page_config(
    page_title="NFL Big Data Bowl 2026", 
    layout="wide", 
    page_icon="🏈",
    initial_sidebar_state="expanded"
)

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
    ["Upload CSV", "Kaggle API", "Local Directory"],
    help="Load data from Kaggle competition, upload files, or use local directory"
)

# ---------------------
# 📤 Upload CSV Option (First - Simplest)
# ---------------------
if data_source == "Upload CSV":
    st.sidebar.markdown("#### 📤 Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files from the NFL dataset"
    )
    
    if uploaded_files and st.sidebar.button("🚀 Load Uploaded Files", type="primary"):
        with st.spinner("📂 Loading files..."):
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

# ---------------------
# 🔐 Kaggle API Option
# ---------------------
elif data_source == "Kaggle API":
    st.sidebar.markdown("#### 🔐 Kaggle Credentials")
    
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
                with st.spinner("🏈 Downloading from Kaggle..."):
                    st.session_state.full_df = load_data_from_kaggle(
                        username=kaggle_username,
                        key=kaggle_key,
                        competition=competition_name
                    )
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Data loaded successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Kaggle Error:")
                st.error(f"**Error details:** {str(e)}")
                st.info("**Tips:**")
                st.info("- Check if you've accepted the competition rules")
                st.info("- Verify your credentials are correct")
                st.info("- Make sure the competition name is exact")
                st.session_state.data_loaded = False
        else:
            st.sidebar.warning("⚠️ Please enter both username and API key")

# ---------------------
# 📂 Local Directory Option
# ---------------------
else:
    st.sidebar.markdown("#### 📂 Local Path")
    data_dir = st.sidebar.text_input(
        "Data Directory Path",
        value="./data",
        help="Path to directory containing CSV files"
    )
    
    if st.sidebar.button("🚀 Load from Local Directory", type="primary"):
        try:
            with st.spinner(f"📂 Loading from {data_dir}..."):
                st.session_state.full_df = load_local_data(data_dir)
            st.session_state.data_loaded = True
            st.sidebar.success("✅ Data loaded!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error:")
            st.error(f"**Error details:** {str(e)}")
            st.session_state.data_loaded = False

# ---------------------
# 📈 Visualization Controls (only show if data loaded)
# ---------------------
if st.session_state.data_loaded:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Visualization Settings")
    
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["Bar", "Line", "Area", "Scatter"],
        help="Select chart type for KPI visualization"
    )
    
    show_top_n = st.sidebar.slider(
        "Show Top N KPIs",
        min_value=5,
        max_value=20,
        value=12,
        help="Number of top KPIs to display"
    )
    
    show_data_info = st.sidebar.checkbox(
        "Show Data Information",
        value=True,
        help="Display dataset statistics"
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
            
            with st.expander("📊 Available Data Columns"):
                available_cols = detect_available_columns(df)
                if available_cols:
                    for category, cols in available_cols.items():
                        st.markdown(f"**{category}:** {', '.join(cols)}")
                else:
                    st.write("**All columns:**", df.columns.tolist())
        except Exception as e:
            st.warning(f"⚠️ Could not generate summary: {str(e)}")
        
        st.markdown("---")
    
    # Compute KPIs
    try:
        with st.spinner("🧮 Computing KPIs..."):
            kpis = compute_all_kpis(df)
    except Exception as e:
        st.error(f"❌ Error computing KPIs: {str(e)}")
        kpis = {}
    
    if not kpis:
        st.warning("⚠️ No KPIs could be computed. Check your data columns.")
        st.info("**Available columns in your data:**")
        st.write(df.columns.tolist())
    else:
        # KPI Cards
        st.markdown("## 📊 Key Performance Indicators")
        
        try:
            sorted_kpis = dict(sorted(
                kpis.items(), 
                key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, 
                reverse=True
            )[:show_top_n])
            
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
                               color_continuous_scale="Viridis", template="plotly_dark",
                               title="KPI Overview")
                elif chart_type == "Line":
                    fig = px.line(df_kpi, x="KPI", y="Value", markers=True, 
                                template="plotly_dark", title="KPI Trends")
                elif chart_type == "Area":
                    fig = px.area(df_kpi, x="KPI", y="Value", 
                                template="plotly_dark", title="KPI Distribution")
                else:  # Scatter
                    fig = px.scatter(df_kpi, x="KPI", y="Value", size="Value", color="Value", 
                                   color_continuous_scale="Plasma", template="plotly_dark",
                                   title="KPI Analysis")
                
                fig.update_layout(
                    paper_bgcolor=COLOR_BG,
                    plot_bgcolor=COLOR_PANEL,
                    font_color=TEXT_COLOR,
                    height=500,
                    xaxis_title="Metric",
                    yaxis_title="Value"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for visualization")
        except Exception as e:
            st.error(f"❌ Error creating chart: {str(e)}")
        
        # Data Preview
        st.markdown("---")
        st.markdown("## 📋 Data Explorer")
        
        try:
            with st.expander("🔍 View Sample Data (First 100 Rows)"):
                st.dataframe(df.head(100), use_container_width=True, height=400)
        except Exception as e:
            st.error(f"❌ Error displaying data: {str(e)}")
        
        try:
            with st.expander("📊 Column Statistics"):
                col_info = get_column_info(df)
                st.dataframe(col_info, use_container_width=True, height=400)
        except Exception as e:
            st.error(f"❌ Error displaying statistics: {str(e)}")

else:
    # Welcome Screen
    st.info("👈 **Get Started:** Use the sidebar to load your NFL data")
    
    st.markdown("""
    ### 🏈 Welcome to NFL Big Data Bowl 2026 Dashboard
    
    #### 📥 How to Load Data:
    
    **Option 1: Upload CSV** (Recommended - Easiest)
    - Click "Upload CSV" in the sidebar
    - Select your CSV files
    - Click "Load Uploaded Files"
    
    **Option 2: Kaggle API**
    - Enter your Kaggle username and API key
    - Competition name: `nfl-big-data-bowl-2026-analytics`
    - Click "Load Data from Kaggle"
    
    **Option 3: Local Directory**
    - Enter the path to your data folder
    - Click "Load from Local Directory"
    
    #### 📊 Features:
    - **14+ KPIs** computed automatically
    - **Interactive visualizations** (Bar, Line, Area, Scatter)
    - **Data exploration** tools
    - **Column statistics** and analysis
    
    #### 🔍 Debug Mode:
    Check the sidebar for import status and diagnostics.
    """)
    
    # Show sample data structure
    with st.expander("📋 Expected Data Structure"):
        st.markdown("""
        Your CSV files should contain NFL tracking data with columns like:
        - `x`, `y` - Player positions
        - `s`, `a` - Speed and acceleration
        - `gameId`, `playId`, `nflId` - Identifiers
        - `yards_gained`, `completion_probability` - Performance metrics
        
        The dashboard will automatically detect available columns and compute relevant KPIs.
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**NFL Big Data Bowl 2026**")
st.sidebar.caption("v1.0 - Analytics Dashboard")
st.sidebar.caption("Debug mode enabled")
