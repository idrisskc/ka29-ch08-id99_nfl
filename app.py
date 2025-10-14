# =======================================================
# app.py - NFL Big Data Bowl 2026 Dashboard with Strategic KPIs
# =======================================================

import sys
import streamlit as st

# =======================================================
# 🔍 SECTION DEBUG
# =======================================================
st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("🔍 Show Debug Info", value=False)

if debug_mode:
    with st.expander("🔍 DEBUG INFORMATION", expanded=True):
        st.write("### 🖥️ System Information")
        st.write(f"**Python version:** {sys.version}")
        st.write(f"**Streamlit version:** {st.__version__}")
        
        st.write("---")
        st.write("### 📦 Testing Imports...")
        
        try:
            import pandas as pd
            st.success(f"✅ pandas {pd.__version__}")
        except Exception as e:
            st.error(f"❌ pandas: {e}")
            st.stop()
        
        try:
            import numpy as np
            st.success(f"✅ numpy {np.__version__}")
        except Exception as e:
            st.error(f"❌ numpy: {e}")
            st.stop()
        
        try:
            import plotly
            import plotly.express as px
            st.success(f"✅ plotly {plotly.__version__}")
        except Exception as e:
            st.error(f"❌ plotly: {e}")
            st.stop()
        
        st.write("---")
        st.write("### 📄 Checking Files...")
        
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

# =======================================================
# 📦 IMPORTS PRINCIPAUX
# =======================================================
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

try:
    from utils import (
        load_data_from_kaggle, 
        compute_all_kpis, 
        load_local_data,
        get_column_info,
        detect_available_columns,
        get_data_summary,
        calculate_all_strategic_kpis,
        get_strategic_kpi_dataframes
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
st.markdown("_Advanced analytics dashboard with strategic KPIs_")
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

# Page Selection
page = st.sidebar.radio(
    "📄 Select View:",
    ["Overview", "Strategic KPIs", "Data Explorer"],
    help="Choose which dashboard page to display"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Kaggle API", "Upload CSV", "Local Directory"]
)

# ---------------------
# 🔐 Data Loading
# ---------------------
if data_source == "Kaggle API":
    st.sidebar.markdown("#### Kaggle Credentials")
    
    try:
        default_username = st.secrets.get("KAGGLE_USERNAME", "")
        default_key = st.secrets.get("KAGGLE_KEY", "")
    except:
        default_username = ""
        default_key = ""
    
    kaggle_username = st.sidebar.text_input("Username", value=default_username)
    kaggle_key = st.sidebar.text_input("API Key", type="password", value=default_key)
    competition_name = st.sidebar.text_input("Competition", value="nfl-big-data-bowl-2026-analytics")
    
    if st.sidebar.button("🚀 Load Data from Kaggle", type="primary"):
        if kaggle_username and kaggle_key:
            try:
                with st.spinner("🏈 Downloading NFL data..."):
                    df_result = load_data_from_kaggle(kaggle_username, kaggle_key, competition_name)
                    
                    if df_result is not None and not df_result.empty:
                        st.session_state.full_df = df_result
                        st.session_state.data_loaded = True
                        st.success(f"✅ Loaded {len(st.session_state.full_df):,} rows!")
                    else:
                        st.error("❌ No data loaded")
                        st.session_state.data_loaded = False
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.data_loaded = False
        else:
            st.sidebar.warning("⚠️ Please enter credentials")

elif data_source == "Upload CSV":
    uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)
    
    if st.sidebar.button("🚀 Load Uploaded Files", type="primary"):
        if uploaded_files:
            with st.spinner("📂 Loading files..."):
                dfs = []
                for file in uploaded_files:
                    try:
                        df = pd.read_csv(file, low_memory=False)
                        dfs.append(df)
                        st.sidebar.success(f"✓ {file.name}: {len(df):,} rows")
                    except Exception as e:
                        st.sidebar.error(f"⚠️ {file.name}: {str(e)}")
                
                if dfs:
                    st.session_state.full_df = pd.concat(dfs, ignore_index=True)
                    st.session_state.data_loaded = True
                    st.sidebar.success(f"✅ Total: {len(st.session_state.full_df):,} rows!")

else:
    data_dir = st.sidebar.text_input("Data Directory", value="./data")
    
    if st.sidebar.button("🚀 Load from Local", type="primary"):
        try:
            df_result = load_local_data(data_dir)
            if df_result is not None and not df_result.empty:
                st.session_state.full_df = df_result
                st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ---------------------
# 📈 Visualization Controls
# ---------------------
if st.session_state.data_loaded and not st.session_state.full_df.empty:
    st.sidebar.markdown("---")
