# =======================================================
# app.py - NFL Big Data Bowl 2026 Dashboard (DEBUG VERSION)
# =======================================================

import sys
import streamlit as st

# Afficher les informations de débogage
st.write("🔍 **Debug Info:**")
st.write(f"Python version: {sys.version}")
st.write(f"Streamlit version: {st.__version__}")

# Test des imports étape par étape
try:
    import pandas as pd
    st.success("✅ pandas imported")
except Exception as e:
    st.error(f"❌ pandas error: {e}")
    st.stop()

try:
    import numpy as np
    st.success("✅ numpy imported")
except Exception as e:
    st.error(f"❌ numpy error: {e}")
    st.stop()

try:
    import plotly.express as px
    st.success("✅ plotly imported")
except Exception as e:
    st.error(f"❌ plotly error: {e}")
    st.stop()

try:
    from datetime import datetime
    st.success("✅ datetime imported")
except Exception as e:
    st.error(f"❌ datetime error: {e}")
    st.stop()

# Test de l'import de utils.py
try:
    import utils
    st.success("✅ utils.py found and imported")
    
    # Vérifier les fonctions
    required_functions = [
        'load_data_from_kaggle',
        'compute_all_kpis',
        'load_local_data',
        'get_column_info',
        'detect_available_columns',
        'get_data_summary'
    ]
    
    for func_name in required_functions:
        if hasattr(utils, func_name):
            st.success(f"✅ Function '{func_name}' exists")
        else:
            st.error(f"❌ Function '{func_name}' missing")
    
except Exception as e:
    st.error(f"❌ utils.py import error: {e}")
    st.info("**Solution:** Make sure utils.py is in the same directory as app.py")
    st.code(str(e), language="python")
    st.stop()

# Si tout est OK, continuer avec l'app normale
st.success("🎉 All imports successful! Starting main app...")

# ---------------------
# ⚙️ Page Configuration
# ---------------------
st.set_page_config(
    page_title="NFL Big Data Bowl 2026", 
    layout="wide", 
    page_icon="🏈",
    initial_sidebar_state="expanded"
)

# Import des fonctions depuis utils
from utils import (
    load_data_from_kaggle, 
    compute_all_kpis, 
    load_local_data,
    get_column_info,
    detect_available_columns,
    get_data_summary
)

# ---------------------
# 🎨 NFL Color Palette
# ---------------------
COLOR_BG = "#0B0C10"
COLOR_PANEL = "#1B263B"
COLOR_GOLD = "#FFD700"
COLOR_SILVER = "#A9A9A9"
TEXT_COLOR = "#E6EEF8"

# ---------------------
# 🎨 Custom CSS
# ---------------------
st.markdown(f"""
<style>
.stApp {{ background-color: {COLOR_BG}; color: {TEXT_COLOR}; }}
.kpi-card {{
    background: linear-gradient(135deg, rgba(193,18,31,0.1), rgba(27,38,59,0.3));
    border: 1px solid rgba(255,215,0,0.2);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}}
.kpi-title {{ color: {COLOR_SILVER}; font-size: 13px; font-weight: 600; margin-bottom: 8px; }}
.kpi-value {{ font-size: 32px; font-weight: 700; color: {COLOR_GOLD}; }}
.kpi-sub {{ color: rgba(255,255,255,0.5); font-size: 11px; }}
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
    ["Upload CSV", "Kaggle API", "Local Directory"]
)

# ---------------------
# 📤 Upload CSV Option (Premier choix - plus simple)
# ---------------------
if data_source == "Upload CSV":
    st.sidebar.markdown("#### Upload Your Data")
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files from the NFL dataset"
    )
    
    if uploaded_files and st.sidebar.button("🚀 Load Files", type="primary"):
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
                st.sidebar.success(f"✅ {len(st.session_state.full_df):,} rows loaded!")
                st.rerun()

# ---------------------
# 🔐 Kaggle API Option
# ---------------------
elif data_source == "Kaggle API":
    st.sidebar.markdown("#### Kaggle Credentials")
    
    try:
        default_username = st.secrets.get("KAGGLE_USERNAME", "")
        default_key = st.secrets.get("KAGGLE_KEY", "")
    except:
        default_username = ""
        default_key = ""
    
    kaggle_username = st.sidebar.text_input("Username", value=default_username)
    kaggle_key = st.sidebar.text_input("API Key", type="password", value=default_key)
    competition_name = st.sidebar.text_input("Competition", value="nfl-big-data-bowl-2025")
    
    if st.sidebar.button("🚀 Load from Kaggle", type="primary"):
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
                st.error(f"❌ Kaggle Error: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Enter credentials")

# ---------------------
# 📂 Local Directory Option
# ---------------------
else:
    st.sidebar.markdown("#### Local Data Path")
    data_dir = st.sidebar.text_input("Directory Path", value="./data")
    
    if st.sidebar.button("🚀 Load from Local", type="primary"):
        try:
            st.session_state.full_df = load_local_data(data_dir)
            st.session_state.data_loaded = True
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ---------------------
# 📈 Visualization Controls
# ---------------------
if st.session_state.data_loaded:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Settings")
    
    chart_type = st.sidebar.selectbox("Chart Type", ["Bar", "Line", "Area", "Scatter"])
    show_top_n = st.sidebar.slider("Top N KPIs", 5, 20, 12)
    show_data_info = st.sidebar.checkbox("Show Data Info", value=True)

# ---------------------
# 🎯 Main Dashboard
# ---------------------
if st.session_state.data_loaded and not st.session_state.full_df.empty:
    df = st.session_state.full_df
    
    # Data Summary
    if show_data_info:
        st.markdown("## 📋 Dataset Overview")
        summary = get_data_summary(df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{summary['Total Rows']:,}")
        col2.metric("Columns", f"{summary['Total Columns']:,}")
        col3.metric("Memory", f"{summary['Memory Usage (MB)']:.1f} MB")
        col4.metric("Missing", f"{summary['Missing %']:.1f}%")
        
        with st.expander("📊 Available Columns"):
            available = detect_available_columns(df)
            if available:
                for cat, cols in available.items():
                    st.markdown(f"**{cat}:** {', '.join(cols)}")
            else:
                st.write(df.columns.tolist())
        
        st.markdown("---")
    
    # KPIs
    with st.spinner("🧮 Computing KPIs..."):
        kpis = compute_all_kpis(df)
    
    if kpis:
        st.markdown("## 📊 Key Performance Indicators")
        
        sorted_kpis = dict(sorted(kpis.items(), 
                                 key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, 
                                 reverse=True)[:show_top_n])
        
        # Display KPIs in cards
        kpi_items = list(sorted_kpis.items())
        for i in range(0, len(kpi_items), 4):
            cols = st.columns(4)
            for j, (name, value) in enumerate(kpi_items[i:i+4]):
                with cols[j]:
                    display = f"{value:.2f}" if not np.isnan(value) else "N/A"
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">{name}</div>
                        <div class="kpi-value">{display}</div>
                        <div class="kpi-sub">Value</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Chart
        st.markdown("## 📈 KPI Visualization")
        df_kpi = pd.DataFrame([{"KPI": k, "Value": v} 
                               for k, v in sorted_kpis.items() 
                               if not np.isnan(v)])
        
        if not df_kpi.empty:
            if chart_type == "Bar":
                fig = px.bar(df_kpi, x="KPI", y="Value", color="Value", template="plotly_dark")
            elif chart_type == "Line":
                fig = px.line(df_kpi, x="KPI", y="Value", markers=True, template="plotly_dark")
            elif chart_type == "Area":
                fig = px.area(df_kpi, x="KPI", y="Value", template="plotly_dark")
            else:
                fig = px.scatter(df_kpi, x="KPI", y="Value", size="Value", template="plotly_dark")
            
            fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_PANEL, 
                            font_color=TEXT_COLOR, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data Preview
        st.markdown("---")
        st.markdown("## 📋 Data Explorer")
        
        with st.expander("🔍 Sample Data"):
            st.dataframe(df.head(100), use_container_width=True, height=400)
        
        with st.expander("📊 Statistics"):
            st.dataframe(get_column_info(df), use_container_width=True)
    else:
        st.warning("⚠️ No KPIs computed. Check your data columns.")

else:
    # Welcome
    st.info("👈 **Get Started:** Load your NFL data from the sidebar")
    
    st.markdown("""
    ### 🏈 Welcome to NFL Big Data Bowl 2026
    
    #### 📥 Quick Start:
    
    1. **Upload CSV** (Easiest) - Upload your CSV files directly
    2. **Kaggle API** - Connect to Kaggle competition
    3. **Local Directory** - Load from a folder path
    
    #### 📊 Features:
    - 14+ KPIs computed automatically
    - Interactive charts
    - Data exploration tools
    """)

st.sidebar.markdown("---")
st.sidebar.caption("NFL Big Data Bowl 2026 v1.0")
