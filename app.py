# =======================================================
# app.py - NFL Big Data Bowl 2026 Dashboard (COMPLETE OPTIMIZED)
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
                
            if 'chart_visualizer.py' in files:
                st.success("✅ chart_visualizer.py found")
            else:
                st.warning("⚠️ chart_visualizer.py NOT FOUND (optional)")
        except Exception as e:
            st.error(f"❌ Error listing files: {e}")
        
        st.write("---")
        st.write("### 🔧 Testing utils.py...")
        
        try:
            import utils
            st.success("✅ utils.py imported")
            
            required_functions = [
                'load_data_from_kaggle',
                'compute_all_kpis',
                'calculate_all_strategic_kpis',
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

try:
    from utils import (
        load_data_from_kaggle, 
        compute_all_kpis,
        calculate_all_strategic_kpis,
        load_local_data,
        get_column_info,
        detect_available_columns,
        get_data_summary
    )
    from chart_visualizer import visualize_kpi
except ImportError as e:
    st.error(f"❌ Error importing modules: {str(e)}")
    st.info("Make sure utils.py and chart_visualizer.py are in the same directory as app.py")
    st.info("Enable 'Show Debug Info' in sidebar for more details")
    st.stop()

# =======================================================
# 🎨 NFL COLOR PALETTE & CHART CONFIGS
# =======================================================
COLOR_BG = "#0B0C10"
COLOR_PANEL = "#1B263B"
COLOR_ACCENT = "#C1121F"
COLOR_GOLD = "#FFD700"
COLOR_SILVER = "#A9A9A9"
TEXT_COLOR = "#E6EEF8"

# Chart type recommendations for each KPI
CHART_RECOMMENDATIONS = {
    'QB_Pressure': '🔥 Heatmap / Box Plot',
    'Route_Efficiency': '🕸️ Radar Chart',
    'Coverage_Heat': '🔥 NFL Field Heatmap',
    'Pass_Timing': '📊 Step Chart / Bar',
    'Separation': '🫧 3D/2D Scatter Plot',
    'Formation_Tendency': '☀️ Sunburst',
    'Win_Probability': '💧 Waterfall',
    'Defense_Reaction': '🎻 Violin Plot',
    'RedZone_Success': '⏱️ Gauge Chart',
    'Tempo_Analysis': '📈 Bar Chart',
    'Movement_Heat': '🔥 NFL Field Heatmap',
    'Pass_Results': '🍩 Doughnut Chart',
    'EP_Analysis': '💧 Waterfall',
    'Coverage_Type': '📊 Stacked Bar',
    'Speed_Distribution': '📊 Histogram',
    'PlayAction_Impact': '🔻 Funnel Chart'
}

KPI_DESCRIPTIONS = {
    'QB_Pressure': 'Quarterback pressure performance by field position with speed/acceleration metrics',
    'Route_Efficiency': 'Receiver route efficiency: depth, lateral movement, speed optimization',
    'Coverage_Heat': 'Defensive coverage density on NFL field (120x53.3 yards) - hotspot visualization',
    'Pass_Timing': 'Pass timing windows: snap to release with pressure indicators',
    'Separation': 'Real-time receiver separation in 3D: X, Y position + separation distance',
    'Formation_Tendency': 'Offensive formation distribution and success patterns',
    'Win_Probability': 'Win probability impact analysis by play situation',
    'Defense_Reaction': 'Defensive reaction time distribution by position',
    'RedZone_Success': 'Red zone conversion efficiency by yard line (0-20 yards)',
    'Tempo_Analysis': 'Offensive tempo and pace impact on success rate',
    'Movement_Heat': 'Player movement intensity heatmap with speed/acceleration overlay',
    'Pass_Results': 'Complete/Incomplete/Interception breakdown analysis',
    'EP_Analysis': 'Expected Points Added (EPA) statistical breakdown',
    'Coverage_Type': 'Man vs Zone coverage effectiveness comparison',
    'Speed_Distribution': 'Player speed distribution analysis by range',
    'PlayAction_Impact': 'Play action vs standard dropback success comparison'
}

# =======================================================
# ⚙️ PAGE CONFIGURATION
# =======================================================
st.set_page_config(
    page_title="NFL Big Data Bowl 2026", 
    layout="wide", 
    page_icon="🏈",
    initial_sidebar_state="expanded"
)

# =======================================================
# 🎨 CUSTOM CSS
# =======================================================
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
.strategic-kpi-header {{
    background: linear-gradient(90deg, rgba(193,18,31,0.3), transparent);
    border-left: 4px solid {COLOR_GOLD};
    padding: 10px 15px;
    margin: 10px 0;
    border-radius: 4px;
}}
.field-marker {{
    background: rgba(50, 205, 50, 0.1);
    border-left: 3px solid #32CD32;
    padding: 8px 12px;
    margin: 5px 0;
    border-radius: 4px;
}}
</style>
""", unsafe_allow_html=True)

# =======================================================
# 🏈 HEADER
# =======================================================
st.title("🏈 NFL Big Data Bowl 2026 - Advanced Analytics")
st.markdown("_Professional-grade analytics with 16 strategic KPIs + NFL field visualizations_")
st.markdown("---")

# =======================================================
# 📊 INITIALIZE SESSION STATE
# =======================================================
if 'full_df' not in st.session_state:
    st.session_state.full_df = pd.DataFrame()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'strategic_kpis' not in st.session_state:
    st.session_state.strategic_kpis = {}
if 'strategic_kpis_calculated' not in st.session_state:
    st.session_state.strategic_kpis_calculated = False

# =======================================================
# 📊 SIDEBAR CONTROLS
# =======================================================
st.sidebar.header("⚙️ Dashboard Controls")
st.sidebar.subheader("📥 Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Kaggle API", "Upload CSV", "Local Directory"],
    help="Load data from Kaggle competition, upload files, or use local directory"
)

# =======================================================
# 🔐 KAGGLE API OPTION
# =======================================================
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
    competition_name = st.sidebar.text_input("Competition Name", value="nfl-big-data-bowl-2026-analytics")
    
    if st.sidebar.button("🚀 Load Data from Kaggle", type="primary"):
        if kaggle_username and kaggle_key:
            try:
                with st.spinner("🏈 Downloading NFL data from Kaggle..."):
                    df_result = load_data_from_kaggle(kaggle_username, kaggle_key, competition_name)
                    
                    if df_result is not None and not df_result.empty:
                        st.session_state.full_df = df_result
                        st.session_state.data_loaded = True
                        st.session_state.strategic_kpis_calculated = False
                        st.success(f"✅ Successfully loaded {len(st.session_state.full_df):,} rows!")
                    else:
                        st.error("❌ No data was loaded from Kaggle")
                        st.session_state.data_loaded = False
                        
            except Exception as e:
                st.error(f"❌ Kaggle API Error: {str(e)}")
                st.session_state.data_loaded = False
        else:
            st.sidebar.warning("⚠️ Please enter both username and API key")

# =======================================================
# 📤 UPLOAD CSV OPTION
# =======================================================
elif data_source == "Upload CSV":
    uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)
    
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
                    st.session_state.strategic_kpis_calculated = False
                    st.sidebar.success(f"✅ Total: {len(st.session_state.full_df):,} rows!")
                else:
                    st.sidebar.error("❌ No files could be loaded")
        else:
            st.sidebar.warning("⚠️ Please upload at least one CSV file")

# =======================================================
# 📂 LOCAL DIRECTORY OPTION
# =======================================================
else:
    data_dir = st.sidebar.text_input("Data Directory Path", value="./data")
    
    if st.sidebar.button("🚀 Load from Local Directory", type="primary"):
        try:
            with st.spinner(f"📂 Loading CSV files from {data_dir}..."):
                df_result = load_local_data(data_dir)
                if df_result is not None and not df_result.empty:
                    st.session_state.full_df = df_result
                    st.session_state.data_loaded = True
                    st.session_state.strategic_kpis_calculated = False
                    st.success(f"✅ Loaded {len(st.session_state.full_df):,} rows")
                else:
                    st.error("❌ No data loaded")
                    st.session_state.data_loaded = False
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.data_loaded = False

# =======================================================
# 📈 VISUALIZATION CONTROLS
# =======================================================
if st.session_state.data_loaded and not st.session_state.full_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analysis Settings")
    
    kpi_type = st.sidebar.radio(
        "KPI Analysis Type",
        ["Basic KPIs (Quick)", "Strategic KPIs (16 Advanced)"],
        help="Choose between basic metrics or 16 advanced strategic KPIs"
    )
    
    chart_type = st.sidebar.selectbox("Chart Type", ["Bar", "Line", "Area", "Scatter"])
    show_top_n = st.sidebar.slider("Show Top N KPIs", min_value=5, max_value=20, value=12)
    show_data_info = st.sidebar.checkbox("Show Data Information", value=True)
    
    if kpi_type == "Strategic KPIs (16 Advanced)":
        if st.sidebar.button("🧠 Calculate 16 Strategic KPIs", type="primary"):
            st.session_state.strategic_kpis = calculate_all_strategic_kpis(st.session_state.full_df)
            st.session_state.strategic_kpis_calculated = True

# =======================================================
# 🎯 MAIN DASHBOARD CONTENT
# =======================================================
if st.session_state.data_loaded and not st.session_state.full_df.empty:
    df = st.session_state.full_df
    
    # =======================================================
    # DATA SUMMARY
    # =======================================================
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
        
        with st.expander("📊 Available Data Columns"):
            try:
                available_cols = detect_available_columns(df)
                if available_cols:
                    for category, cols in available_cols.items():
                        st.markdown(f"**{category}:** {', '.join(cols)}")
                else:
                    col_list = df.columns.tolist()
                    st.write(f"Total: {len(col_list)} columns")
                    st.write(", ".join(col_list[:30]))
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.markdown("---")
    
    # =======================================================
    # BASIC KPIs
    # =======================================================
    if kpi_type == "Basic KPIs (Quick)":
        try:
            with st.spinner("🧮 Computing Basic KPIs..."):
                kpis = compute_all_kpis(df)
        except Exception as e:
            st.error(f"❌ Error computing KPIs: {str(e)}")
            kpis = {}
        
        if kpis:
            st.markdown("## 📊 Basic Key Performance Indicators")
            
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
            st.markdown("## 📈 KPI Visualization")
            
            df_kpi = pd.DataFrame([{"KPI": k, "Value": v} for k, v in sorted_kpis.items() if not np.isnan(v)])
            
            if not df_kpi.empty:
                if chart_type == "Bar":
                    fig = px.bar(df_kpi, x="KPI", y="Value", color="Value", color_continuous_scale="Viridis", template="plotly_dark")
                elif chart_type == "Line":
                    fig = px.line(df_kpi, x="KPI", y="Value", markers=True, template="plotly_dark")
                elif chart_type == "Area":
                    fig = px.area(df_kpi, x="KPI", y="Value", template="plotly_dark")
                else:
                    fig = px.scatter(df_kpi, x="KPI", y="Value", size="Value", color="Value", color_continuous_scale="Plasma", template="plotly_dark")
                
                fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_PANEL, font_color=TEXT_COLOR, height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    # =======================================================
    # STRATEGIC KPIs (16 ADVANCED)
    # =======================================================
    else:
        st.markdown("## 🧠 Strategic Football Intelligence KPIs")
        
        if st.session_state.strategic_kpis_calculated and st.session_state.strategic_kpis:
            st.success(f"✅ {len(st.session_state.strategic_kpis)} Strategic KPIs calculated successfully!")
            
            # NFL Field Reference
            with st.expander("🏈 NFL Field Reference Guide"):
                st.markdown("""
                <div class="field-marker">
                <strong>NFL Field Dimensions:</strong>
                <ul>
                <li>Length: 120 yards (0 = Own Goal Line, 100 = Opponent Goal Line, 100-120 = End Zone)</li>
                <li>Width: 53.3 yards (sideline to sideline)</li>
                <li>Red Zone: 20 yards from goal line (yards 100-120)</li>
                <li>Hash Marks: 18.5 feet apart (centered on field)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Create tabs for better organization
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔥 Performance Analytics (1-4)", 
                "🎯 Tactical Analysis (5-8)", 
                "📊 Efficiency Metrics (9-12)", 
                "⚡ Advanced Stats (13-16)"
            ])
            
            kpi_names = list(st.session_state.strategic_kpis.keys())
            
            # Helper function to render KPI
            def render_kpi(kpi_name, kpi_data, col, key_suffix):
                with col:
                    st.markdown(f"""
                    <div class="strategic-kpi-header">
                        <h3>{CHART_RECOMMENDATIONS.get(kpi_name, '📊')} {kpi_name.replace('_', ' ').title()}</h3>
                        <p style="margin:0; font-size:0.9em; color: {COLOR_SILVER};">{KPI_DESCRIPTIONS.get(kpi_name, '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if isinstance(kpi_data, pd.DataFrame) and not kpi_data.empty:
                        with st.expander("📊 View Raw Data", expanded=False):
                            st.dataframe(kpi_data.head(20), use_container_width=True)
                        
                        try:
                            fig = visualize_kpi(kpi_name, kpi_data)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_{kpi_name}_{key_suffix}")
                            else:
                                st.info("💡 Visualization not available for this data structure")
                        except Exception as e:
                            if debug_mode:
                                st.error(f"Chart error: {e}")
                            # Fallback simple chart
                            if len(kpi_data.columns) >= 2:
                                fig = px.bar(kpi_data.head(10), x=kpi_data.columns[0], y=kpi_data.columns[1], template="plotly_dark")
                                fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_PANEL, font_color=TEXT_COLOR, height=350)
                                st.plotly_chart(fig, use_container_width=True, key=f"fallback_{kpi_name}_{key_suffix}")
                    else:
                        st.info("⚠️ No data available for this KPI")
            
            # TAB 1: KPIs 1-4
            with tab1:
                for i in range(0, min(4, len(kpi_names)), 2):
                    col1, col2 = st.columns(2)
                    
                    if i < len(kpi_names):
                        render_kpi(kpi_names[i], st.session_state.strategic_kpis[kpi_names[i]], col1, "tab1_1")
                    
                    if i + 1 < len(kpi_names):
                        render_kpi(kpi_names[i + 1], st.session_state.strategic_kpis[kpi_names[i + 1]], col2, "tab1_2")
            
            # TAB 2: KPIs 5-8
            with tab2:
                for i in range(4, min(8, len(kpi_names)), 2):
                    col1, col2 = st.columns(2)
                    
                    if i < len(kpi_names):
                        render_kpi(kpi_names[i], st.session_state.strategic_kpis[kpi_names[i]], col1, "tab2_1")
                    
                    if i + 1 < len(kpi_names):
                        render_kpi(kpi_names[i + 1], st.session_state.strategic_kpis[kpi_names[i + 1]], col2, "tab2_2")
            
            # TAB 3: KPIs 9-12
            with tab3:
                for i in range(8, min(12, len(kpi_names)), 2):
                    col1, col2 = st.columns(2)
                    
                    if i < len(kpi_names):
                        render_kpi(kpi_names[i], st.session_state.strategic_kpis[kpi_names[i]], col1, "tab3_1")
                    
                    if i + 1 < len(kpi_names):
                        render_kpi(kpi_names[i + 1], st.session_state.strategic_kpis[kpi_names[i + 1]], col2, "tab3_2")
            
            # TAB 4: KPIs 13-16
            with tab4:
                for i in range(12, len(kpi_names), 2):
                    col1, col2 = st.columns(2)
                    
                    if i < len(kpi_names):
                        render_kpi(kpi_names[i], st.session_state.strategic_kpis[kpi_names[i]], col1, "tab4_1")
                    
                    if i + 1 < len(kpi_names):
                        render_kpi(kpi_names[i + 1], st.session_state.strategic_kpis[kpi_names[i + 1]], col2, "tab4_2")
        
        else:
            st.info("👈 Click 'Calculate 16 Strategic KPIs' in the sidebar to start advanced analysis")
            
            # Show preview cards
            st.markdown("### 🎯 Strategic KPIs Preview")
            st.markdown("Click the button above to generate real-time analytics with interactive NFL field visualizations")
            
            kpi_preview = [
                ("🔥 QB Pressure", "Heatmap/Box", "Pressure performance by field position"),
                ("🕸️ Route Efficiency", "Radar", "Route depth & lateral movement"),
                ("🔥 Coverage Heat", "NFL Field", "120x53.3 yard coverage heatmap"),
                ("📊 Pass Timing", "Step Chart", "Snap to release timing windows"),
                ("🫧 Separation", "3D Scatter", "3D receiver separation analysis"),
                ("☀️ Formation", "Sunburst", "Formation distribution patterns"),
                ("💧 Win Probability", "Waterfall", "WP impact by situation"),
                ("🎻 Defense Reaction", "Violin", "Reaction time by position"),
                ("⏱️ Red Zone", "Gauge", "Success rate by yard line"),
                ("📈 Tempo", "Bar Chart", "Pace impact analysis"),
                ("🔥 Movement", "NFL Field", "Movement density heatmap"),
                ("🍩 Pass Results", "Doughnut", "Complete/Incomplete breakdown"),
                ("💧 EPA", "Waterfall", "Expected Points Added"),
                ("📊 Coverage Type", "Stacked", "Man vs Zone effectiveness"),
                ("📊 Speed Dist", "Histogram", "Speed range distribution"),
                ("🔻 Play Action", "Funnel", "PA vs standard comparison")
            ]
            
            for i in range(0, len(kpi_preview), 4):
                cols = st.columns(4)
                for j, (emoji_name, chart_type, desc) in enumerate(kpi_preview[i:i+4]):
                    with cols[j]:
                        st.markdown(f"""
                        <div class="kpi-card" style="min-height: 150px;">
                            <div class="kpi-title">{emoji_name}</div>
                            <div style="color: {COLOR_GOLD}; font-size: 14px; margin: 8px 0;">{chart_type}</div>
                            <div class="kpi-sub">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # =======================================================
    # DATA EXPLORER
    # =======================================================
    st.markdown("---")
    st.markdown("## 📋 Data Explorer")
    
    with st.expander("🔍 View Sample Data"):
        try:
            st.dataframe(df.head(100), use_container_width=True, height=400)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with st.expander("📊 Column Statistics"):
        try:
            col_info = get_column_info(df)
            st.dataframe(col_info, use_container_width=True, height=400)
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    # =======================================================
    # WELCOME SCREEN
    # =======================================================
    st.info("👈 **Get Started:** Use the sidebar to load your NFL data")
    
    st.markdown("""
    ### 🏈 Welcome to NFL Big Data Bowl 2026 - Professional Analytics Dashboard
    
    #### 📥 How to Load Data:
    
    **Option 1: Kaggle API** (Recommended)
    - Enter your Kaggle username and API key
    - Competition: `nfl-big-data-bowl-2026-analytics`
    - Click "Load Data from Kaggle"
    
    **Option 2: Upload CSV**
    - Upload your CSV files from the competition
    
    **Option 3: Local Directory**
    - Specify path to your data folder
    
    #### 📊 Features:
    
    **🎯 Basic KPIs (14+ Metrics)**
    - Quick performance overview
    - Speed, acceleration, position metrics
    - Expected points analysis
    - Player and play statistics
    
    **🧠 Strategic KPIs (16 Advanced Metrics)**
    1. **QB Pressure Analysis** - Pressure performance by field position
    2. **Route Efficiency** - Route depth, lateral movement optimization
    3. **Coverage Heat** - NFL field heatmap (120x53.3 yards)
    4. **Pass Timing** - Snap to release with pressure indicators
    5. **Player Separation** - Real-time 3D separation analysis
    6. **Formation Tendencies** - Formation distribution patterns
    7. **Win Probability** - WP impact by play situation
    8. **Defense Reaction** - Reaction time by position
    9. **Red Zone Success** - Conversion efficiency by yard line
    10. **Tempo Analysis** - Offensive pace impact
    11. **Movement Heatmap** - Movement density with speed overlay
    12. **Pass Results** - Complete/Incomplete/INT breakdown
    13. **EPA Analysis** - Expected Points Added statistics
    14. **Coverage Type** - Man vs Zone effectiveness
    15. **Speed Distribution** - Player speed range analysis
    16. **Play Action Impact** - PA vs standard dropback comparison
    
    #### 🎨 Visualization Types:
    - **NFL Field Heatmaps** - 120x53.3 yard field with hotspot analysis
    - **3D Scatter Plots** - Separation analysis with X, Y, Z dimensions
    - **2D Scatter Plots** - Alternative field position visualization
    - **Radar Charts** - Route efficiency metrics
    - **Gauge Charts** - Red zone success rates
    - **Violin Plots** - Reaction time distributions
    - **Waterfall Charts** - Win probability & EPA
    - **Sunburst Charts** - Formation hierarchies
    - **And many more...**
    
    #### 🚀 Getting Started:
    1. Load your data using the sidebar
    2. Choose "Basic KPIs" for quick overview
    3. Select "Strategic KPIs" for deep dive analysis
    4. Explore interactive visualizations
    5. View raw data in expandable sections
    
    #### 💡 Pro Tips:
    - Enable debug mode to troubleshoot issues
    - Use NFL Field Reference Guide for dimension context
    - Hover over charts for detailed information
    - Download charts using Plotly's built-in tools
    """)

# =======================================================
# FOOTER
# =======================================================
st.sidebar.markdown("---")
st.sidebar.markdown("**NFL Big Data Bowl 2026**")
st.sidebar.caption("v4.0 - Professional Analytics with NFL Field Visualizations")
st.sidebar.caption("Optimized for data analysts & football strategists")
