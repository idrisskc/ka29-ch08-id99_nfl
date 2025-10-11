# =======================================================
# app.py - NFL Analytics Dashboard (Advanced, Dark theme, styled)
# =======================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime

# Import utils (assumed present in the repo)
# utils must provide:
#   - load_data(base_dir="./data")
#   - compute_all_kpis_and_aggregate(full_df, df_input, df_out, df_supp, tr_sample, prethrow)
#   - download_from_kaggle(username, key, base_dir="./data")  (optional)
from utils import load_data, compute_all_kpis_and_aggregate
try:
    from utils import download_from_kaggle
except Exception:
    download_from_kaggle = None

# =======================================================
# NFL Palette (colors)
# =======================================================
COLOR_BG = "#0B0C10"        # black deep for background
COLOR_PANEL = "#1B263B"     # navy for panels
COLOR_ACCENT = "#C1121F"    # red (accent/badges)
COLOR_GOLD = "#FFD700"      # gold for highlights
COLOR_SILVER = "#A9A9A9"    # silver for borders / secondary text
TEXT_COLOR = "#E6EEF8"      # light text color

# =======================================================
# Page config
# =======================================================
st.set_page_config(page_title="NFL Analytics Dashboard", layout="wide", page_icon="🏈")

# =======================================================
# Global CSS for dark theme + card styles + sidebar
# =======================================================
st.markdown(
    f"""
    <style>
    :root {{
        --bg: {COLOR_BG};
        --panel: {COLOR_PANEL};
        --accent: {COLOR_ACCENT};
        --gold: {COLOR_GOLD};
        --silver: {COLOR_SILVER};
        --text: {TEXT_COLOR};
    }}
    /* Page background */
    .stApp {{
        background: var(--bg);
        color: var(--text);
    }}
    /* Sidebar */
    .css-1d391kg {{ /* streamlit sidebar class may vary; include a wide selector fallback */ }
    }
    .stSidebar {{
        background: linear-gradient(180deg, var(--panel), rgba(15,17,20,0.95));
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.03);
    }}
    /* KPI card */
    .kpi-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(169,169,169,0.12);
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        color: var(--text);
    }}
    .kpi-title {{
        color: var(--silver);
        font-size: 13px;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-size: 28px;
        font-weight: 700;
        color: var(--text);
    }}
    .kpi-sub {{
        font-size: 12px;
        color: rgba(255,255,255,0.6);
        margin-top: 6px;
    }}
    /* small badge */
    .kpi-badge-up {{
        background: linear-gradient(90deg, rgba(255,215,0,0.12), rgba(255,215,0,0.06));
        color: var(--gold);
        padding: 4px 8px;
        border-radius: 10px;
        font-weight:600;
        border: 1px solid rgba(255,215,0,0.12);
        margin-left: 8px;
        font-size: 12px;
    }}
    .kpi-badge-down {{
        background: linear-gradient(90deg, rgba(193,18,31,0.08), rgba(193,18,31,0.04));
        color: var(--accent);
        padding: 4px 8px;
        border-radius: 10px;
        font-weight:600;
        border: 1px solid rgba(193,18,31,0.12);
        margin-left: 8px;
        font-size: 12px;
    }}

    /* style for plotly graphs in dark */
    .js-plotly-plot .plotly .modebar { background: rgba(0,0,0,0.2) !important; }

    /* adjust streamlit headers color */
    .css-1v3fvcr h1, .css-1v3fvcr h2, .css-1v3fvcr h3 {{
        color: var(--text);
    }}

    /* Dataframe styling */
    .stDataFrame div[role="table"] {{
        border-radius: 8px;
        overflow: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =======================================================
# Helper: small KPI card renderer (HTML)
# =======================================================
def render_kpi_card(title: str, value, subtitle: str = "", delta: float = None):
    """
    Renders a small KPI card with optional delta (positive => gold badge, negative => red badge)
    """
    delta_html = ""
    if delta is not None and not (np.isnan(delta)):
        if delta >= 0:
            delta_html = f'<span class="kpi-badge-up">+{delta:.1f}%</span>'
        else:
            delta_html = f'<span class="kpi-badge-down">{delta:.1f}%</span>'
    value_str = f"{value:.2f}" if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(value) else str(value)
    html = f"""
    <div class="kpi-card">
      <div class="kpi-title">{title} {delta_html}</div>
      <div class="kpi-value">{value_str}</div>
      <div class="kpi-sub">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# =======================================================
# Sidebar: filters & data controls
# =======================================================
st.sidebar.markdown("<h3 style='color:var(--text)'>Settings</h3>", unsafe_allow_html=True)
# Date range
today = datetime.today().date()
start_default = today.replace(year=today.year - 1)
start_date = st.sidebar.date_input("Start date", start_default)
end_date = st.sidebar.date_input("End date", today)
# timeframe selector
timeframe = st.sidebar.selectbox("Select time frame", ["Daily", "Weekly", "Monthly"])
# chart type
chart_type = st.sidebar.selectbox("Select a chart type", ["Bar", "Line", "Area"])
st.sidebar.markdown("---")

# Kaggle download controls (use st.secrets if present)
use_kaggle = st.sidebar.checkbox("Download from Kaggle (requires secrets)", value=False)
if use_kaggle:
    # prefer secrets if set
    kaggle_user = st.secrets.get("KAGGLE_USERNAME", os.getenv("KAGGLE_USERNAME", ""))
    kaggle_key = st.secrets.get("KAGGLE_KEY", os.getenv("KAGGLE_KEY", ""))
    kaggle_user_input = st.sidebar.text_input("Kaggle username", value=kaggle_user)
    kaggle_key_input = st.sidebar.text_input("Kaggle key", value=kaggle_key, type="password")
    if st.sidebar.button("📥 Download & extract Kaggle dataset"):
        if not kaggle_user_input or not kaggle_key_input:
            st.sidebar.error("Provide Kaggle credentials in secrets or the fields above.")
        else:
            if download_from_kaggle is None:
                st.sidebar.error("download_from_kaggle not implemented in utils.py")
            else:
                with st.spinner("Downloading from Kaggle..."):
                    try:
                        download_from_kaggle(kaggle_user_input, kaggle_key_input, base_dir="./data")
                        st.sidebar.success("Downloaded & extracted to ./data")
                    except Exception as e:
                        st.sidebar.error(f"Error during Kaggle download: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("Filters for the displayed data:")
team_filter = st.sidebar.selectbox("Team", ["All"], index=0)
week_min, week_max = st.sidebar.slider("Week range", 1, 18, (1, 18))
st.sidebar.markdown("---")
load_btn = st.sidebar.button("Load & Analyze")

# =======================================================
# Main: load data and compute KPIs (on press)
# =======================================================
if not load_btn:
    st.markdown("<div style='color:var(--silver)'>Use the left panel to load data (local or Kaggle) and apply filters.</div>", unsafe_allow_html=True)
    st.stop()

# Attempt to load data
with st.spinner("Loading data and computing KPIs..."):
    try:
        full_df, df_input, df_out, df_supp, tr_sample, prethrow = load_data(base_dir="./data")
    except Exception as e:
        st.error(f"Error in load_data(): {e}")
        st.stop()

    # Basic presence check
    if full_df is None or (isinstance(full_df, pd.DataFrame) and full_df.empty):
        st.warning("No data found in ./data. Use the Kaggle downloader or upload a small sample into the data/ folder.")
        st.stop()

    # Apply basic filters: week range, team if available, date range if game_date exists
    filtered_df = full_df.copy()
    if 'week' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['week'] >= week_min) & (filtered_df['week'] <= week_max)]
    if team_filter != "All" and 'possession_team' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['possession_team'] == team_filter]
    # if date columns exist, apply date filter (try multiple column names)
    date_cols = [c for c in filtered_df.columns if 'date' in c.lower() or 'game_date' in c.lower()]
    if date_cols:
        # try to coerce first candidate
        try:
            filtered_df[date_cols[0]] = pd.to_datetime(filtered_df[date_cols[0]])
            filtered_df = filtered_df[(filtered_df[date_cols[0]].dt.date >= start_date) & (filtered_df[date_cols[0]].dt.date <= end_date)]
        except Exception:
            pass

    # compute KPIs (main wrapper)
    try:
        kpis, df_pass_kpis = compute_all_kpis_and_aggregate(filtered_df, df_input, df_out, df_supp, tr_sample, prethrow)
    except Exception as e:
        st.error(f"Error computing KPIs: {e}")
        st.stop()

# =======================================================
# Layout: KPI cards row
# =======================================================
st.markdown("## Selected Duration")
# Build a responsive grid of KPI cards (4 per row)
kpi_items = list(kpis.items())
cards_per_row = 4
rows = [kpi_items[i:i + cards_per_row] for i in range(0, len(kpi_items), cards_per_row)]

for row in rows:
    cols = st.columns(len(row), gap="large")
    for c, (k, v) in zip(cols, row):
        with c:
            # For demo, create a random delta (replace with real delta if you have time comparison)
            delta_val = np.random.uniform(-5, 5)
            render_kpi_card(k, v, subtitle="Aggregate value", delta=delta_val)

st.markdown("---")

# =======================================================
# Main visual (KPI comparison)
# =======================================================
st.markdown("### KPI Comparison")
numeric_kpis = {k: v for k, v in kpis.items() if isinstance(v, (int, float, np.floating, np.integer)) and not np.isnan(v)}
if numeric_kpis:
    df_kpi = pd.DataFrame(list(numeric_kpis.items()), columns=["KPI", "Value"])
    color_seq = [COLOR_GOLD if val >= 0 else COLOR_ACCENT for val in df_kpi['Value'].values]
    if chart_type == "Bar":
        fig = px.bar(df_kpi, x="KPI", y="Value", color="KPI", text="Value", template="plotly_dark")
    elif chart_type == "Line":
        fig = px.line(df_kpi, x="KPI", y="Value", markers=True, template="plotly_dark")
    else:  # Area
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_kpi['KPI'], y=df_kpi['Value'], fill='tozeroy', mode='lines+markers'))
        fig.update_layout(template="plotly_dark")
    fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, height=420, title="Aggregated KPI comparison")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No numeric KPIs to visualize.")

# =======================================================
# Tabs: Pass analytics, Defensive, Compare, Export
# =======================================================
tab1, tab2, tab3, tab4 = st.tabs(["Pass Analytics", "Defensive Analysis", "Compare Players", "Export"])

# --- Tab 1: Pass Analytics
with tab1:
    st.header("🎯 Pass Analytics")
    if df_pass_kpis is None or df_pass_kpis.empty:
        st.info("No pass KPIs available for the selected filters.")
    else:
        # Filters
        cols = st.columns(3)
        game_options = ["All"] + sorted(df_pass_kpis['game_id'].dropna().unique().tolist()) if 'game_id' in df_pass_kpis.columns else ["All"]
        sel_game = cols[0].selectbox("Game", options=game_options, index=0)
        play_options = ["All"] + sorted(df_pass_kpis['play_id'].dropna().unique().tolist()) if 'play_id' in df_pass_kpis.columns else ["All"]
        sel_play = cols[1].selectbox("Play", options=play_options, index=0)
        sel_metric = cols[2].selectbox("Color metric", options=[c for c in df_pass_kpis.select_dtypes(include=[np.number]).columns if c not in ['game_id','play_id','nfl_id']] , index=0)

        dpf = df_pass_kpis.copy()
        if sel_game != "All":
            dpf = dpf[dpf['game_id'] == sel_game]
        if sel_play != "All":
            dpf = dpf[dpf['play_id'] == sel_play]

        # Scatter ADR vs CLI
        if 'ADR' in dpf.columns and 'cli_final' in dpf.columns:
            fig = px.scatter(dpf, x='ADR', y='cli_final', color=sel_metric if sel_metric in dpf.columns else None,
                             size='ME' if 'ME' in dpf.columns else None, template="plotly_dark",
                             hover_data=['game_id','play_id','nfl_id'])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dpf.head(200), use_container_width=True)

# --- Tab 2: Defensive Analysis
with tab2:
    st.header("🛡️ Defensive Analysis")
    if 'nearest_def_dist' in filtered_df.columns:
        fig = px.histogram(filtered_df, x='nearest_def_dist', nbins=40, template="plotly_dark", title="Nearest defender distance distribution")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("nearest_def_dist column not available in data.")

    # ADR distribution from pass KPIs if available
    if df_pass_kpis is not None and 'ADR' in df_pass_kpis.columns:
        fig = px.box(df_pass_kpis, y='ADR', template="plotly_dark", title="ADR per play distribution")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Compare Players
with tab3:
    st.header("👥 Compare Players")
    player_col = 'nfl_id' if 'nfl_id' in filtered_df.columns else None
    if player_col is None:
        st.warning("Player ID column (nfl_id) not found in dataset.")
    else:
        players = sorted(filtered_df[player_col].dropna().unique().tolist())
        if len(players) < 2:
            st.info("Not enough distinct players to compare.")
        else:
            p1 = st.selectbox("Player 1", players, index=0)
            p2 = st.selectbox("Player 2", players, index=1 if len(players)>1 else 0)
            def summary_for_player(pid):
                sub = filtered_df[filtered_df[player_col] == pid]
                return {
                    'mean_speed': float(sub['s'].mean()) if 's' in sub.columns else np.nan,
                    'max_speed': float(sub['s'].max()) if 's' in sub.columns else np.nan,
                    'mean_acc': float(sub['a'].mean()) if 'a' in sub.columns else np.nan,
                    'n_plays': int(sub['play_id'].nunique()) if 'play_id' in sub.columns else len(sub)
                }
            s1 = summary_for_player(p1)
            s2 = summary_for_player(p2)

            # Show side-by-side cards
            left, right = st.columns(2)
            with left:
                st.subheader(f"Player {p1}")
                st.json(s1)
            with right:
                st.subheader(f"Player {p2}")
                st.json(s2)

            # Radar comparison
            labels = ['mean_speed','max_speed','mean_acc','n_plays']
            v1 = [s1.get(l,0) or 0 for l in labels]
            v2 = [s2.get(l,0) or 0 for l in labels]
            # normalize to max
            arr = np.array([v1, v2], dtype=float)
            maxvals = arr.max(axis=0)
            maxvals[maxvals==0] = 1.0
            v1n = (arr[0]/maxvals).tolist()
            v2n = (arr[1]/maxvals).tolist()
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=v1n + [v1n[0]], theta=labels + [labels[0]], fill='toself', name=str(p1)))
            fig.add_trace(go.Scatterpolar(r=v2n + [v2n[0]], theta=labels + [labels[0]], fill='toself', name=str(p2)))
            fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Export
with tab4:
    st.header("💾 Export Results")
    if df_pass_kpis is None or df_pass_kpis.empty:
        st.info("No pass KPIs to export.")
    else:
        csv = df_pass_kpis.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Pass KPIs (CSV)", data=csv, file_name="pass_kpis.csv", mime="text/csv")
        # Excel
        towrite = pd.ExcelWriter("pass_kpis.xlsx", engine="xlsxwriter")
        df_pass_kpis.to_excel(towrite, sheet_name="pass_kpis", index=False)
        towrite.save()
        with open("pass_kpis.xlsx", "rb") as f:
            st.download_button("⬇️ Download Pass KPIs (XLSX)", data=f, file_name="pass_kpis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Dark theme • NFL palette • Advanced interactive dashboard")
