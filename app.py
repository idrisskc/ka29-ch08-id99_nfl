# =======================================================
# app.py
# 🏈 NFL Big Data Bowl 2026 - Analytics Dashboard (combined + Kaggle)
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

# Import utils (must exist in repo)
# utils.py should provide: load_data(base_dir="./data"), compute_all_kpis_and_aggregate(...)
from utils import load_data, compute_all_kpis_and_aggregate
try:
    from utils import download_from_kaggle as utils_download_from_kaggle
except Exception:
    utils_download_from_kaggle = None

# ---------------------
# Theme / Colors (NFL palette)
# ---------------------
COLOR_BG = "#0B0C10"        # deep black
COLOR_PANEL = "#1B263B"     # navy
COLOR_ACCENT = "#C1121F"    # red
COLOR_GOLD = "#FFD700"      # gold
COLOR_SILVER = "#A9A9A9"    # silver
TEXT_COLOR = "#E6EEF8"

# Page config
st.set_page_config(page_title="NFL Big Data Bowl 2026", layout="wide", page_icon="🏈")
st.title("🏈 NFL Big Data Bowl 2026 - Analytics Dashboard")
st.markdown("Explore, visualise, and understand **NFL tracking data** from the 2026 Big Data Bowl.")

# ---------------------
# CSS - dark theme + KPI card styles
# ---------------------
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
    .stApp {{ background: var(--bg); color: var(--text); }}
    /* Sidebar */
    .stSidebar {{ background: linear-gradient(180deg, var(--panel), rgba(11,12,16,0.95)); padding: 1rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.03); }}
    /* KPI card */
    .kpi-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(169,169,169,0.12);
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
        color: var(--text);
    }}
    .kpi-title {{ color: var(--silver); font-size:13px; margin-bottom:6px; }}
    .kpi-value {{ font-size:26px; font-weight:700; }}
    .kpi-sub {{ color: rgba(255,255,255,0.6); font-size:12px; margin-top:6px; }}
    .kpi-badge-up {{ background: rgba(255,215,0,0.12); color: var(--gold); padding:4px 8px; border-radius:10px; border:1px solid rgba(255,215,0,0.12); font-weight:600; font-size:12px; }}
    .kpi-badge-down {{ background: rgba(193,18,31,0.08); color: var(--accent); padding:4px 8px; border-radius:10px; border:1px solid rgba(193,18,31,0.12); font-weight:600; font-size:12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------
# Helper: render KPI card HTML
# ---------------------
def render_kpi_card(title: str, value, subtitle: str = "", delta: float = None):
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

# ---------------------
# Sidebar: controls & Kaggle credentials
# ---------------------
st.sidebar.header("Settings")

# Date range controls
today = datetime.today().date()
start_default = today.replace(year=today.year - 1)
start_date = st.sidebar.date_input("Start date", start_default)
end_date = st.sidebar.date_input("End date", today)

# Timeframe and chart type
timeframe = st.sidebar.selectbox("Select time frame", ["Daily", "Weekly", "Monthly"])
chart_type = st.sidebar.selectbox("Select a chart type", ["Bar", "Line", "Area"])

st.sidebar.markdown("---")
st.sidebar.markdown("Data source")

data_mode = st.sidebar.radio("Source:", ["Kaggle (download)", "Local folder (./data)"])
base_dir = "./data"

# Provide Kaggle creds: first check st.secrets, then fields
secrets_user = st.secrets.get("KAGGLE_USERNAME", "") if st.secrets else ""
secrets_key = st.secrets.get("KAGGLE_KEY", "") if st.secrets else ""

kaggle_user = st.sidebar.text_input("Kaggle username", value=secrets_user)
kaggle_key = st.sidebar.text_input("Kaggle key", value=secrets_key, type="password")

# Button to download from Kaggle (only when user chooses)
if data_mode == "Kaggle (download)":
    st.sidebar.markdown("⚠️ Please do NOT commit Kaggle keys to a public repo. Use Streamlit secrets.")
    if st.sidebar.button("📥 Download dataset from Kaggle"):
        if not kaggle_user or not kaggle_key:
            st.sidebar.error("Enter Kaggle username & key (or configure in Streamlit Secrets).")
        else:
            # Prefer utils helper if provided
            try:
                if utils_download_from_kaggle is not None:
                    utils_download_from_kaggle(kaggle_user, kaggle_key, base_dir=base_dir)
                    st.sidebar.success("Download + extraction complete (utils helper).")
                else:
                    # fallback: use kaggle CLI via subprocess
                    os.environ["KAGGLE_USERNAME"] = kaggle_user
                    os.environ["KAGGLE_KEY"] = kaggle_key
                    kaggle_json_dir = os.path.expanduser("~/.kaggle")
                    os.makedirs(kaggle_json_dir, exist_ok=True)
                    with open(os.path.join(kaggle_json_dir, "kaggle.json"), "w") as f:
                        f.write('{"username":"%s","key":"%s"}' % (kaggle_user, kaggle_key))
                    os.chmod(os.path.join(kaggle_json_dir, "kaggle.json"), 0o600)
                    # make sure data folder exists
                    os.makedirs(base_dir, exist_ok=True)
                    # run kaggle CLI
                    try:
                        subprocess.run(["kaggle", "competitions", "download", "-c", "nfl-big-data-bowl-2026-analytics", "-p", base_dir, "--force"], check=True)
                        # find the zip (common name)
                        zip_candidates = [p for p in Path(base_dir).glob("*.zip")]
                        if zip_candidates:
                            z = zip_candidates[0]
                            with zipfile.ZipFile(z, "r") as zp:
                                zp.extractall(base_dir)
                            st.sidebar.success("Downloaded and extracted dataset to ./data")
                        else:
                            st.sidebar.warning("Download finished but no zip found in ./data to extract.")
                    except subprocess.CalledProcessError as ke:
                        st.sidebar.error(f"Kaggle CLI error: {ke}")
            except Exception as e:
                st.sidebar.error(f"Error during Kaggle download: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("Filters")
team_options = ["All"]
# We'll try to fill team_options later once data is loaded
week_range = st.sidebar.slider("Week range", 1, 18, (1, 18))
st.sidebar.markdown("---")
load_btn = st.sidebar.button("Load & Analyze")

# ---------------------
# Load & analyze on click
# ---------------------
if not load_btn:
    st.info("Use the left panel to download (optional) and then click 'Load & Analyze'.")
    st.stop()

with st.spinner("Loading data and computing KPIs..."):
    # call user's utils.load_data
    try:
        full_df, df_input, df_out, df_supp, tr_sample, prethrow = load_data(base_dir=base_dir)
    except Exception as e:
        st.error(f"Error in load_data(): {e}")
        st.stop()

    # Basic check
    if full_df is None or (isinstance(full_df, pd.DataFrame) and full_df.empty):
        st.warning("No data found in ./data. Use Kaggle downloader or place sample CSVs in ./data/ and try again.")
        st.stop()

    # attempt to populate team options if available
    if 'possession_team' in full_df.columns:
        team_options = ["All"] + sorted(full_df['possession_team'].dropna().unique().tolist())
    # apply filters: week & team & date range if columns exist
    filtered_df = full_df.copy()
    if 'week' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['week'] >= week_range[0]) & (filtered_df['week'] <= week_range[1])]
    # apply team filter if user selected something other than All
    team_filter = st.sidebar.selectbox("Team", options=team_options, index=0)
    if team_filter != "All" and 'possession_team' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['possession_team'] == team_filter]
    # date range by trying common date columns
    date_cols = [c for c in filtered_df.columns if 'date' in c.lower() or 'game_date' in c.lower()]
    if date_cols:
        try:
            dcol = date_cols[0]
            filtered_df[dcol] = pd.to_datetime(filtered_df[dcol], errors='coerce')
            filtered_df = filtered_df[(filtered_df[dcol].dt.date >= start_date) & (filtered_df[dcol].dt.date <= end_date)]
        except Exception:
            pass

    # compute KPIs
    try:
        kpis, df_pass_kpis = compute_all_kpis_and_aggregate(filtered_df, df_input, df_out, df_supp, tr_sample, prethrow)
    except Exception as e:
        st.error(f"Error computing KPIs: {e}")
        st.stop()

# ---------------------
# KPI cards row
# ---------------------
st.markdown("## Selected Duration")
kpi_items = list(kpis.items())
cards_per_row = 4
rows = [kpi_items[i:i + cards_per_row] for i in range(0, len(kpi_items), cards_per_row)]

for row in rows:
    cols = st.columns(len(row), gap="large")
    for c, (k, v) in zip(cols, row):
        with c:
            # example delta (replace with real delta if you have history)
            delta_val = float(np.random.uniform(-8, 8))
            render_kpi_card(k, v, subtitle="Aggregate value", delta=delta_val)

st.markdown("---")

# KPI comparison chart
st.markdown("### KPI Comparison")
numeric_kpis = {k: v for k, v in kpis.items() if isinstance(v, (int, float, np.floating, np.integer)) and not np.isnan(v)}
if numeric_kpis:
    df_kpi = pd.DataFrame(list(numeric_kpis.items()), columns=["KPI", "Value"])
    if chart_type == "Bar":
        fig = px.bar(df_kpi, x="KPI", y="Value", color="KPI", template="plotly_dark", text="Value")
    elif chart_type == "Line":
        fig = px.line(df_kpi, x="KPI", y="Value", markers=True, template="plotly_dark")
    else:  # Area
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_kpi['KPI'], y=df_kpi['Value'], fill='tozeroy', mode='lines+markers'))
        fig.update_layout(template="plotly_dark")
    fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, height=420)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No numeric KPIs to visualize.")

# ---------------------
# Tabs for more analyses
# ---------------------
tab1, tab2, tab3, tab4 = st.tabs(["Pass Analytics", "Defensive Analysis", "Compare Players", "Export"])

with tab1:
    st.header("🎯 Pass Analytics")
    if df_pass_kpis is None or df_pass_kpis.empty:
        st.info("No pass KPIs available for the selected filters.")
    else:
        cols = st.columns(3)
        game_opts = ["All"] + sorted(df_pass_kpis['game_id'].dropna().unique().tolist()) if 'game_id' in df_pass_kpis.columns else ["All"]
        sel_game = cols[0].selectbox("Game", options=game_opts)
        play_opts = ["All"] + sorted(df_pass_kpis['play_id'].dropna().unique().tolist()) if 'play_id' in df_pass_kpis.columns else ["All"]
        sel_play = cols[1].selectbox("Play", options=play_opts)
        sel_metric = cols[2].selectbox("Color metric", options=[c for c in df_pass_kpis.select_dtypes(include=[np.number]).columns if c not in ['game_id','play_id','nfl_id']], index=0)

        dpf = df_pass_kpis.copy()
        if sel_game != "All":
            dpf = dpf[dpf['game_id'] == sel_game]
        if sel_play != "All":
            dpf = dpf[dpf['play_id'] == sel_play]

        if 'ADR' in dpf.columns and 'cli_final' in dpf.columns:
            fig = px.scatter(dpf, x='ADR', y='cli_final', color=sel_metric if sel_metric in dpf.columns else None,
                             size='ME' if 'ME' in dpf.columns else None, template="plotly_dark",
                             hover_data=['game_id','play_id','nfl_id'])
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dpf.head(200), use_container_width=True)

with tab2:
    st.header("🛡️ Defensive Analysis")
    if 'nearest_def_dist' in filtered_df.columns:
        fig = px.histogram(filtered_df, x='nearest_def_dist', nbins=40, template="plotly_dark", title="Nearest defender distance")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("nearest_def_dist not available.")
    if df_pass_kpis is not None and 'ADR' in df_pass_kpis.columns:
        fig = px.box(df_pass_kpis, y='ADR', template="plotly_dark", title="ADR distribution")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("👥 Compare Players")
    player_col = 'nfl_id' if 'nfl_id' in filtered_df.columns else None
    if player_col is None:
        st.warning("nfl_id column not found.")
    else:
        players = sorted(filtered_df[player_col].dropna().unique().tolist())
        if len(players) < 2:
            st.info("Not enough players to compare.")
        else:
            p1 = st.selectbox("Player 1", players, index=0)
            p2 = st.selectbox("Player 2", players, index=1)
            def summary(pid):
                sub = filtered_df[filtered_df[player_col] == pid]
                return {
                    'mean_speed': float(sub['s'].mean()) if 's' in sub.columns else np.nan,
                    'max_speed': float(sub['s'].max()) if 's' in sub.columns else np.nan,
                    'mean_acc': float(sub['a'].mean()) if 'a' in sub.columns else np.nan,
                    'n_plays': int(sub['play_id'].nunique()) if 'play_id' in sub.columns else len(sub)
                }
            s1 = summary(p1); s2 = summary(p2)
            left, right = st.columns(2)
            with left:
                st.subheader(f"Player {p1}")
                st.json(s1)
            with right:
                st.subheader(f"Player {p2}")
                st.json(s2)
            labels = ['mean_speed','max_speed','mean_acc','n_plays']
            v1 = [s1.get(l,0) or 0 for l in labels]
            v2 = [s2.get(l,0) or 0 for l in labels]
            arr = np.array([v1, v2], dtype=float)
            maxvals = arr.max(axis=0); maxvals[maxvals==0] = 1.0
            v1n = (arr[0]/maxvals).tolist(); v2n = (arr[1]/maxvals).tolist()
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=v1n + [v1n[0]], theta=labels + [labels[0]], fill='toself', name=str(p1)))
            fig.add_trace(go.Scatterpolar(r=v2n + [v2n[0]], theta=labels + [labels[0]], fill='toself', name=str(p2)))
            fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("💾 Export")
    if df_pass_kpis is None or df_pass_kpis.empty:
        st.info("No pass KPIs to export.")
    else:
        csv = df_pass_kpis.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Pass KPIs (CSV)", data=csv, file_name="pass_kpis.csv", mime="text/csv")
        # Excel safe write to buffer
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_pass_kpis.to_excel(writer, index=False, sheet_name="pass_kpis")
            writer.save()
        st.download_button("⬇️ Download Pass KPIs (XLSX)", data=buffer.getvalue(), file_name="pass_kpis.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Dark theme • NFL palette • Kaggle download integrated (use Streamlit secrets for credentials)")
