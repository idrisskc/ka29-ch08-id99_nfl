# =======================================================
# 🏈 app.py - NFL Big Data Bowl 2026 Dashboard
# =======================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from utils import download_from_kaggle, load_data, compute_all_kpis_and_aggregate

# =======================================================
# 🎨 Configuration du thème Streamlit
# =======================================================
st.set_page_config(
    page_title="NFL Big Data Dashboard",
    page_icon="🏈",
    layout="wide",
)

# Palette NFL
NFL_COLORS = {
    "background": "#0B162A",  # Bleu marine foncé
    "card_bg": "#1A2634",     # Gris bleuté
    "accent": "#C60C30",      # Rouge NFL
    "secondary": "#A5ACAF",   # Argenté
    "highlight": "#FFD700"    # Or
}

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {NFL_COLORS['background']};
        color: white;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    .metric-card {{
        background-color: {NFL_COLORS['card_bg']};
        border: 1px solid {NFL_COLORS['secondary']};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    }}
    .sidebar .sidebar-content {{
        background-color: {NFL_COLORS['card_bg']};
    }}
    </style>
""", unsafe_allow_html=True)

# =======================================================
# ⚙️ Chargement et préparation des données
# =======================================================
base_dir = "./data"
if not os.path.exists(base_dir) or not os.listdir(base_dir):
    st.info("🔄 Téléchargement des données NFL depuis Kaggle...")
    download_from_kaggle(
        os.environ.get("KAGGLE_USERNAME"),
        os.environ.get("KAGGLE_KEY"),
        base_dir=base_dir
    )

full_df, df_input, df_out, df_supp, tr_sample, prethrow = load_data(base_dir)
kpis, df_pass_kpis = compute_all_kpis_and_aggregate(full_df, df_input, df_out, df_supp, tr_sample, prethrow)

# =======================================================
# 🎛️ Barre latérale : filtres
# =======================================================
st.sidebar.header("⚙️ Settings")
years = sorted(df_pass_kpis.index % 3 + 2023)
selected_year = st.sidebar.selectbox("Select Season", options=years, index=0)
chart_type = st.sidebar.selectbox("Select chart type", ["Bar", "Line", "Scatter"])
metric_to_show = st.sidebar.selectbox("Select KPI metric", df_pass_kpis.columns[:-2])

# =======================================================
# 🧮 Section KPI
# =======================================================
st.markdown("### 🏈 NFL Big Data Bowl KPIs Overview")

cols = st.columns(4)
kpi_names = list(kpis.keys())[:4]

for i, col in enumerate(cols):
    key = kpi_names[i]
    value = kpis[key]
    col.markdown(
        f"""
        <div class="metric-card">
            <h4 style='color:{NFL_COLORS["highlight"]};'>{key}</h4>
            <h2 style='color:white;'>{value:.2f}</h2>
        </div>
        """, unsafe_allow_html=True
    )

# =======================================================
# 📊 Section principale : graphique dynamique
# =======================================================
st.markdown("## 📊 Pass Analysis")

if chart_type == "Bar":
    fig = px.bar(df_pass_kpis, x="player_id", y=metric_to_show,
                 color="defender_id", title=f"{metric_to_show} by Player")
elif chart_type == "Line":
    fig = px.line(df_pass_kpis, x=df_pass_kpis.index, y=metric_to_show,
                  color="player_id", title=f"{metric_to_show} Trend")
else:
    fig = px.scatter(df_pass_kpis, x="cli_final", y=metric_to_show,
                     color="player_id", title=f"Scatter of {metric_to_show}")

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor=NFL_COLORS["card_bg"],
    paper_bgcolor=NFL_COLORS["card_bg"],
    font_color="white",
    title_font_size=20
)
st.plotly_chart(fig, use_container_width=True)

# =======================================================
# 📈 Tableau récapitulatif
# =======================================================
st.markdown("### 📋 Pass KPI Table")
st.dataframe(df_pass_kpis.head(15), use_container_width=True)
