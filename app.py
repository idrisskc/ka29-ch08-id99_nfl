# =======================================================
# 🏈 NFL Big Data Bowl 2026 - Analytics Dashboard (Streamlit)
# =======================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from utils import load_data, compute_all_kpis_and_aggregate, download_from_kaggle

# =======================================================
# ⚙️ Configuration de la page Streamlit
# =======================================================
st.set_page_config(page_title="NFL Big Data Bowl 2026", layout="wide", page_icon="🏈")
st.title("🏈 NFL Big Data Bowl 2026 - Analytics Dashboard")
st.markdown("Explore, visualise, and understand **NFL tracking data** from the 2026 Big Data Bowl.")

# =======================================================
# 📦 Paramètres (Kaggle ou Local)
# =======================================================
st.sidebar.header("⚙️ Configuration des données")

data_mode = st.sidebar.radio("Source des données :", ["Télécharger depuis Kaggle", "Charger localement"])
base_dir = "./data"

if data_mode == "Télécharger depuis Kaggle":
    kaggle_username = st.sidebar.text_input("Kaggle Username", os.getenv("KAGGLE_USERNAME", ""))
    kaggle_key = st.sidebar.text_input("Kaggle API Key", os.getenv("KAGGLE_KEY", ""), type="password")
    if st.sidebar.button("📥 Télécharger les données Kaggle"):
        if not kaggle_username or not kaggle_key:
            st.error("Veuillez entrer vos identifiants Kaggle.")
            st.stop()
        download_from_kaggle(kaggle_username, kaggle_key, base_dir)
        st.success("✅ Données téléchargées et extraites avec succès.")
else:
    st.sidebar.info("Assurez-vous d'avoir placé les fichiers dans `./data/` avant de charger.")

# =======================================================
# 📊 Chargement des données et calcul des KPIs
# =======================================================
if st.sidebar.button("🚀 Charger et analyser"):
    try:
        full_df, df_input, df_out, df_supp, tr_sample, prethrow = load_data(base_dir=base_dir)
        kpis, df_pass_kpis = compute_all_kpis_and_aggregate(full_df, df_input, df_out, df_supp, tr_sample, prethrow)
        st.success("✅ Données chargées avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.stop()
else:
    st.info("Clique sur **Charger et analyser** pour lancer l'analyse.")
    st.stop()

# =======================================================
# 📈 Affichage des KPIs
# =======================================================
st.subheader("📈 Indicateurs de performance globaux")
cols = st.columns(4)
for i, (k, v) in enumerate(kpis.items()):
    with cols[i % 4]:
        st.metric(label=k, value=f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else "—")

# =======================================================
# 🎯 Analyse des passes (Pass KPIs)
# =======================================================
if not df_pass_kpis.empty:
    st.subheader("🎯 Analyse des passes")

    col1, col2 = st.columns(2)
    player_selected = col1.selectbox("Sélectionner un Quarterback (QB)", df_pass_kpis["player_id"].unique())
    defender_selected = col2.selectbox("Sélectionner un Défenseur", df_pass_kpis["defender_id"].unique())

    filtered_df = df_pass_kpis[
        (df_pass_kpis["player_id"] == player_selected) &
        (df_pass_kpis["defender_id"] == defender_selected)
    ]

    fig = px.scatter(
        filtered_df,
        x="cli_final", y="ADR",
        size="sm_max", color="ME",
        hover_data=["max_dai", "cci_n_def_in_R"],
        title=f"Relations entre KPIs ({player_selected} vs {defender_selected})"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(filtered_df.head(20), use_container_width=True)
else:
    st.warning("Aucun KPI de passe disponible.")

# =======================================================
# 📜 Pied de page
# =======================================================
st.markdown("---")
st.caption("NFL Big Data Bowl 2026 • Streamlit Dashboard • Données Kaggle automatisées")
