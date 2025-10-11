# =======================================================
# 🏈 NFL Analytics Dashboard - Streamlit App
# =======================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import load_data, compute_all_kpis_and_aggregate

# =======================================================
# ⚙️ Configuration de la page
# =======================================================
st.set_page_config(
    page_title="NFL Big Data Bowl 2026 Analytics",
    layout="wide",
    page_icon="🏈"
)

st.title("🏈 NFL Big Data Bowl 2026 - Analytics Dashboard")
st.markdown("""
Ce **tableau de bord interactif** permet d'explorer les données du NFL Big Data Bowl 2026, 
d'analyser les performances à travers des **KPIs clés** et de visualiser les **tendances par joueur et défenseur**.
""")

# =======================================================
# 📂 Chargement des données
# =======================================================
with st.sidebar:
    st.header("⚙️ Paramètres de chargement")
    st.markdown("**Source des données** : dossier contenant les fichiers CSV extraits de Kaggle.")
    base_dir = st.text_input("Chemin du dossier de données :", "./data")
    load_btn = st.button("📥 Charger les données")

if load_btn:
    try:
        full_df, df_input, df_out, df_supp, tr_sample, prethrow = load_data(base_dir=base_dir)
        st.session_state["data_loaded"] = True
        st.success("✅ Données chargées avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.stop()
else:
    st.info("👉 Charge les données via le menu latéral pour continuer.")
    st.stop()

# =======================================================
# 📊 Calcul et affichage des KPIs globaux
# =======================================================
st.subheader("📈 Indicateurs de performance (KPIs)")

kpis, df_pass_kpis = compute_all_kpis_and_aggregate(full_df, df_input, df_out, df_supp, tr_sample, prethrow)

cols = st.columns(4)
for i, (k, v) in enumerate(kpis.items()):
    with cols[i % 4]:
        st.metric(label=k, value=f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else "—")

# =======================================================
# 🎯 Analyse des KPIs de passes
# =======================================================
if not df_pass_kpis.empty:
    st.subheader("🎯 Analyse des passes (Pass KPIs)")
    
    # Sélecteurs interactifs
    col1, col2 = st.columns(2)
    player_selected = col1.selectbox("Sélectionner un Quarterback (QB)", df_pass_kpis["player_id"].unique())
    defender_selected = col2.selectbox("Sélectionner un Défenseur", df_pass_kpis["defender_id"].unique())

    filtered_df = df_pass_kpis[
        (df_pass_kpis["player_id"] == player_selected) &
        (df_pass_kpis["defender_id"] == defender_selected)
    ]

    # Graphique scatter
    fig = px.scatter(
        filtered_df,
        x="cli_final",
        y="ADR",
        size="sm_max",
        color="ME",
        hover_data=["max_dai", "cci_n_def_in_R"],
        title=f"Relations entre KPIs de passes ({player_selected} vs {defender_selected})"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tableau détaillé
    st.dataframe(filtered_df.head(20), use_container_width=True)
else:
    st.warning("Aucun KPI de passe disponible dans les données chargées.")

# =======================================================
# 📜 Pied de page
# =======================================================
st.markdown("---")
st.caption("Développé pour l’analyse du NFL Big Data Bowl 2026 • Exécution sur Streamlit Cloud / Colab")
