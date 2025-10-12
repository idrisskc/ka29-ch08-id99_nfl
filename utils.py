# =======================================================
# utils.py - Kaggle Data Access & NFL Dashboard KPIs
# =======================================================
import os
import pandas as pd
import numpy as np
import streamlit as st

# =======================================================
# 📦 Lecture directe des CSV depuis Kaggle
# =======================================================
def load_data(username=None, key=None, competition="nfl-big-data-bowl-2026-analytics"):
    """
    Charge directement les fichiers CSV du dataset Kaggle pour visualisation.
    Ne télécharge pas sur le disque si non nécessaire.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        import streamlit as st
        st.error("❌ Module kaggle non installé. Faites `pip install kaggle`")
        st.stop()

    import os
    import pandas as pd
    import streamlit as st
    from io import BytesIO

    # Configuration de l'API Kaggle
    if username and key:
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
    
    api = KaggleApi()
    api.authenticate()

    # Liste des fichiers CSV du dataset
    files = api.competition_list_files(competition)
    csv_files = [f.name for f in files if f.name.endswith(".csv")]

    if not csv_files:
        st.error("❌ Aucun fichier CSV trouvé dans le dataset Kaggle.")
        st.stop()

    st.info(f"📄 {len(csv_files)} fichiers CSV trouvés dans le dataset {competition}")

    dfs = []
    for file_name in csv_files:
        st.info(f"📥 Chargement de {file_name} depuis Kaggle...")
        file_content = api.competition_download_file(competition, file_name, path=None)
        df = pd.read_csv(BytesIO(file_content), low_memory=False)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    st.success(f"✅ {len(csv_files)} fichiers chargés avec succès ({len(full_df):,} lignes totales).")
    return full_df


# =======================================================
# 📈 Calcul des KPIs NFL
# =======================================================
def compute_all_kpis_and_aggregate(full_df):
    """Calcule les indicateurs de performance (KPIs) à partir des données NFL."""
    def safe_mean(df, col):
        return float(df[col].mean()) if col in df.columns else np.nan

    kpis = {
        'PPE (Yards Gained)': safe_mean(full_df, 'yards_gained'),
        'CBR (Completion Prob)': safe_mean(full_df, 'completion_probability'),
        'FFM (Frame ID)': safe_mean(full_df, 'frame_id'),
        'ADY (Distance)': safe_mean(full_df, 'distance'),
        'TDR (Time)': safe_mean(full_df, 'time'),
        'PEI (Event)': safe_mean(full_df, 'event'),
        'CWE (Closest Defender Dist)': safe_mean(full_df, 'closest_defender_distance'),
        'EDS (End Speed)': safe_mean(full_df, 'end_speed'),
        'VMC (Speed)': safe_mean(full_df, 's'),
        'PMA (Play Result)': safe_mean(full_df, 'play_result'),
        'PER (Expected Yards)': safe_mean(full_df, 'expected_yards'),
        'RCI (Receiver ID)': safe_mean(full_df, 'receiver_id'),
        'SMV (Speed Max)': safe_mean(full_df, 'speed_max'),
        'AEF (Acceleration)': safe_mean(full_df, 'acceleration')
    }

    # Données simulées KPI passes
    df_pass_kpis = pd.DataFrame({
        'cli_final': np.random.rand(50),
        'max_dai': np.random.rand(50),
        'ADR': np.random.rand(50),
        'sm_max': np.random.rand(50),
        'ME': np.random.rand(50),
        'cci_n_def_in_R': np.random.rand(50),
        'player_id': np.random.choice(['QB_1', 'QB_2', 'QB_3'], 50),
        'defender_id': np.random.choice(['DEF_1', 'DEF_2', 'DEF_3'], 50)
    })

    return kpis, df_pass_kpis
