# =======================================================
# utils.py - Robust Kaggle + Data Utils for NFL Dashboard
# =======================================================
import os
import pandas as pd
import numpy as np
import zipfile
import subprocess
import shutil
import streamlit as st

# =======================================================
# 📦 Téléchargement et extraction depuis Kaggle
# =======================================================
def download_from_kaggle(username=None, key=None, base_dir="./data"):
    """
    Télécharge et extrait le dataset NFL Big Data Bowl depuis Kaggle.
    Si le répertoire data existe déjà avec des fichiers CSV, il est réutilisé.
    """
    os.makedirs(base_dir, exist_ok=True)

    # Vérifie s’il existe déjà des CSV dans ./data
    existing_csvs = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
    if existing_csvs:
        st.info(f"✅ {len(existing_csvs)} fichiers CSV déjà présents dans {base_dir}. Téléchargement ignoré.")
        return

    if username and key:
        # Configuration des credentials Kaggle
        kaggle_json_path = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_json_path, exist_ok=True)
        kaggle_json_file = os.path.join(kaggle_json_path, "kaggle.json")
        with open(kaggle_json_file, "w") as f:
            f.write(f'{{"username":"{username}","key":"{key}"}}')
        os.chmod(kaggle_json_file, 0o600)
    else:
        st.warning("⚠️ Aucun identifiant Kaggle fourni. Téléchargement ignoré.")
        return

    # Vérifie la disponibilité du module kaggle
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except Exception:
        st.error("❌ Le module Kaggle n'est pas disponible sur ce serveur.")
        st.stop()

    # Téléchargement du dataset
    try:
        st.info("📥 Téléchargement du dataset NFL depuis Kaggle...")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", 
             "nfl-big-data-bowl-2026-analytics", "-p", base_dir, "--force"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Erreur lors du téléchargement Kaggle : {e}")
        st.stop()

    # Extraction du ZIP
    for file in os.listdir(base_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(base_dir, file)
            st.info(f"📦 Extraction de {zip_path} ...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(base_dir)
            st.success(f"✅ Fichiers extraits dans {base_dir}")
            os.remove(zip_path)


# =======================================================
# 📊 Chargement des données (avec fallback automatique)
# =======================================================
def load_data(use_kaggle=False, username=None, key=None, base_dir="./data"):
    """
    Charge les données NFL soit depuis Kaggle, soit localement.
    Crée le répertoire ./data s'il n'existe pas.
    """
    os.makedirs(base_dir, exist_ok=True)

    # Mode Kaggle
    if use_kaggle:
        download_from_kaggle(username=username, key=key, base_dir=base_dir)

    # Vérifie que des CSV existent
    csv_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".csv")]
    if not csv_files:
        st.error("❌ Aucun fichier CSV trouvé dans ./data/. Téléchargez-les manuellement ou activez use_kaggle=True.")
        st.stop()

    # Chargement des CSV
    dfs = []
    for csv in csv_files:
        try:
            df = pd.read_csv(csv, low_memory=False)
            dfs.append(df)
        except Exception as e:
            st.warning(f"⚠️ Impossible de lire {csv}: {e}")

    if not dfs:
        st.error("❌ Aucun DataFrame valide n'a pu être chargé.")
        st.stop()

    # Concaténation
    full_df = pd.concat(dfs, ignore_index=True)
    st.success(f"📄 {len(csv_files)} fichiers chargés avec succès ({len(full_df):,} lignes totales).")
    return full_df


# =======================================================
# 📈 Calcul des KPIs NFL
# =======================================================
def compute_all_kpis_and_aggregate(full_df, df_input=None, df_out=None, df_supp=None, tr_sample=None, prethrow=None):
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

    # ✅ Données simulées KPI passes
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
