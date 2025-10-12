# =======================================================
# utils.py - Robust Kaggle + Data Utils for NFL Dashboard
# =======================================================
import os
import pandas as pd
import numpy as np
import zipfile
import subprocess
import streamlit as st

# =======================================================
# 📦 Téléchargement et extraction depuis Kaggle
# =======================================================
def download_from_kaggle(username, key, base_dir="./data"):
    """Télécharge et extrait le dataset NFL Big Data Bowl depuis Kaggle (robuste)."""
    os.makedirs(base_dir, exist_ok=True)

    # Configuration API
    kaggle_json_path = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_json_path, exist_ok=True)
    kaggle_json_file = os.path.join(kaggle_json_path, "kaggle.json")
    with open(kaggle_json_file, "w") as f:
        f.write(f'{{"username":"{username}","key":"{key}"}}')
    os.chmod(kaggle_json_file, 0o600)

    # Vérifier que le module Kaggle est bien installé
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except Exception as e:
        st.error("❌ Le module Kaggle n'est pas disponible sur ce serveur.")
        st.stop()

    # Téléchargement
    try:
        st.info("📥 Téléchargement du dataset NFL depuis Kaggle...")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", "nfl-big-data-bowl-2026-analytics", "-p", base_dir, "--force"],
            check=True,
            capture_output=True,
            text=True
        )
        st.text(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Erreur Kaggle : {e.stderr or e}")
        st.stop()

    # Extraction ZIP
    for file in os.listdir(base_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(base_dir, file)
            st.info(f"📦 Extraction de {zip_path} ...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(base_dir)
            st.success(f"✅ Fichiers extraits dans {base_dir}")


# =======================================================
# 📊 Chargement des données
# =======================================================
def load_data(base_dir="./data"):
    """Charge tous les fichiers CSV extraits du répertoire data."""
    csv_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        st.error("❌ Aucun fichier CSV trouvé dans ./data après extraction.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    st.info(f"📄 {len(csv_files)} fichiers CSV détectés.")
    dfs = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, low_memory=False)
            dfs.append(df)
        except Exception as e:
            st.warning(f"⚠️ Impossible de lire {path}: {e}")

    if not dfs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df, *([pd.DataFrame()] * 5)


# =======================================================
# 📈 Calcul des KPIs NFL
# =======================================================
def compute_all_kpis_and_aggregate(full_df, df_input, df_out, df_supp, tr_sample, prethrow):
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

    # ✅ Données de simulation KPI passes
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
