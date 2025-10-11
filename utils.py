# =======================================================
# 🔧 utils.py - Fonctions pour NFL Analytics Dashboard
# =======================================================
import os
import pandas as pd
import numpy as np
import zipfile
import subprocess

# =======================================================
# 🔹 Téléchargement et extraction depuis Kaggle
# =======================================================
def download_from_kaggle(username, key, base_dir="./data"):
    """
    Télécharge les fichiers de la compétition Kaggle et les extrait.
    """
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    kaggle_json_path = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_json_path, exist_ok=True)
    with open(os.path.join(kaggle_json_path, "kaggle.json"), "w") as f:
        f.write(f'{{"username":"{username}","key":"{key}"}}')
    os.chmod(os.path.join(kaggle_json_path, "kaggle.json"), 0o600)

    os.makedirs(base_dir, exist_ok=True)
    zip_path = os.path.join(base_dir, "nfl_data.zip")

    print("📦 Téléchargement depuis Kaggle...")
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "nfl-big-data-bowl-2026-analytics",
        "-p", base_dir, "--force"
    ], check=True)

    # Extraction
    print("📂 Extraction...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(base_dir)

    print("✅ Données extraites dans :", base_dir)

# =======================================================
# 🔹 Chargement local des données
# =======================================================
def load_data(base_dir="./data"):
    full_df, df_input, df_out, df_supp, tr_sample, prethrow = [pd.DataFrame()]*6

    supp_path = os.path.join(base_dir, "supplementary_data.csv")
    if os.path.exists(supp_path):
        df_supp = pd.read_csv(supp_path)

    train_dir = os.path.join(base_dir, "train")
    if os.path.exists(train_dir):
        csv_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
        if csv_files:
            df_input = pd.read_csv(os.path.join(train_dir, csv_files[0]))
            full_df = df_input.copy()

    return full_df, df_input, df_out, df_supp, tr_sample, prethrow

# =======================================================
# 🔹 Calcul des KPIs
# =======================================================
def compute_all_kpis_and_aggregate(full_df, df_input, df_out, df_supp, tr_sample, prethrow):
    def safe_mean(df, col): return float(df[col].mean()) if col in df.columns else np.nan

    kpis = {
        'PPE': safe_mean(full_df, 'yards_gained'),
        'CBR': safe_mean(full_df, 'completion_probability'),
        'FFM': safe_mean(full_df, 'frame_id'),
        'ADY': safe_mean(full_df, 'distance'),
        'TDR': safe_mean(full_df, 'time'),
        'PEI': safe_mean(full_df, 'event'),
        'CWE': safe_mean(full_df, 'closest_defender_distance'),
        'EDS': safe_mean(full_df, 'end_speed'),
        'VMC': safe_mean(full_df, 's'),
        'PMA': safe_mean(full_df, 'play_result'),
        'PER': safe_mean(full_df, 'expected_yards'),
        'RCI': safe_mean(full_df, 'receiver_id'),
        'SMV': safe_mean(full_df, 'speed_max'),
        'AEF': safe_mean(full_df, 'acceleration')
    }

    # Simulated pass KPIs
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
