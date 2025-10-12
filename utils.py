# =======================================================
# utils.py - Utilities for NFL Analytics Dashboard
# =======================================================
import os
import pandas as pd
import numpy as np
import zipfile
import subprocess

def download_from_kaggle(username, key, base_dir="./data"):
    """Download and extract Kaggle dataset."""
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    kaggle_json_path = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_json_path, exist_ok=True)
    with open(os.path.join(kaggle_json_path, "kaggle.json"), "w") as f:
        f.write(f'{{"username":"{username}","key":"{key}"}}')
    os.chmod(os.path.join(kaggle_json_path, "kaggle.json"), 0o600)

    os.makedirs(base_dir, exist_ok=True)
    zip_path = os.path.join(base_dir, "nfl_data.zip")

    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "nfl-big-data-bowl-2026-analytics",
        "-p", base_dir, "--force"
    ], check=True)

    zip_files = [f for f in os.listdir(base_dir) if f.endswith(".zip")]
    if zip_files:
        with zipfile.ZipFile(os.path.join(base_dir, zip_files[0]), "r") as zf:
            zf.extractall(base_dir)
    print("✅ Dataset ready in", base_dir)

def load_data(base_dir="./data"):
    """Load dataset locally."""
    full_df, df_input, df_out, df_supp, tr_sample, prethrow = [pd.DataFrame()]*6
    supp_path = os.path.join(base_dir, "supplementary_data.csv")
    if os.path.exists(supp_path):
        df_supp = pd.read_csv(supp_path)
    train_dir = os.path.join(base_dir, "train")
    if os.path.exists(train_dir):
        csvs = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
        if csvs:
            df_input = pd.read_csv(os.path.join(train_dir, csvs[0]))
            full_df = df_input.copy()
    return full_df, df_input, df_out, df_supp, tr_sample, prethrow

def compute_all_kpis_and_aggregate(full_df, df_input, df_out, df_supp, tr_sample, prethrow):
    """Compute simulated NFL KPIs."""
    def safe_mean(df, col): return float(df[col].mean()) if col in df.columns else np.nan
    kpis = {
        "Avg Yards Gained": safe_mean(full_df, "yards_gained"),
        "Completion Prob": safe_mean(full_df, "completion_probability"),
        "Avg Speed": safe_mean(full_df, "s"),
        "Accel Mean": safe_mean(full_df, "a")
    }
    df_pass_kpis = pd.DataFrame({
        "ADR": np.random.rand(50),
        "ME": np.random.rand(50),
        "cli_final": np.random.rand(50)
    })
    return kpis, df_pass_kpis
