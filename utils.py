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

    os.makedirs(base_dir, exist_ok=True)
    kaggle_json_path = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_json_path, exist_ok=True)
    with open(os.path.join(kaggle_json_path, "kaggle.json"), "w") as f:
        f.write(f'{{"username":"{username}","key":"{key}"}}')

    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "nfl-big-data-bowl-2026-analytics",
        "-p", base_dir, "--force"
    ], check=True)

    # Extract all ZIP files
    for file in os.listdir(base_dir):
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(base_dir, file), "r") as zf:
                zf.extractall(base_dir)
            print(f"✅ Extracted {file}")

def load_data(base_dir="./data"):
    """Load all CSV files found in data directory."""
    csv_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Load first CSV as base DataFrame
    main_df = pd.read_csv(csv_files[0])
    dfs = [main_df] + [pd.read_csv(f) for f in csv_files[1:]]
    full_df = pd.concat(dfs, ignore_index=True, axis=0)
    return full_df, *([pd.DataFrame()] * 5)

def compute_all_kpis_and_aggregate(full_df, *args):
    """Compute KPIs safely."""
    def safe_mean(col): return np.nanmean(full_df[col]) if col in full_df.columns else np.nan
    kpis = {
        "Avg Yards Gained": safe_mean("yards_gained"),
        "Completion Prob": safe_mean("completion_probability"),
        "Avg Speed": safe_mean("s"),
        "Accel Mean": safe_mean("a")
    }
    df_pass_kpis = pd.DataFrame({"KPI": list(kpis.keys()), "Value": list(kpis.values())})
    return kpis, df_pass_kpis
