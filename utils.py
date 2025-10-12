# =======================================================
# utils.py - Kaggle Data Access & NFL Dashboard KPIs
# =======================================================
import os
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import zipfile
import shutil


# =======================================================
# 📦 Load Data from Kaggle API
# =======================================================
def load_data_from_kaggle(username, key, competition="nfl-big-data-bowl-2025"):
    """
    Load CSV files directly from Kaggle competition using the API.
    
    Args:
        username (str): Kaggle username
        key (str): Kaggle API key
        competition (str): Competition name/slug
    
    Returns:
        pd.DataFrame: Combined dataframe from all CSV files
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        st.error("❌ Kaggle module not installed. Run: `pip install kaggle`")
        st.stop()
    
    # Set environment variables for authentication
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    
    try:
        # Initialize and authenticate API
        api = KaggleApi()
        api.authenticate()
        
        # Clean up any existing temp directory
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
        os.makedirs('temp_data', exist_ok=True)
        
        st.info(f"📥 Downloading all competition files from {competition}...")
        
        # Download ALL competition files at once (this handles the directory structure)
        api.competition_download_files(
            competition,
            path='temp_data',
            force=True,
            quiet=False
        )
        
        # Find and extract the main zip file
        zip_files = [f for f in os.listdir('temp_data') if f.endswith('.zip')]
        
        if not zip_files:
            st.error("❌ No zip file downloaded from competition")
            st.stop()
        
        main_zip = os.path.join('temp_data', zip_files[0])
        st.info(f"📦 Extracting {zip_files[0]}...")
        
        # Extract the main competition zip
        with zipfile.ZipFile(main_zip, 'r') as zip_ref:
            zip_ref.extractall('temp_data')
        
        # Now find all CSV files in the extracted structure
        csv_files = []
        for root, dirs, files in os.walk('temp_data'):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            st.error("❌ No CSV files found in extracted data")
            st.stop()
        
        st.success(f"✅ Found {len(csv_files)} CSV files")
        
        # Load each CSV file
        dfs = []
        progress_bar = st.progress(0)
        
        for idx, file_path in enumerate(csv_files):
            file_name = os.path.basename(file_path)
            st.info(f"📥 Loading {file_name}... ({idx+1}/{len(csv_files)})")
            
            try:
                df = pd.read_csv(file_path, low_memory=False)
                dfs.append(df)
                st.success(f"✓ Loaded {file_name}: {len(df):,} rows, {len(df.columns)} columns")
                
            except Exception as e:
                st.warning(f"⚠️ Skipped {file_name}: {str(e)}")
            
            # Update progress
            progress_bar.progress((idx + 1) / len(csv_files))
        
        if not dfs:
            st.error("❌ No data could be loaded from any files")
            st.stop()
        
        # Combine all dataframes
        st.info("🔄 Combining all data...")
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Clean up temp files
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
        
        st.success(f"✅ Successfully loaded {len(csv_files)} files with {len(full_df):,} total rows and {len(full_df.columns)} columns")
        
        return full_df
        
    except Exception as e:
        st.error(f"❌ Error loading data from Kaggle: {str(e)}")
        st.info("💡 Tips:")
        st.info("- Make sure you've accepted the competition rules on Kaggle")
        st.info("- Verify your Kaggle credentials are correct")
        st.info("- Check that the competition name is correct")
        
        # Clean up on error
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
        
        st.stop()


# =======================================================
# 📈 Compute All KPIs
# =======================================================
def compute_all_kpis(df):
    """
    Calculate all Key Performance Indicators from NFL tracking data.
    
    Args:
        df (pd.DataFrame): NFL tracking data
    
    Returns:
        dict: Dictionary of KPI names and values
    """
    
    def safe_mean(dataframe, column_name):
        """Safely compute mean, handling missing columns and NaN values."""
        if column_name in dataframe.columns:
            values = pd.to_numeric(dataframe[column_name], errors='coerce')
            return float(values.mean()) if not values.isna().all() else np.nan
        return np.nan
    
    def safe_max(dataframe, column_name):
        """Safely compute max value."""
        if column_name in dataframe.columns:
            values = pd.to_numeric(dataframe[column_name], errors='coerce')
            return float(values.max()) if not values.isna().all() else np.nan
        return np.nan
    
    def safe_min(dataframe, column_name):
        """Safely compute min value."""
        if column_name in dataframe.columns:
            values = pd.to_numeric(dataframe[column_name], errors='coerce')
            return float(values.min()) if not values.isna().all() else np.nan
        return np.nan
    
    def safe_count_unique(dataframe, column_name):
        """Safely count unique values."""
        if column_name in dataframe.columns:
            return float(dataframe[column_name].nunique())
        return np.nan
    
    def safe_std(dataframe, column_name):
        """Safely compute standard deviation."""
        if column_name in dataframe.columns:
            values = pd.to_numeric(dataframe[column_name], errors='coerce')
            return float(values.std()) if not values.isna().all() else np.nan
        return np.nan
    
    # Get all available columns
    cols = df.columns.tolist()
    
    # Define KPIs with multiple possible column names
    kpis = {}
    
    # Yards metrics
    for col in ['yards_gained', 'yardsGained', 'yards', 'yardage']:
        if col in cols:
            kpis['PPE (Yards Gained)'] = safe_mean(df, col)
            break
    
    # Completion probability
    for col in ['completion_probability', 'completionProbability', 'comp_prob']:
        if col in cols:
            kpis['CBR (Completion Prob)'] = safe_mean(df, col)
            break
    
    # Frame ID
    for col in ['frame_id', 'frameId', 'frame']:
        if col in cols:
            kpis['FFM (Frame ID)'] = safe_mean(df, col)
            break
    
    # Distance
    for col in ['dis', 'distance', 'dist']:
        if col in cols:
            kpis['ADY (Distance)'] = safe_mean(df, col)
            break
    
    # Time
    for col in ['time', 'frameTime', 'game_time']:
        if col in cols:
            kpis['TDR (Time)'] = safe_mean(df, col)
            break
    
    # Defender distance
    for col in ['closest_defender_distance', 'defenderDistance', 'defender_dist']:
        if col in cols:
            kpis['CWE (Defender Dist)'] = safe_mean(df, col)
            break
    
    # End speed
    for col in ['end_speed', 'endSpeed', 'final_speed']:
        if col in cols:
            kpis['EDS (End Speed)'] = safe_mean(df, col)
            break
    
    # Speed
    for col in ['s', 'speed', 'velocity']:
        if col in cols:
            kpis['VMC (Speed)'] = safe_mean(df, col)
            kpis['SMV (Speed Max)'] = safe_max(df, col)
            kpis['Speed Min'] = safe_min(df, col)
            kpis['Speed Std Dev'] = safe_std(df, col)
            break
    
    # Play result
    for col in ['play_result', 'playResult', 'result']:
        if col in cols:
            kpis['PMA (Play Result)'] = safe_mean(df, col)
            break
    
    # Expected yards
    for col in ['expected_yards', 'expectedYards', 'exp_yards']:
        if col in cols:
            kpis['PER (Expected Yards)'] = safe_mean(df, col)
            break
    
    # Acceleration
    for col in ['a', 'acceleration', 'accel']:
        if col in cols:
            kpis['AEF (Acceleration)'] = safe_mean(df, col)
            kpis['Max Acceleration'] = safe_max(df, col)
            break
    
    # Direction
    for col in ['dir', 'direction', 'heading']:
        if col in cols:
            kpis['DIR (Direction)'] = safe_mean(df, col)
            break
    
    # Orientation
    for col in ['o', 'orientation', 'orient']:
        if col in cols:
            kpis['ORI (Orientation)'] = safe_mean(df, col)
            break
    
    # Position metrics
    if 'x' in cols:
        kpis['Avg X Position'] = safe_mean(df, 'x')
        kpis['Max X Position'] = safe_max(df, 'x')
    
    if 'y' in cols:
        kpis['Avg Y Position'] = safe_mean(df, 'y')
        kpis['Y Position Range'] = safe_max(df, 'y') - safe_min(df, 'y') if safe_max(df, 'y') != np.nan else np.nan
    
    # Unique counts
    for col in ['nflId', 'playerId', 'player_id']:
        if col in cols:
            kpis['Unique Players'] = safe_count_unique(df, col)
            break
    
    for col in ['playId', 'play_id']:
        if col in cols:
            kpis['Unique Plays'] = safe_count_unique(df, col)
            break
    
    for col in ['gameId', 'game_id']:
        if col in cols:
            kpis['Unique Games'] = safe_count_unique(df, col)
            break
    
    # Remove any NaN KPIs
    kpis = {k: v for k, v in kpis.items() if not np.isnan(v)}
    
    return kpis


# =======================================================
# 🔄 Alternative: Load from Local Directory
# =======================================================
def load_local_data(data_dir="./data"):
    """
    Load CSV files from a local directory.
    
    Args:
        data_dir (str): Path to directory containing CSV files
    
    Returns:
        pd.DataFrame: Combined dataframe from all CSV files
    """
    if not os.path.exists(data_dir):
        st.error(f"❌ Directory not found: {data_dir}")
        st.stop()
    
    # Find all CSV files recursively
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        st.error(f"❌ No CSV files found in {data_dir}")
        st.stop()
    
    st.info(f"📄 Found {len(csv_files)} CSV files in {data_dir}")
    
    dfs = []
    progress_bar = st.progress(0)
    
    for idx, file_path in enumerate(csv_files):
        file_name = os.path.basename(file_path)
        st.info(f"📥 Loading {file_name}... ({idx+1}/{len(csv_files)})")
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            dfs.append(df)
            st.success(f"✓ {file_name}: {len(df):,} rows")
        except Exception as e:
            st.warning(f"⚠️ Skipped {file_name}: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(csv_files))
    
    if not dfs:
        st.error("❌ No data could be loaded")
        st.stop()
    
    full_df = pd.concat(dfs, ignore_index=True)
    st.success(f"✅ Loaded {len(full_df):,} total rows from {len(dfs)} files")
    
    return full_df


# =======================================================
# 📊 Additional Helper Functions
# =======================================================
def get_column_info(df):
    """Get detailed information about dataframe columns."""
    info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null': df.isnull().sum().values,
        'Null %': (df.isnull().sum().values / len(df) * 100).round(2),
        'Unique': [df[col].nunique() for col in df.columns]
    })
    return info


def detect_available_columns(df):
    """Detect which standard NFL columns are available in the dataframe."""
    standard_columns = {
        'Tracking Data': ['x', 'y', 's', 'a', 'dis', 'o', 'dir'],
        'Identifiers': ['gameId', 'playId', 'nflId', 'frameId'],
        'Performance Metrics': ['yards_gained', 'completion_probability', 'expected_yards'],
        'Player Info': ['displayName', 'jerseyNumber', 'position', 'team']
    }
    
    available = {}
    for category, cols in standard_columns.items():
        found = [col for col in cols if col in df.columns]
        if found:
            available[category] = found
    
    return available


def get_data_summary(df):
    """Generate a comprehensive summary of the dataset."""
    summary = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Memory Usage (MB)': df.memory_usage(deep=True).sum() / 1024**2,
        'Duplicate Rows': df.duplicated().sum(),
        'Total Missing Values': df.isnull().sum().sum(),
        'Missing %': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100).round(2)
    }
    return summary
