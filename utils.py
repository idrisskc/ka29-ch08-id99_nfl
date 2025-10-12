# =======================================================
# utils.py - Kaggle Data Access & NFL Dashboard KPIs
# =======================================================
import os
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO


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
        
        # Get list of files from competition
        files = api.competition_list_files(competition)
        csv_files = [f.name for f in files.files if f.name.endswith('.csv')]
        
        if not csv_files:
            st.error(f"❌ No CSV files found in competition: {competition}")
            st.stop()
        
        st.info(f"📄 Found {len(csv_files)} CSV files in {competition}")
        
        # Download and load each CSV file
        dfs = []
        progress_bar = st.progress(0)
        
        for idx, file_name in enumerate(csv_files):
            st.info(f"📥 Loading {file_name}... ({idx+1}/{len(csv_files)})")
            
            try:
                # Download file to temp location
                api.competition_download_file(
                    competition, 
                    file_name, 
                    path='temp_data',
                    force=True
                )
                
                # Read the downloaded file
                file_path = f"temp_data/{file_name}"
                if file_name.endswith('.zip'):
                    # Handle zipped files
                    import zipfile
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall('temp_data')
                    actual_file = file_name.replace('.zip', '')
                    df = pd.read_csv(f"temp_data/{actual_file}", low_memory=False)
                else:
                    df = pd.read_csv(file_path, low_memory=False)
                
                dfs.append(df)
                st.success(f"✓ Loaded {file_name}: {len(df):,} rows")
                
            except Exception as e:
                st.warning(f"⚠️ Skipped {file_name}: {str(e)}")
            
            # Update progress
            progress_bar.progress((idx + 1) / len(csv_files))
        
        if not dfs:
            st.error("❌ No data could be loaded from any files")
            st.stop()
        
        # Combine all dataframes
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Clean up temp files
        import shutil
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
        
        st.success(f"✅ Successfully loaded {len(csv_files)} files with {len(full_df):,} total rows")
        
        return full_df
        
    except Exception as e:
        st.error(f"❌ Error loading data from Kaggle: {str(e)}")
        st.info("💡 Tip: Make sure you've accepted the competition rules on Kaggle")
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
    
    def safe_count_unique(dataframe, column_name):
        """Safely count unique values."""
        if column_name in dataframe.columns:
            return float(dataframe[column_name].nunique())
        return np.nan
    
    # Define KPIs with fallback column names
    kpis = {
        'PPE (Yards Gained)': safe_mean(df, 'yards_gained') or safe_mean(df, 'yardsGained') or safe_mean(df, 'yards'),
        
        'CBR (Completion Prob)': safe_mean(df, 'completion_probability') or safe_mean(df, 'completionProbability'),
        
        'FFM (Frame ID)': safe_mean(df, 'frame_id') or safe_mean(df, 'frameId'),
        
        'ADY (Distance)': safe_mean(df, 'dis') or safe_mean(df, 'distance'),
        
        'TDR (Time)': safe_mean(df, 'time') or safe_mean(df, 'frameTime'),
        
        'CWE (Defender Dist)': safe_mean(df, 'closest_defender_distance') or safe_mean(df, 'defenderDistance'),
        
        'EDS (End Speed)': safe_mean(df, 'end_speed') or safe_mean(df, 'endSpeed'),
        
        'VMC (Speed)': safe_mean(df, 's') or safe_mean(df, 'speed'),
        
        'PMA (Play Result)': safe_mean(df, 'play_result') or safe_mean(df, 'playResult'),
        
        'PER (Expected Yards)': safe_mean(df, 'expected_yards') or safe_mean(df, 'expectedYards'),
        
        'SMV (Speed Max)': safe_max(df, 's') or safe_max(df, 'speed'),
        
        'AEF (Acceleration)': safe_mean(df, 'a') or safe_mean(df, 'acceleration'),
        
        'DIR (Direction)': safe_mean(df, 'dir') or safe_mean(df, 'direction'),
        
        'ORI (Orientation)': safe_mean(df, 'o') or safe_mean(df, 'orientation'),
    }
    
    # Add derived metrics if possible
    try:
        # Unique players
        if 'nflId' in df.columns:
            kpis['Unique Players'] = safe_count_unique(df, 'nflId')
        
        # Unique plays
        if 'playId' in df.columns:
            kpis['Unique Plays'] = safe_count_unique(df, 'playId')
        
        # Average X position (field position)
        if 'x' in df.columns:
            kpis['Avg Field Position'] = safe_mean(df, 'x')
        
        # Average Y position (lateral position)
        if 'y' in df.columns:
            kpis['Avg Lateral Position'] = safe_mean(df, 'y')
    
    except Exception as e:
        pass  # Silently skip derived metrics if there's an error
    
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
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        st.error(f"❌ No CSV files found in {data_dir}")
        st.stop()
    
    st.info(f"📄 Found {len(csv_files)} CSV files in {data_dir}")
    
    dfs = []
    for file_name in csv_files:
        file_path = os.path.join(data_dir, file_name)
        st.info(f"📥 Loading {file_name}...")
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            dfs.append(df)
            st.success(f"✓ {file_name}: {len(df):,} rows")
        except Exception as e:
            st.warning(f"⚠️ Skipped {file_name}: {str(e)}")
    
    if not dfs:
        st.error("❌ No data could be loaded")
        st.stop()
    
    full_df = pd.concat(dfs, ignore_index=True)
    st.success(f"✅ Loaded {len(full_df):,} total rows")
    
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
        'Unique': [df[col].nunique() for col in df.columns]
    })
    return info


def detect_available_columns(df):
    """Detect which standard NFL columns are available in the dataframe."""
    standard_columns = {
        'tracking': ['x', 'y', 's', 'a', 'dis', 'o', 'dir'],
        'play': ['gameId', 'playId', 'nflId', 'frameId'],
        'metrics': ['yards_gained', 'completion_probability', 'expected_yards']
    }
    
    available = {}
    for category, cols in standard_columns.items():
        available[category] = [col for col in cols if col in df.columns]
    
    return available
