# =======================================================
# utils.py - OPTIMIZED (NO SCIPY VERSION)
# =======================================================
import os
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import zipfile
import shutil
import gc


# =======================================================
# 📦 Load Data from Kaggle API (OPTIMIZED)
# =======================================================
def load_data_from_kaggle(username, key, competition="nfl-big-data-bowl-2026-analytics"):
    """Load CSV files from Kaggle - OPTIMIZED VERSION"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        st.error("❌ Kaggle module not installed")
        return pd.DataFrame()
    
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
        os.makedirs('temp_data', exist_ok=True)
        
        st.info("📥 Downloading competition files...")
        
        api.competition_download_files(competition, path='temp_data', force=True, quiet=False)
        
        zip_files = [f for f in os.listdir('temp_data') if f.endswith('.zip')]
        
        if not zip_files:
            st.error("❌ No zip file downloaded")
            return pd.DataFrame()
        
        main_zip = os.path.join('temp_data', zip_files[0])
        st.info(f"📦 Extracting {zip_files[0]}...")
        
        with zipfile.ZipFile(main_zip, 'r') as zip_ref:
            zip_ref.extractall('temp_data')
        
        csv_files = []
        for root, dirs, files in os.walk('temp_data'):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            st.error("❌ No CSV files found")
            return pd.DataFrame()
        
        st.success(f"✅ Found {len(csv_files)} CSV files")
        
        MAX_FILES = 10
        
        if len(csv_files) > MAX_FILES:
            st.warning(f"⚠️ Loading only first {MAX_FILES} files to prevent memory issues.")
            csv_files = csv_files[:MAX_FILES]
        
        dfs = []
        progress_bar = st.progress(0)
        
        for idx, file_path in enumerate(csv_files):
            file_name = os.path.basename(file_path)
            
            try:
                st.info(f"📥 Loading {file_name}... ({idx+1}/{len(csv_files)})")
                
                df = pd.read_csv(file_path, low_memory=False)
                df = optimize_dataframe(df)
                
                dfs.append(df)
                st.success(f"✓ {file_name}: {len(df):,} rows, {len(df.columns)} cols")
                
                del df
                gc.collect()
                
            except Exception as e:
                st.warning(f"⚠️ Skipped {file_name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(csv_files))
        
        if not dfs:
            st.error("❌ No data loaded")
            return pd.DataFrame()
        
        st.info("🔄 Combining all data...")
        full_df = pd.concat(dfs, ignore_index=True)
        
        del dfs
        gc.collect()
        
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
        
        st.success(f"✅ Loaded {len(full_df):,} total rows, {len(full_df.columns)} columns")
        
        return full_df
        
    except Exception as e:
        st.error(f"❌ Kaggle Error: {str(e)}")
        
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
        
        return pd.DataFrame()


def optimize_dataframe(df):
    """Optimise les types de données pour réduire l'utilisation mémoire"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'object':
            num_unique = df[col].nunique()
            if num_unique / len(df) < 0.5:
                df[col] = df[col].astype('category')
        
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')
        
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')
    
    return df


# =======================================================
# Custom Entropy Function (replaces scipy)
# =======================================================
def calculate_entropy(values):
    """Calculate Shannon entropy without scipy"""
    value_counts = pd.Series(values).value_counts(normalize=True)
    entropy_value = -np.sum(value_counts * np.log2(value_counts + 1e-10))
    return entropy_value


# =======================================================
# 📈 Compute Basic KPIs
# =======================================================
def compute_all_kpis(df):
    """Calculate basic KPIs"""
    
    def safe_mean(dataframe, column_name):
        try:
            if column_name in dataframe.columns:
                values = pd.to_numeric(dataframe[column_name], errors='coerce')
                result = float(values.mean())
                return result if not np.isnan(result) else np.nan
        except:
            pass
        return np.nan
    
    def safe_max(dataframe, column_name):
        try:
            if column_name in dataframe.columns:
                values = pd.to_numeric(dataframe[column_name], errors='coerce')
                result = float(values.max())
                return result if not np.isnan(result) else np.nan
        except:
            pass
        return np.nan
    
    def safe_count_unique(dataframe, column_name):
        try:
            if column_name in dataframe.columns:
                return float(dataframe[column_name].nunique())
        except:
            pass
        return np.nan
    
    kpis = {}
    cols = df.columns.tolist()
    
    # Basic KPIs
    for col in ['yards_gained', 'yardsGained', 'yards']:
        if col in cols:
            val = safe_mean(df, col)
            if not np.isnan(val):
                kpis['Yards Gained'] = val
                break
    
    for col in ['s', 'speed']:
        if col in cols:
            val_mean = safe_mean(df, col)
            val_max = safe_max(df, col)
            if not np.isnan(val_mean):
                kpis['Speed Avg'] = val_mean
            if not np.isnan(val_max):
                kpis['Speed Max'] = val_max
            break
    
    for col in ['a', 'acceleration']:
        if col in cols:
            val = safe_mean(df, col)
            if not np.isnan(val):
                kpis['Acceleration'] = val
                break
    
    if 'x' in cols:
        val = safe_mean(df, 'x')
        if not np.isnan(val):
            kpis['Avg X Position'] = val
    
    if 'y' in cols:
        val = safe_mean(df, 'y')
        if not np.isnan(val):
            kpis['Avg Y Position'] = val
    
    for col in ['nflId', 'nfl_id']:
        if col in cols:
            val = safe_count_unique(df, col)
            if not np.isnan(val):
                kpis['Unique Players'] = val
                break
    
    for col in ['playId', 'play_id']:
        if col in cols:
            val = safe_count_unique(df, col)
            if not np.isnan(val):
                kpis['Unique Plays'] = val
                break
    
    kpis = {k: v for k, v in kpis.items() if not np.isnan(v)}
    
    return kpis


# =======================================================
# 🎯 STRATEGIC KPIs
# =======================================================

def calculate_defensive_pressure_index(df):
    """KPI 1: Defensive Pressure Index"""
    try:
        if 'player_side' not in df.columns:
            return None
        
        defense_df = df[df['player_side'] == 'Defense'].copy()
        
        required_cols = ['game_id', 'play_id', 'defenders_in_the_box', 's', 'dropback_distance']
        if not all(col in defense_df.columns for col in required_cols):
            return None
        
        dpi_data = defense_df.groupby(['game_id', 'play_id']).agg({
            'defenders_in_the_box': 'first',
            's': 'mean',
            'dropback_distance': 'first'
        }).reset_index()
        
        dpi_data['DPI'] = (dpi_data['defenders_in_the_box'] * dpi_data['s']) / (dpi_data['dropback_distance'] + 1)
        
        return dpi_data['DPI'].mean()
    except:
        return None


def calculate_coverage_vulnerability_matrix(df):
    """KPI 2: Coverage Vulnerability Matrix"""
    try:
        required_cols = ['team_coverage_type', 'receiver_alignment', 'yards_gained', 'expected_points']
        if not all(col in df.columns for col in required_cols):
            return None
        
        cvm_data = df.groupby(['team_coverage_type', 'receiver_alignment']).agg({
            'yards_gained': 'mean',
            'expected_points': 'mean'
        }).reset_index()
        
        cvm_data['CVM'] = cvm_data['yards_gained'] / (cvm_data['expected_points'] + 0.1)
        
        return cvm_data
    except:
        return None


def calculate_route_efficiency_score(df):
    """KPI 3: Route Efficiency Score"""
    try:
        if 'player_role' not in df.columns or 'route_of_targeted_receiver' not in df.columns:
            return None
        
        receivers_df = df[df['player_role'] == 'Targeted Receiver'].copy()
        
        if len(receivers_df) == 0:
            return None
        
        res_data = receivers_df.groupby('route_of_targeted_receiver').agg({
            'yards_gained': 'mean',
            'pass_result': lambda x: (x == 'C').sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        
        res_data.columns = ['route', 'avg_yards', 'completion_rate']
        res_data['RES'] = res_data['avg_yards'] * res_data['completion_rate']
        
        return res_data
    except:
        return None


def calculate_win_probability_leverage(df):
    """KPI 4: Win Probability Leverage"""
    try:
        if 'home_team_win_probability_added' not in df.columns:
            return None
        
        wpl_data = df.groupby(['down', 'quarter']).agg({
            'home_team_win_probability_added': lambda x: abs(x).mean()
        }).reset_index()
        
        wpl_data.columns = ['down', 'quarter', 'WPL']
        
        return wpl_data
    except:
        return None


def calculate_red_zone_efficiency(df):
    """KPI 5: Red Zone Conversion Efficiency"""
    try:
        if 'absolute_yardline_number' not in df.columns:
            return None
        
        rz_df = df[df['absolute_yardline_number'] <= 20].copy()
        
        if len(rz_df) == 0:
            return None
        
        rzce = rz_df.groupby('pass_result').size().to_dict()
        total = sum(rzce.values())
        
        completion_rate = rzce.get('C', 0) / total if total > 0 else 0
        
        return completion_rate * 100
    except:
        return None


def calculate_formation_predictability(df):
    """KPI 6: Formation Predictability Index (using custom entropy)"""
    try:
        if 'offense_formation' not in df.columns or 'pass_result' not in df.columns:
            return None
        
        fpi_data = df.groupby('offense_formation')['pass_result'].apply(
            lambda x: calculate_entropy(x) if len(x) > 1 else 0
        ).reset_index()
        
        fpi_data.columns = ['formation', 'entropy']
        fpi_data['FPI'] = 1 / (fpi_data['entropy'] + 0.1)
        
        return fpi_data
    except:
        return None


def calculate_spatial_advantage(df):
    """KPI 7: Spatial Advantage Score"""
    try:
        if 'player_role' not in df.columns:
            return None
        
        receivers = df[df['player_role'] == 'Targeted Receiver'][['game_id', 'play_id', 'x', 'y']].copy()
        defenders = df[df['player_role'] == 'Defensive Coverage'][['game_id', 'play_id', 'x', 'y']].copy()
        
        if len(receivers) == 0 or len(defenders) == 0:
            return None
        
        receivers.columns = ['game_id', 'play_id', 'rx', 'ry']
        
        merged = receivers.merge(defenders, on=['game_id', 'play_id'], how='left')
        merged['separation'] = np.sqrt((merged['rx'] - merged['x'])**2 + (merged['ry'] - merged['y'])**2)
        
        avg_separation = merged['separation'].mean()
        
        return (avg_separation / 53.3) * 100 if not np.isnan(avg_separation) else None
    except:
        return None


def calculate_all_strategic_kpis(df):
    """Calculate all strategic KPIs and return as dict"""
    strategic_kpis = {}
    
    # DPI
    dpi = calculate_defensive_pressure_index(df)
    if dpi is not None:
        strategic_kpis['Defensive Pressure Index'] = dpi
    
    # Red Zone
    rzce = calculate_red_zone_efficiency(df)
    if rzce is not None:
        strategic_kpis['Red Zone Efficiency %'] = rzce
    
    # Spatial Advantage
    sas = calculate_spatial_advantage(df)
    if sas is not None:
        strategic_kpis['Spatial Advantage Score'] = sas
    
    return strategic_kpis


def get_strategic_kpi_dataframes(df):
    """Get detailed DataFrames for strategic KPIs visualization"""
    kpi_dfs = {}
    
    # Coverage Vulnerability Matrix
    cvm = calculate_coverage_vulnerability_matrix(df)
    if cvm is not None and not cvm.empty:
        kpi_dfs['CVM'] = cvm
    
    # Route Efficiency
    res = calculate_route_efficiency_score(df)
    if res is not None and not res.empty:
        kpi_dfs['RES'] = res
    
    # Win Probability Leverage
    wpl = calculate_win_probability_leverage(df)
    if wpl is not None and not wpl.empty:
        kpi_dfs['WPL'] = wpl
    
    # Formation Predictability
    fpi = calculate_formation_predictability(df)
    if fpi is not None and not fpi.empty:
        kpi_dfs['FPI'] = fpi
    
    return kpi_dfs


# =======================================================
# 🔄 Load from Local Directory
# =======================================================
def load_local_data(data_dir="./data"):
    """Load CSV files from local directory"""
    if not os.path.exists(data_dir):
        st.error(f"❌ Directory not found: {data_dir}")
        return pd.DataFrame()
    
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        st.error(f"❌ No CSV files in {data_dir}")
        return pd.DataFrame()
    
    st.info(f"📄 Found {len(csv_files)} CSV files")
    
    dfs = []
    progress_bar = st.progress(0)
    
    for idx, file_path in enumerate(csv_files):
        file_name = os.path.basename(file_path)
        st.info(f"📥 Loading {file_name}...")
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df = optimize_dataframe(df)
            dfs.append(df)
            st.success(f"✓ {file_name}: {len(df):,} rows")
        except Exception as e:
            st.warning(f"⚠️ Skipped {file_name}: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(csv_files))
    
    if not dfs:
        st.error("❌ No data loaded")
        return pd.DataFrame()
    
    full_df = pd.concat(dfs, ignore_index=True)
    st.success(f"✅ Loaded {len(full_df):,} total rows")
    
    return full_df


# =======================================================
# 📊 Helper Functions
# =======================================================
def get_column_info(df):
    """Get column information"""
    try:
        info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Null %': (df.isnull().sum().values / len(df) * 100).round(2),
            'Unique': [df[col].nunique() for col in df.columns]
        })
        return info
    except Exception as e:
        st.error(f"Error in get_column_info: {str(e)}")
        return pd.DataFrame()


def detect_available_columns(df):
    """Detect standard NFL columns"""
    standard_columns = {
        'Tracking Data': ['x', 'y', 's', 'a', 'dis', 'o', 'dir'],
        'Identifiers': ['gameId', 'playId', 'nflId', 'frameId', 'game_id', 'play_id', 'nfl_id', 'frame_id'],
        'Performance Metrics': ['yards_gained', 'expected_points', 'expected_points_added'],
        'Player Info': ['player_name', 'player_position', 'player_side', 'player_role'],
        'Strategic': ['team_coverage_type', 'offense_formation', 'receiver_alignment']
    }
    
    available = {}
    for category, cols in standard_columns.items():
        found = [col for col in cols if col in df.columns]
        if found:
            available[category] = found
    
    return available


def get_data_summary(df):
    """Generate dataset summary"""
    try:
        summary = {
            'Total Rows': len(df),
            'Total Columns': len(df.columns),
            'Memory Usage (MB)': df.memory_usage(deep=True).sum() / 1024**2,
            'Duplicate Rows': df.duplicated().sum(),
            'Total Missing Values': df.isnull().sum().sum(),
            'Missing %': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        }
        return summary
    except Exception as e:
        st.error(f"Error in get_data_summary: {str(e)}")
        return {
            'Total Rows': 0,
            'Total Columns': 0,
            'Memory Usage (MB)': 0,
            'Duplicate Rows': 0,
            'Total Missing Values': 0,
            'Missing %': 0
        }
