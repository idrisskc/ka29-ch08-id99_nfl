# =======================================================
# utils.py - OPTIMIZED for Streamlit Cloud
# =======================================================
import os
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import zipfile
import shutil
import gc  # Garbage collector pour libérer la mémoire


# =======================================================
# 📦 Load Data from Kaggle API (OPTIMIZED)
# =======================================================
def load_data_from_kaggle(username, key, competition="nfl-big-data-bowl-2026-analytics"):
    """
    Load CSV files from Kaggle - OPTIMIZED VERSION
    Limite le nombre de fichiers et la taille des données
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        st.error("❌ Kaggle module not installed")
        return pd.DataFrame()
    
    # Configuration API
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Nettoyage
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
        os.makedirs('temp_data', exist_ok=True)
        
        st.info("📥 Downloading competition files...")
        
        # Télécharger tous les fichiers
        api.competition_download_files(
            competition,
            path='temp_data',
            force=True,
            quiet=False
        )
        
        # Trouver et extraire le zip principal
        zip_files = [f for f in os.listdir('temp_data') if f.endswith('.zip')]
        
        if not zip_files:
            st.error("❌ No zip file downloaded")
            return pd.DataFrame()
        
        main_zip = os.path.join('temp_data', zip_files[0])
        st.info(f"📦 Extracting {zip_files[0]}...")
        
        with zipfile.ZipFile(main_zip, 'r') as zip_ref:
            zip_ref.extractall('temp_data')
        
        # Trouver tous les CSV
        csv_files = []
        for root, dirs, files in os.walk('temp_data'):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            st.error("❌ No CSV files found")
            return pd.DataFrame()
        
        st.success(f"✅ Found {len(csv_files)} CSV files")
        
        # OPTIMISATION: Limiter le nombre de fichiers pour éviter les crashes
        MAX_FILES = 25  # Limite à 10 fichiers pour commencer
        
        if len(csv_files) > MAX_FILES:
            st.warning(f"⚠️ Found {len(csv_files)} files. Loading only first {MAX_FILES} to prevent memory issues.")
            csv_files = csv_files[:MAX_FILES]
        
        # Charger les fichiers avec SAMPLE pour réduire la mémoire
        dfs = []
        progress_bar = st.progress(0)
        
        for idx, file_path in enumerate(csv_files):
            file_name = os.path.basename(file_path)
            
            try:
                st.info(f"📥 Loading {file_name}... ({idx+1}/{len(csv_files)})")
                
                # Charger avec dtype optimization
                df = pd.read_csv(
                    file_path, 
                    low_memory=False,
                    # Limite optionnelle: prendre seulement les premières lignes
                    # nrows=50000  # Décommenter si toujours des problèmes de mémoire
                )
                
                # Optimiser les types de données pour économiser la mémoire
                df = optimize_dataframe(df)
                
                dfs.append(df)
                st.success(f"✓ {file_name}: {len(df):,} rows, {len(df.columns)} cols")
                
                # Libérer la mémoire
                del df
                gc.collect()
                
            except Exception as e:
                st.warning(f"⚠️ Skipped {file_name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(csv_files))
        
        if not dfs:
            st.error("❌ No data loaded")
            return pd.DataFrame()
        
        # Combiner les dataframes
        st.info("🔄 Combining all data...")
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Nettoyage final
        del dfs
        gc.collect()
        
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
        
        st.success(f"✅ Loaded {len(full_df):,} total rows, {len(full_df.columns)} columns")
        
        return full_df
        
    except Exception as e:
        st.error(f"❌ Kaggle Error: {str(e)}")
        
        # Nettoyage en cas d'erreur
        if os.path.exists('temp_data'):
            shutil.rmtree('temp_data')
        
        return pd.DataFrame()


def optimize_dataframe(df):
    """Optimise les types de données pour réduire l'utilisation mémoire"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'object':
            # Convertir les strings en catégories si peu de valeurs uniques
            num_unique = df[col].nunique()
            if num_unique / len(df) < 0.5:  # Si moins de 50% de valeurs uniques
                df[col] = df[col].astype('category')
        
        elif col_type == 'float64':
            # Réduire la précision des floats
            df[col] = df[col].astype('float32')
        
        elif col_type == 'int64':
            # Réduire la taille des ints
            df[col] = df[col].astype('int32')
    
    return df


# =======================================================
# 📈 Compute All KPIs (SIMPLIFIED)
# =======================================================
def compute_all_kpis(df):
    """Calculate KPIs - version simplifiée et robuste"""
    
    def safe_mean(dataframe, column_name):
        """Calcul sécurisé de la moyenne"""
        try:
            if column_name in dataframe.columns:
                values = pd.to_numeric(dataframe[column_name], errors='coerce')
                result = float(values.mean())
                return result if not np.isnan(result) else np.nan
        except:
            pass
        return np.nan
    
    def safe_max(dataframe, column_name):
        """Calcul sécurisé du max"""
        try:
            if column_name in dataframe.columns:
                values = pd.to_numeric(dataframe[column_name], errors='coerce')
                result = float(values.max())
                return result if not np.isnan(result) else np.nan
        except:
            pass
        return np.nan
    
    def safe_count_unique(dataframe, column_name):
        """Compte des valeurs uniques"""
        try:
            if column_name in dataframe.columns:
                return float(dataframe[column_name].nunique())
        except:
            pass
        return np.nan
    
    # KPIs avec fallbacks multiples
    kpis = {}
    
    # Liste de tous les noms de colonnes possibles
    cols = df.columns.tolist()
    
    # Yards
    for col in ['yards_gained', 'yardsGained', 'yards', 'yardage']:
        if col in cols:
            val = safe_mean(df, col)
            if not np.isnan(val):
                kpis['PPE (Yards Gained)'] = val
                break
    
    # Completion
    for col in ['completion_probability', 'completionProbability']:
        if col in cols:
            val = safe_mean(df, col)
            if not np.isnan(val):
                kpis['CBR (Completion Prob)'] = val
                break
    
    # Frame
    for col in ['frame_id', 'frameId', 'frame']:
        if col in cols:
            val = safe_mean(df, col)
            if not np.isnan(val):
                kpis['FFM (Frame ID)'] = val
                break
    
    # Distance
    for col in ['dis', 'distance', 'dist']:
        if col in cols:
            val = safe_mean(df, col)
            if not np.isnan(val):
                kpis['ADY (Distance)'] = val
                break
    
    # Speed
    for col in ['s', 'speed', 'velocity']:
        if col in cols:
            val_mean = safe_mean(df, col)
            val_max = safe_max(df, col)
            if not np.isnan(val_mean):
                kpis['VMC (Speed Avg)'] = val_mean
            if not np.isnan(val_max):
                kpis['SMV (Speed Max)'] = val_max
            break
    
    # Acceleration
    for col in ['a', 'acceleration', 'accel']:
        if col in cols:
            val = safe_mean(df, col)
            if not np.isnan(val):
                kpis['AEF (Acceleration)'] = val
                break
    
    # Direction
    for col in ['dir', 'direction']:
        if col in cols:
            val = safe_mean(df, col)
            if not np.isnan(val):
                kpis['DIR (Direction)'] = val
                break
    
    # Orientation
    for col in ['o', 'orientation']:
        if col in cols:
            val = safe_mean(df, col)
            if not np.isnan(val):
                kpis['ORI (Orientation)'] = val
                break
    
    # Position X
    if 'x' in cols:
        val = safe_mean(df, 'x')
        if not np.isnan(val):
            kpis['Avg X Position'] = val
    
    # Position Y
    if 'y' in cols:
        val = safe_mean(df, 'y')
        if not np.isnan(val):
            kpis['Avg Y Position'] = val
    
    # Unique counts
    for col in ['nflId', 'playerId', 'player_id']:
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
    
    for col in ['gameId', 'game_id']:
        if col in cols:
            val = safe_count_unique(df, col)
            if not np.isnan(val):
                kpis['Unique Games'] = val
                break
    
    # Filtrer les NaN
    kpis = {k: v for k, v in kpis.items() if not np.isnan(v)}
    
    return kpis


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
        'Identifiers': ['gameId', 'playId', 'nflId', 'frameId'],
        'Performance Metrics': ['yards_gained', 'completion_probability'],
        'Player Info': ['displayName', 'jerseyNumber', 'position']
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
