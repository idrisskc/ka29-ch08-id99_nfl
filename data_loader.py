# =======================================================
# data_loader.py - Optimized Data Loading for Large Files
# Handles 700MB+ files with chunking and caching
# =======================================================
import pandas as pd
import numpy as np
import streamlit as st
import os
import gc
import psutil
from pathlib import Path
import dask.dataframe as dd
from datetime import datetime, timedelta
import hashlib


class OptimizedDataLoader:
    """Gestionnaire de chargement optimisé pour gros fichiers"""
    
    def __init__(self, cache_dir=".cache", chunk_size=50000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.memory_threshold = 0.85  # 85% de RAM max
        
    def get_memory_usage(self):
        """Retourne l'utilisation mémoire actuelle"""
        return psutil.virtual_memory().percent / 100
    
    def get_file_hash(self, filepath):
        """Génère un hash unique pour le fichier"""
        file_stat = os.stat(filepath)
        hash_str = f"{filepath}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def get_cache_path(self, filepath, suffix=""):
        """Retourne le chemin du cache pour un fichier"""
        file_hash = self.get_file_hash(filepath)
        return self.cache_dir / f"{file_hash}{suffix}.parquet"
    
    def should_use_cache(self, filepath):
        """Vérifie si le cache existe et est valide"""
        cache_path = self.get_cache_path(filepath)
        if not cache_path.exists():
            return False
        
        # Vérifier si le cache est récent (< 1 jour)
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < timedelta(days=1)
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_with_dask(_self, filepath, columns=None, sample_frac=None):
        """Charge un gros fichier avec Dask (streaming)"""
        try:
            # Lire avec Dask pour éviter de tout charger en mémoire
            ddf = dd.read_csv(
                filepath,
                usecols=columns,
                dtype_backend='numpy_nullable',
                blocksize='64MB'
            )
            
            # Échantillonnage si demandé
            if sample_frac and sample_frac < 1.0:
                ddf = ddf.sample(frac=sample_frac, random_state=42)
            
            # Conversion en Pandas avec limite mémoire
            with st.spinner(f"Loading {os.path.basename(filepath)}..."):
                df = ddf.compute()
            
            return df
            
        except Exception as e:
            st.error(f"Dask loading failed: {e}")
            return None
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_with_chunks(_self, filepath, columns=None, max_rows=None):
        """Charge par chunks pour contrôle mémoire"""
        chunks = []
        rows_loaded = 0
        
        try:
            # Lire le header pour détecter les colonnes disponibles
            header_df = pd.read_csv(filepath, nrows=0)
            available_cols = header_df.columns.tolist()
            
            if columns:
                cols_to_load = [c for c in columns if c in available_cols]
            else:
                cols_to_load = available_cols
            
            # Charger par chunks
            chunk_iterator = pd.read_csv(
                filepath,
                usecols=cols_to_load,
                chunksize=_self.chunk_size,
                low_memory=False
            )
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(chunk_iterator):
                # Vérifier la mémoire avant de continuer
                if _self.get_memory_usage() > _self.memory_threshold:
                    st.warning(f"⚠️ Memory limit reached. Loaded {rows_loaded:,} rows")
                    break
                
                chunks.append(chunk)
                rows_loaded += len(chunk)
                
                # Limiter le nombre de lignes si spécifié
                if max_rows and rows_loaded >= max_rows:
                    break
                
                # Update progress
                if i % 5 == 0:
                    progress_bar.progress(min(rows_loaded / (max_rows or 1000000), 1.0))
                    status_text.text(f"Loaded {rows_loaded:,} rows...")
            
            progress_bar.empty()
            status_text.empty()
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Chunk loading failed: {e}")
            return pd.DataFrame()
    
    def save_to_cache(self, df, filepath):
        """Sauvegarde dans le cache Parquet (plus rapide)"""
        try:
            cache_path = self.get_cache_path(filepath)
            df.to_parquet(cache_path, engine='pyarrow', compression='snappy')
            return True
        except Exception as e:
            st.warning(f"Cache save failed: {e}")
            return False
    
    def load_from_cache(self, filepath):
        """Charge depuis le cache Parquet"""
        try:
            cache_path = self.get_cache_path(filepath)
            if cache_path.exists():
                return pd.read_parquet(cache_path, engine='pyarrow')
            return None
        except Exception as e:
            st.warning(f"Cache load failed: {e}")
            return None
    
    def smart_load(self, filepath, columns=None, max_rows=None, use_cache=True):
        """
        Chargement intelligent avec stratégie adaptative
        - Vérifie le cache d'abord
        - Utilise Dask pour files > 500MB
        - Chunking pour files < 500MB
        """
        file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
        
        # Essayer le cache d'abord
        if use_cache and self.should_use_cache(filepath):
            with st.spinner("Loading from cache..."):
                cached_df = self.load_from_cache(filepath)
                if cached_df is not None:
                    st.success(f"✅ Loaded from cache: {len(cached_df):,} rows")
                    return cached_df
        
        # Stratégie selon la taille
        if file_size_mb > 500:
            st.info(f"📊 Large file detected ({file_size_mb:.1f} MB). Using Dask streaming...")
            sample_frac = min(1.0, 500 / file_size_mb) if max_rows is None else None
            df = self.load_with_dask(filepath, columns, sample_frac)
        else:
            st.info(f"📊 Loading file ({file_size_mb:.1f} MB) with chunking...")
            df = self.load_with_chunks(filepath, columns, max_rows)
        
        # Sauvegarder dans le cache si réussi
        if df is not None and not df.empty and use_cache:
            self.save_to_cache(df, filepath)
        
        return df
    
    def optimize_dataframe(self, df):
        """Optimise la mémoire du DataFrame"""
        initial_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                num_unique = df[col].nunique()
                if num_unique / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            
            elif col_type == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            elif col_type == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        final_mem = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (1 - final_mem / initial_mem) * 100
        
        if reduction > 10:
            st.success(f"💾 Memory optimized: {initial_mem:.1f}MB → {final_mem:.1f}MB ({reduction:.1f}% reduction)")
        
        return df
    
    def clear_cache(self):
        """Nettoie le cache"""
        try:
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            st.success("🧹 Cache cleared")
        except Exception as e:
            st.error(f"Cache clear failed: {e}")


# =======================================================
# FONCTION HELPER POUR STREAMLIT
# =======================================================
@st.cache_resource
def get_data_loader():
    """Singleton pour le data loader"""
    return OptimizedDataLoader()


def load_nfl_data(filepath, columns=None, max_rows=None, use_cache=True):
    """Interface simplifiée pour charger des données NFL"""
    loader = get_data_loader()
    
    with st.spinner(f"Loading {os.path.basename(filepath)}..."):
        df = loader.smart_load(filepath, columns, max_rows, use_cache)
        
        if df is not None and not df.empty:
            # Optimiser la mémoire
            df = loader.optimize_dataframe(df)
            
            # Afficher les stats
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.info(f"📊 Loaded: {len(df):,} rows × {len(df.columns)} columns ({memory_mb:.1f} MB)")
            
            return df
        else:
            st.error("❌ Failed to load data")
            return pd.DataFrame()


def clear_all_caches():
    """Nettoie tous les caches Streamlit + fichiers"""
    st.cache_data.clear()
    st.cache_resource.clear()
    
    loader = get_data_loader()
    loader.clear_cache()
    
    gc.collect()
    st.success("🧹 All caches cleared")
