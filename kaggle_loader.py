# =======================================================
# kaggle_loader.py - Kaggle Authentication & Data Loading
# Integrated with Streamlit for NFL Big Data Bowl 2026
# =======================================================
import os
import sys
import zipfile
import json
import shutil
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict


class KaggleDataLoader:
    """Gestionnaire d'authentification et chargement Kaggle"""
    
    def __init__(self, competition="nfl-big-data-bowl-2026-analytics"):
        self.competition = competition
        self.base_dir = Path("114239_nfl_competition_files_published_analytics_final")
        self.kaggle_api = None
        self.authenticated = False
        
    def setup_kaggle_credentials(self, username: str, api_key: str) -> bool:
        """
        Configure les credentials Kaggle
        
        Args:
            username: Kaggle username
            api_key: Kaggle API key
            
        Returns:
            bool: True si succès
        """
        try:
            # Set environment variables
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = api_key
            
            # Create .kaggle directory
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)
            
            # Write kaggle.json
            kaggle_json = kaggle_dir / "kaggle.json"
            credentials = {
                "username": username,
                "key": api_key
            }
            
            with open(kaggle_json, 'w') as f:
                json.dump(credentials, f)
            
            # Set permissions (Unix/Mac)
            if sys.platform != 'win32':
                os.chmod(kaggle_json, 0o600)
            
            st.success("✅ Kaggle credentials configured")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to setup credentials: {e}")
            return False
    
    def authenticate(self, username: str = None, api_key: str = None) -> bool:
        """
        Authentifie avec l'API Kaggle
        
        Args:
            username: Kaggle username (optionnel si dans secrets)
            api_key: Kaggle API key (optionnel si dans secrets)
            
        Returns:
            bool: True si authentifié
        """
        try:
            # Try to get from Streamlit secrets first
            if username is None or api_key is None:
                try:
                    username = st.secrets.get("KAGGLE_USERNAME", username)
                    api_key = st.secrets.get("KAGGLE_KEY", api_key)
                except:
                    pass
            
            # Setup credentials
            if username and api_key:
                if not self.setup_kaggle_credentials(username, api_key):
                    return False
            
            # Import and authenticate
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
            
            self.authenticated = True
            st.success("✅ Kaggle API authenticated")
            return True
            
        except ImportError:
            st.error("❌ Kaggle package not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            st.error(f"❌ Authentication failed: {e}")
            return False
    
    def list_competition_files(self) -> List[str]:
        """Liste les fichiers disponibles dans la compétition"""
        if not self.authenticated:
            st.warning("⚠️ Not authenticated")
            return []
        
        try:
            with st.spinner("📋 Listing competition files..."):
                files = self.kaggle_api.competition_list_files(self.competition)
                file_list = [f.name for f in files]
                
            st.success(f"✅ Found {len(file_list)} files")
            return file_list
            
        except Exception as e:
            st.error(f"❌ Failed to list files: {e}")
            return []
    
    def download_competition_data(self, force_download: bool = False) -> bool:
        """
        Télécharge les données de la compétition
        
        Args:
            force_download: Force le téléchargement même si déjà présent
            
        Returns:
            bool: True si succès
        """
        if not self.authenticated:
            st.warning("⚠️ Not authenticated")
            return False
        
        try:
            # Create base directory
            self.base_dir.mkdir(exist_ok=True)
            
            # Check if already downloaded
            zip_path = self.base_dir / f"{self.competition}.zip"
            
            if zip_path.exists() and not force_download:
                st.info(f"📦 Data already downloaded: {zip_path}")
                return True
            
            # Download
            with st.spinner(f"📥 Downloading from Kaggle (this may take several minutes)..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                self.kaggle_api.competition_download_files(
                    self.competition,
                    path=str(self.base_dir),
                    force=force_download,
                    quiet=False
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Download complete")
            
            # Verify download
            if zip_path.exists():
                size_mb = zip_path.stat().st_size / (1024 ** 2)
                st.success(f"✅ Downloaded: {size_mb:.2f} MB")
                return True
            else:
                st.error("❌ Download failed - file not found")
                return False
                
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            return False
    
    def extract_data(self, force_extract: bool = False) -> bool:
        """
        Extrait les fichiers téléchargés
        
        Args:
            force_extract: Force l'extraction même si déjà fait
            
        Returns:
            bool: True si succès
        """
        try:
            zip_path = self.base_dir / f"{self.competition}.zip"
            
            if not zip_path.exists():
                st.error(f"❌ ZIP file not found: {zip_path}")
                return False
            
            # Check if already extracted
            extracted_marker = self.base_dir / ".extracted"
            if extracted_marker.exists() and not force_extract:
                st.info("📂 Data already extracted")
                return True
            
            # Extract
            with st.spinner("📦 Extracting files..."):
                progress_bar = st.progress(0)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    total_files = len(file_list)
                    
                    for i, file in enumerate(file_list):
                        zip_ref.extract(file, self.base_dir)
                        if i % 10 == 0:
                            progress_bar.progress((i + 1) / total_files)
                
                progress_bar.progress(1.0)
            
            # Create marker file
            extracted_marker.touch()
            
            st.success("✅ Extraction complete")
            return True
            
        except Exception as e:
            st.error(f"❌ Extraction failed: {e}")
            return False
    
    def get_data_structure(self) -> Dict:
        """
        Explore la structure des données téléchargées
        
        Returns:
            Dict: Structure des fichiers
        """
        structure = {
            'directories': [],
            'files': [],
            'total_size_mb': 0
        }
        
        try:
            for root, dirs, files in os.walk(self.base_dir):
                level = str(root).replace(str(self.base_dir), "").count(os.sep)
                
                for d in dirs:
                    structure['directories'].append({
                        'path': str(Path(root) / d),
                        'level': level
                    })
                
                for f in files:
                    file_path = Path(root) / f
                    try:
                        size_mb = file_path.stat().st_size / (1024 ** 2)
                        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        structure['files'].append({
                            'name': f,
                            'path': str(file_path),
                            'size_mb': size_mb,
                            'modified': mod_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'level': level
                        })
                        
                        structure['total_size_mb'] += size_mb
                    except:
                        pass
            
            return structure
            
        except Exception as e:
            st.error(f"❌ Failed to get structure: {e}")
            return structure
    
    def find_csv_files(self, pattern: str = "*.csv") -> List[Path]:
        """
        Trouve tous les fichiers CSV dans les données
        
        Args:
            pattern: Pattern de recherche (ex: "*.csv", "train/*.csv")
            
        Returns:
            List[Path]: Liste des chemins CSV
        """
        csv_files = list(self.base_dir.rglob(pattern))
        return sorted(csv_files)
    
    def load_csv_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Charge un fichier CSV avec optimisations
        
        Args:
            file_path: Chemin du fichier
            **kwargs: Arguments pour pd.read_csv
            
        Returns:
            DataFrame: Données chargées
        """
        try:
            with st.spinner(f"Loading {file_path.name}..."):
                df = pd.read_csv(file_path, **kwargs)
            
            st.success(f"✅ Loaded {len(df):,} rows from {file_path.name}")
            return df
            
        except Exception as e:
            st.error(f"❌ Failed to load {file_path.name}: {e}")
            return pd.DataFrame()
    
    def cleanup(self, remove_zip: bool = False):
        """
        Nettoie les fichiers temporaires
        
        Args:
            remove_zip: Supprimer aussi le ZIP téléchargé
        """
        try:
            if remove_zip:
                zip_path = self.base_dir / f"{self.competition}.zip"
                if zip_path.exists():
                    zip_path.unlink()
                    st.success("🗑️ ZIP file removed")
            
            # Remove marker
            marker = self.base_dir / ".extracted"
            if marker.exists():
                marker.unlink()
            
            st.success("🧹 Cleanup complete")
            
        except Exception as e:
            st.warning(f"⚠️ Cleanup warning: {e}")


# =======================================================
# FONCTIONS HELPER POUR STREAMLIT
# =======================================================

@st.cache_resource
def get_kaggle_loader(competition="nfl-big-data-bowl-2026-analytics"):
    """Singleton pour le Kaggle loader"""
    return KaggleDataLoader(competition)


def kaggle_authentication_ui():
    """
    Interface Streamlit pour l'authentification Kaggle
    
    Returns:
        tuple: (authenticated, loader)
    """
    st.sidebar.markdown("### 🔐 Kaggle Authentication")
    
    loader = get_kaggle_loader()
    
    # Try to get from secrets first
    try:
        username = st.secrets.get("KAGGLE_USERNAME", "")
        api_key = st.secrets.get("KAGGLE_KEY", "")
        use_secrets = True
    except:
        username = ""
        api_key = ""
        use_secrets = False
    
    # Show input fields if not in secrets
    if not use_secrets or not username or not api_key:
        with st.sidebar.expander("📝 Enter Credentials", expanded=not loader.authenticated):
            username = st.text_input(
                "Kaggle Username",
                value=username,
                help="Your Kaggle username"
            )
            api_key = st.text_input(
                "Kaggle API Key",
                value=api_key,
                type="password",
                help="Get it from kaggle.com/account"
            )
    
    # Authenticate button
    if st.sidebar.button("🔑 Authenticate", type="primary", disabled=loader.authenticated):
        if username and api_key:
            if loader.authenticate(username, api_key):
                st.sidebar.success("✅ Authenticated")
                st.rerun()
        else:
            st.sidebar.error("❌ Please enter both username and API key")
    
    # Show status
    if loader.authenticated:
        st.sidebar.success("✅ Kaggle API Connected")
    else:
        st.sidebar.warning("⚠️ Not authenticated")
    
    return loader.authenticated, loader


def kaggle_data_download_ui(loader: KaggleDataLoader):
    """
    Interface Streamlit pour télécharger les données
    
    Args:
        loader: KaggleDataLoader instance
    """
    st.sidebar.markdown("### 📥 Data Download")
    
    # List files button
    if st.sidebar.button("📋 List Competition Files"):
        files = loader.list_competition_files()
        if files:
            with st.sidebar.expander("📁 Available Files", expanded=True):
                for f in files:
                    st.text(f"📄 {f}")
    
    # Download options
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        force_download = st.checkbox("Force Download", value=False)
    
    with col2:
        force_extract = st.checkbox("Force Extract", value=False)
    
    # Download button
    if st.sidebar.button("📥 Download & Extract Data", type="primary"):
        # Download
        if loader.download_competition_data(force_download):
            # Extract
            if loader.extract_data(force_extract):
                st.sidebar.success("✅ Data ready!")
                
                # Show structure
                with st.sidebar.expander("📂 Data Structure"):
                    structure = loader.get_data_structure()
                    st.metric("Total Files", len(structure['files']))
                    st.metric("Total Size", f"{structure['total_size_mb']:.2f} MB")
                
                st.rerun()


def display_kaggle_data_explorer(loader: KaggleDataLoader):
    """
    Affiche un explorateur de données Kaggle
    
    Args:
        loader: KaggleDataLoader instance
    """
    st.markdown("## 📂 Kaggle Data Explorer")
    
    # Get structure
    structure = loader.get_data_structure()
    
    if not structure['files']:
        st.info("📭 No data downloaded yet. Use the sidebar to download.")
        return
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files", len(structure['files']))
    with col2:
        st.metric("Total Size", f"{structure['total_size_mb']:.2f} MB")
    with col3:
        csv_count = sum(1 for f in structure['files'] if f['name'].endswith('.csv'))
        st.metric("CSV Files", csv_count)
    
    # File list
    st.markdown("### 📄 Files")
    
    # Convert to DataFrame
    files_df = pd.DataFrame(structure['files'])
    
    if not files_df.empty:
        # Filter options
        col1, col2 = st.columns([3, 1])
        with col1:
            file_filter = st.text_input("🔍 Filter files", placeholder="e.g., train, .csv")
        with col2:
            sort_by = st.selectbox("Sort by", ["name", "size_mb", "modified"])
        
        # Apply filter
        if file_filter:
            files_df = files_df[files_df['name'].str.contains(file_filter, case=False)]
        
        # Sort
        files_df = files_df.sort_values(sort_by, ascending=False if sort_by == "size_mb" else True)
        
        # Display
        st.dataframe(
            files_df[['name', 'size_mb', 'modified']],
            use_container_width=True,
            height=400
        )
        
        # Download selected file
        if not files_df.empty:
            selected_file = st.selectbox(
                "Select file to preview",
                files_df['name'].tolist()
            )
            
            if st.button(f"📊 Preview {selected_file}"):
                file_path = files_df[files_df['name'] == selected_file]['path'].iloc[0]
                
                if selected_file.endswith('.csv'):
                    df = loader.load_csv_file(Path(file_path), nrows=1000)
                    if not df.empty:
                        st.dataframe(df.head(100), use_container_width=True)
                        
                        st.markdown("#### 📊 Quick Stats")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows (preview)", len(df))
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
