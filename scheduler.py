# =======================================================
# scheduler.py - Auto-Refresh Scheduler for Streamlit
# Prevents "goes down" issues with periodic refresh
# =======================================================
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import time
import gc


class StreamlitScheduler:
    """
    Gestionnaire de rafraîchissement automatique pour Streamlit
    Évite les timeouts et maintient la session active
    """
    
    def __init__(self, refresh_interval_minutes=30, memory_cleanup_interval=10):
        self.refresh_interval = refresh_interval_minutes * 60 * 1000  # Convert to ms
        self.memory_cleanup_interval = memory_cleanup_interval * 60  # Convert to seconds
        self.last_cleanup = time.time()
        
    def setup_autorefresh(self, interval_minutes=30, key="autorefresh"):
        """
        Configure le rafraîchissement automatique
        
        Args:
            interval_minutes: Intervalle de rafraîchissement en minutes
            key: Clé unique pour le widget
        """
        interval_ms = interval_minutes * 60 * 1000
        
        # Auto-refresh component
        count = st_autorefresh(
            interval=interval_ms,
            limit=None,  # Pas de limite
            key=key
        )
        
        return count
    
    def check_memory_cleanup(self):
        """Vérifie si un nettoyage mémoire est nécessaire"""
        current_time = time.time()
        
        if current_time - self.last_cleanup > self.memory_cleanup_interval:
            self.cleanup_memory()
            self.last_cleanup = current_time
            return True
        return False
    
    def cleanup_memory(self):
        """Nettoie la mémoire et les caches"""
        with st.spinner("🧹 Memory cleanup..."):
            # Clear Streamlit caches
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            if hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
            
            # Python garbage collection
            gc.collect()
            
            st.success("✅ Memory cleaned", icon="🧹")
    
    def display_session_info(self):
        """Affiche les informations de session"""
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()
        
        session_duration = datetime.now() - st.session_state.session_start
        hours, remainder = divmod(session_duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"⏱️ Session: {hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_next_refresh_time(self, interval_minutes=30):
        """Calcule le prochain temps de rafraîchissement"""
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
        next_refresh = st.session_state.last_refresh + timedelta(minutes=interval_minutes)
        time_until_refresh = next_refresh - datetime.now()
        
        if time_until_refresh.total_seconds() <= 0:
            st.session_state.last_refresh = datetime.now()
            return "Refreshing now..."
        
        minutes_left = int(time_until_refresh.total_seconds() / 60)
        seconds_left = int(time_until_refresh.total_seconds() % 60)
        
        return f"Next refresh in {minutes_left}:{seconds_left:02d}"
    
    def create_status_panel(self, interval_minutes=30):
        """Crée un panneau de statut dans la sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.markdown("### ⚙️ System Status")
            
            # Session duration
            session_info = self.display_session_info()
            st.caption(session_info)
            
            # Next refresh
            refresh_info = self.get_next_refresh_time(interval_minutes)
            st.caption(f"🔄 {refresh_info}")
            
            # Memory cleanup button
            if st.button("🧹 Clean Memory Now", use_container_width=True):
                self.cleanup_memory()
                st.rerun()
    
    def healthcheck(self):
        """Vérifie l'état de santé de l'application"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'memory_cleaned': False
        }
        
        # Check memory cleanup
        if self.check_memory_cleanup():
            health_status['memory_cleaned'] = True
        
        return health_status


# =======================================================
# FONCTIONS HELPER POUR STREAMLIT
# =======================================================
@st.cache_resource
def get_scheduler(refresh_interval=30, memory_cleanup_interval=10):
    """Singleton pour le scheduler"""
    return StreamlitScheduler(refresh_interval, memory_cleanup_interval)


def setup_dashboard_scheduler(
    refresh_interval_minutes=30,
    show_status=True,
    enable_autorefresh=True
):
    """
    Configure le scheduler pour le dashboard
    
    Args:
        refresh_interval_minutes: Intervalle de rafraîchissement en minutes
        show_status: Afficher le panneau de statut
        enable_autorefresh: Activer le rafraîchissement automatique
    
    Returns:
        scheduler: Instance du scheduler
        refresh_count: Nombre de rafraîchissements
    """
    scheduler = get_scheduler(
        refresh_interval=refresh_interval_minutes,
        memory_cleanup_interval=10
    )
    
    refresh_count = 0
    
    # Setup auto-refresh si activé
    if enable_autorefresh:
        refresh_count = scheduler.setup_autorefresh(
            interval_minutes=refresh_interval_minutes,
            key="dashboard_refresh"
        )
    
    # Afficher le statut si demandé
    if show_status:
        scheduler.create_status_panel(refresh_interval_minutes)
    
    # Vérifier le nettoyage mémoire
    scheduler.check_memory_cleanup()
    
    return scheduler, refresh_count


def add_keepalive_ping():
    """
    Ajoute un ping invisible pour maintenir la connexion
    Utilise un composant HTML avec meta refresh
    """
    st.markdown(
        """
        <meta http-equiv="refresh" content="1800">
        <script>
            // Ping every 5 minutes to keep connection alive
            setInterval(function() {
                fetch(window.location.href)
                    .then(response => console.log('Keepalive ping'))
                    .catch(error => console.log('Ping failed:', error));
            }, 300000); // 5 minutes
        </script>
        """,
        unsafe_allow_html=True
    )


def initialize_session_state():
    """Initialise les variables de session nécessaires"""
    if 'session_start' not in st.session_state:
        st.session_state.session_start = datetime.now()
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False


# =======================================================
# EXEMPLE D'UTILISATION
# =======================================================
def example_usage():
    """
    Exemple d'utilisation du scheduler dans un app Streamlit
    """
    st.title("NFL Dashboard with Auto-Refresh")
    
    # Initialiser la session
    initialize_session_state()
    
    # Setup scheduler (30 min refresh)
    scheduler, refresh_count = setup_dashboard_scheduler(
        refresh_interval_minutes=30,
        show_status=True,
        enable_autorefresh=True
    )
    
    # Ajouter keepalive
    add_keepalive_ping()
    
    # Afficher le compteur de refresh
    if refresh_count > 0:
        st.info(f"🔄 Dashboard refreshed {refresh_count} times")
        st.session_state.refresh_count = refresh_count
    
    # Votre contenu dashboard ici
    st.write("Dashboard content...")
    
    # Health check en bas de page
    with st.expander("🏥 System Health"):
        health = scheduler.healthcheck()
        st.json(health)
