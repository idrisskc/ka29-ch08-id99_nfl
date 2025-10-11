import os
import pandas as pd
import numpy as np

# =======================================================
# 🔹 Chargement des données locales (ou depuis Kaggle)
# =======================================================
def load_data(base_dir="./data"):
    """
    Charge les différents fichiers CSV à partir du répertoire spécifié.
    Retourne 6 DataFrames (pour compatibilité future).
    """
    full_df, df_input, df_out, df_supp, tr_sample, prethrow = [pd.DataFrame()]*6

    # Fichier de données supplémentaires
    supp_path = os.path.join(base_dir, "supplementary_data.csv")
    if os.path.exists(supp_path):
        df_supp = pd.read_csv(supp_path)

    # Exemple de fichier d'entraînement
    train_dir = os.path.join(base_dir, "train")
    if os.path.exists(train_dir):
        csv_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
        if csv_files:
            df_input = pd.read_csv(os.path.join(train_dir, csv_files[0]))
            full_df = df_input.copy()

    return full_df, df_input, df_out, df_supp, tr_sample, prethrow


# =======================================================
# 🔹 Calcul des KPIs agrégés
# =======================================================
def compute_all_kpis_and_aggregate(full_df, df_input, df_out, df_supp, tr_sample, prethrow):
    """
    Calcule les indicateurs de performance (KPIs) et un échantillon de KPIs de passes.
    """
    def safe_mean(df, col):
        """Retourne la moyenne d'une colonne si elle existe, sinon NaN."""
        return float(df[col].mean()) if col in df.columns else np.nan

    # Ensemble de KPIs fictifs (à remplacer avec des vraies colonnes du dataset)
    kpis = {
        'PPE': safe_mean(full_df, 'yards_gained'),             # Points par essai
        'CBR': safe_mean(full_df, 'completion_probability'),   # Taux de complétion
        'FFM': safe_mean(full_df, 'frame_id'),                 # Frame moyenne
        'ADY': safe_mean(full_df, 'distance'),                 # Distance moyenne
        'TDR': safe_mean(full_df, 'time'),                     # Temps moyen
        'PEI': safe_mean(full_df, 'event'),                    # Événement moyen
        'CWE': safe_mean(full_df, 'closest_defender_distance'),# Distance au défenseur
        'EDS': safe_mean(full_df, 'end_speed'),                # Vitesse finale
        'VMC': safe_mean(full_df, 's'),                        # Vitesse max
        'PMA': safe_mean(full_df, 'play_result'),              # Résultat moyen
        'PER': safe_mean(full_df, 'expected_yards'),           # Yards attendus
        'RCI': safe_mean(full_df, 'receiver_id'),              # ID receveur moyen
        'SMV': safe_mean(full_df, 'speed_max'),                # Vitesse max moyenne
        'AEF': safe_mean(full_df, 'acceleration'),             # Accélération moyenne
    }

    # Données simulées pour illustrer le dashboard
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
