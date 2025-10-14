# =======================================================
# strategic_kpis.py - Advanced KPIs for NFL Strategy
# =======================================================
import pandas as pd
import numpy as np
from scipy.stats import entropy


def calculate_defensive_pressure_index(df):
    """
    KPI 1: Defensive Pressure Index (DPI)
    Mesure l'intensité de la pression défensive
    """
    try:
        # Filtrer les joueurs défensifs
        defense_df = df[df['player_side'] == 'Defense'].copy()
        
        # Grouper par play
        dpi_data = defense_df.groupby(['game_id', 'play_id']).agg({
            'defenders_in_the_box': 'first',
            's': 'mean',  # Vitesse moyenne défenseurs
            'dropback_distance': 'first'
        }).reset_index()
        
        # Calculer DPI
        dpi_data['DPI'] = (
            dpi_data['defenders_in_the_box'] * 
            dpi_data['s']
        ) / (dpi_data['dropback_distance'] + 1)  # +1 pour éviter division par 0
        
        return dpi_data[['game_id', 'play_id', 'DPI']]
    
    except Exception as e:
        print(f"Error calculating DPI: {e}")
        return pd.DataFrame()


def calculate_route_efficiency_score(df):
    """
    KPI 2: Route Efficiency Score (RES)
    Efficacité des routes en fonction de la distance parcourue
    """
    try:
        # Filtrer les targeted receivers
        receivers_df = df[df['player_role'] == 'Targeted Receiver'].copy()
        
        # Calculer distance parcourue
        receivers_df['distance_traveled'] = np.sqrt(
            receivers_df.groupby(['game_id', 'play_id', 'nfl_id'])['x'].diff()**2 +
            receivers_df.groupby(['game_id', 'play_id', 'nfl_id'])['y'].diff()**2
        )
        
        # Agréger par route
        res_data = receivers_df.groupby(['game_id', 'play_id', 'route_of_targeted_receiver']).agg({
            'distance_traveled': 'sum',
            'ball_land_x': 'first',
            'ball_land_y': 'first',
            'x': 'first',
            'y': 'first',
            'pass_result': 'first'
        }).reset_index()
        
        # Distance de la balle
        res_data['ball_distance'] = np.sqrt(
            (res_data['ball_land_x'] - res_data['x'])**2 +
            (res_data['ball_land_y'] - res_data['y'])**2
        )
        
        # Completion rate (1 si Complete, 0 sinon)
        res_data['completion'] = (res_data['pass_result'] == 'C').astype(int)
        
        # RES
        res_data['RES'] = (
            res_data['ball_distance'] / 
            (res_data['distance_traveled'] + 1)
        ) * res_data['completion']
        
        return res_data[['game_id', 'play_id', 'route_of_targeted_receiver', 'RES']]
    
    except Exception as e:
        print(f"Error calculating RES: {e}")
        return pd.DataFrame()


def calculate_coverage_vulnerability_matrix(df):
    """
    KPI 3: Coverage Vulnerability Matrix (CVM)
    Vulnérabilités par type de coverage
    """
    try:
        cvm_data = df.groupby(['team_coverage_type', 'receiver_alignment']).agg({
            'yards_gained': 'mean',
            'expected_points': 'mean',
            'pass_result': 'count'
        }).reset_index()
        
        cvm_data.columns = ['coverage_type', 'receiver_alignment', 'avg_yards', 'avg_ep', 'plays']
        
        # CVM Score
        cvm_data['CVM'] = cvm_data['avg_yards'] / (cvm_data['avg_ep'] + 0.1)
        
        return cvm_data
    
    except Exception as e:
        print(f"Error calculating CVM: {e}")
        return pd.DataFrame()


def calculate_pass_timing_optimization_index(df):
    """
    KPI 4: Pass Timing Optimization Index (PTOI)
    Timing optimal du throw
    """
    try:
        # Prendre le frame du throw (dernier frame input)
        ptoi_data = df.groupby(['game_id', 'play_id']).agg({
            'frame_id': 'max',
            'pass_result': 'first'
        }).reset_index()
        
        ptoi_data.columns = ['game_id', 'play_id', 'throw_frame', 'pass_result']
        
        # Success (1 si Complete, 0.5 si autre, 0 si Sack/INT)
        ptoi_data['success'] = ptoi_data['pass_result'].map({
            'C': 1.0,
            'I': 0.5,
            'S': 0.0,
            'IN': 0.0,
            'R': 0.7
        }).fillna(0.5)
        
        # Fenêtre optimale (hypothèse: 15-25 frames)
        ptoi_data['optimal_window'] = 20
        ptoi_data['PTOI'] = (
            ptoi_data['optimal_window'] / 
            (ptoi_data['throw_frame'] + 1)
        ) * ptoi_data['success']
        
        return ptoi_data[['game_id', 'play_id', 'throw_frame', 'PTOI']]
    
    except Exception as e:
        print(f"Error calculating PTOI: {e}")
        return pd.DataFrame()


def calculate_spatial_advantage_score(df):
    """
    KPI 5: Spatial Advantage Score (SAS)
    Avantage spatial receiver vs defender
    """
    try:
        # Targeted receivers
        receivers = df[df['player_role'] == 'Targeted Receiver'][['game_id', 'play_id', 'x', 'y']].copy()
        receivers.columns = ['game_id', 'play_id', 'rx', 'ry']
        
        # Defensive coverage
        defenders = df[df['player_role'] == 'Defensive Coverage'][['game_id', 'play_id', 'x', 'y']].copy()
        
        # Joindre
        merged = receivers.merge(defenders, on=['game_id', 'play_id'], how='left')
        
        # Distance receiver-defender
        merged['separation'] = np.sqrt(
            (merged['rx'] - merged['x'])**2 +
            (merged['ry'] - merged['y'])**2
        )
        
        # Agréger
        sas_data = merged.groupby(['game_id', 'play_id']).agg({
            'separation': 'mean'
        }).reset_index()
        
        # SAS (normalisé 0-100)
        sas_data['SAS'] = (sas_data['separation'] / 53.3) * 100  # 53.3 = largeur terrain
        
        return sas_data
    
    except Exception as e:
        print(f"Error calculating SAS: {e}")
        return pd.DataFrame()


def calculate_formation_predictability_index(df):
    """
    KPI 6: Formation Predictability Index (FPI)
    Mesure la prévisibilité via entropie de Shannon
    """
    try:
        fpi_data = df.groupby(['offense_formation', 'receiver_alignment'])['pass_result'].apply(
            lambda x: entropy(x.value_counts(normalize=True))
        ).reset_index()
        
        fpi_data.columns = ['formation', 'alignment', 'entropy']
        
        # FPI (plus l'entropie est haute, moins c'est prévisible)
        # Inverser pour avoir: FPI élevé = prévisible
        fpi_data['FPI'] = 1 / (fpi_data['entropy'] + 0.1)
        
        return fpi_data
    
    except Exception as e:
        print(f"Error calculating FPI: {e}")
        return pd.DataFrame()


def calculate_win_probability_leverage(df):
    """
    KPI 7: Win Probability Leverage (WPL)
    Impact sur la win probability
    """
    try:
        wpl_data = df.groupby(['game_id', 'play_id', 'down', 'quarter']).agg({
            'home_team_win_probability_added': 'first',
            'yards_to_go': 'first'
        }).reset_index()
        
        # Weight par situation (4th down = 3x, red zone = 2x)
        wpl_data['situation_weight'] = 1
        wpl_data.loc[wpl_data['down'] == 4, 'situation_weight'] = 3
        wpl_data.loc[wpl_data['yards_to_go'] <= 20, 'situation_weight'] *= 2
        
        # WPL
        wpl_data['WPL'] = abs(wpl_data['home_team_win_probability_added']) * wpl_data['situation_weight']
        
        return wpl_data[['game_id', 'play_id', 'down', 'quarter', 'WPL']]
    
    except Exception as e:
        print(f"Error calculating WPL: {e}")
        return pd.DataFrame()


def calculate_defensive_reaction_time(df):
    """
    KPI 8: Defensive Reaction Time (DRT)
    Vitesse de réaction défensive post-snap
    """
    try:
        defense_df = df[df['player_side'] == 'Defense'].copy()
        
        # Changement de direction entre frames
        defense_df['dir_change'] = defense_df.groupby(['game_id', 'play_id', 'nfl_id'])['dir'].diff().abs()
        
        # DRT par coverage
        drt_data = defense_df.groupby(['team_coverage_type']).agg({
            'dir_change': 'mean',
            's': 'mean'
        }).reset_index()
        
        drt_data.columns = ['coverage_type', 'avg_dir_change', 'avg_speed']
        
        # DRT (changement rapide = bon)
        drt_data['DRT'] = drt_data['avg_dir_change'] * drt_data['avg_speed']
        
        return drt_data
    
    except Exception as e:
        print(f"Error calculating DRT: {e}")
        return pd.DataFrame()


def calculate_red_zone_conversion_efficiency(df):
    """
    KPI 9: Red Zone Conversion Efficiency (RZCE)
    Performance en red zone
    """
    try:
        # Filtrer red zone
        rz_df = df[df['absolute_yardline_number'] <= 20].copy()
        
        rzce_data = rz_df.groupby(['offense_formation', 'down']).agg({
            'pass_result': lambda x: (x == 'C').sum() / len(x),  # Completion %
            'yards_gained': 'mean',
            'play_id': 'count'
        }).reset_index()
        
        rzce_data.columns = ['formation', 'down', 'completion_rate', 'avg_yards', 'attempts']
        
        # RZCE
        rzce_data['RZCE'] = rzce_data['completion_rate'] * rzce_data['avg_yards']
        
        return rzce_data
    
    except Exception as e:
        print(f"Error calculating RZCE: {e}")
        return pd.DataFrame()


def calculate_tempo_impact_score(df):
    """
    KPI 10: Tempo Impact Score (TIS)
    Impact du tempo sur la production
    """
    try:
        # Convertir game_clock en secondes
        def clock_to_seconds(clock_str):
            try:
                m, s = map(int, str(clock_str).split(':'))
                return m * 60 + s
            except:
                return 0
        
        df['clock_seconds'] = df['game_clock'].apply(clock_to_seconds)
        
        # Tempo = plays per minute
        tis_data = df.groupby(['game_id', 'quarter']).agg({
            'play_id': 'count',
            'clock_seconds': lambda x: x.max() - x.min(),
            'yards_gained': 'mean',
            'expected_points_added': 'mean'
        }).reset_index()
        
        tis_data.columns = ['game_id', 'quarter', 'plays', 'time_elapsed', 'avg_yards', 'avg_epa']
        
        # TIS
        tis_data['tempo'] = tis_data['plays'] / (tis_data['time_elapsed'] / 60 + 0.1)
        tis_data['TIS'] = tis_data['tempo'] * tis_data['avg_epa']
        
        return tis_data[['game_id', 'quarter', 'tempo', 'TIS']]
    
    except Exception as e:
        print(f"Error calculating TIS: {e}")
        return pd.DataFrame()


# =======================================================
# Master Function: Calculate All Strategic KPIs
# =======================================================
def calculate_all_strategic_kpis(df):
    """
    Calcule tous les KPIs stratégiques
    Retourne un dictionnaire de DataFrames
    """
    kpis = {}
    
    print("📊 Calculating Strategic KPIs...")
    
    try:
        kpis['DPI'] = calculate_defensive_pressure_index(df)
        print("✅ DPI calculated")
    except:
        print("❌ DPI failed")
    
    try:
        kpis['RES'] = calculate_route_efficiency_score(df)
        print("✅ RES calculated")
    except:
        print("❌ RES failed")
    
    try:
        kpis['CVM'] = calculate_coverage_vulnerability_matrix(df)
        print("✅ CVM calculated")
    except:
        print("❌ CVM failed")
    
    try:
        kpis['PTOI'] = calculate_pass_timing_optimization_index(df)
        print("✅ PTOI calculated")
    except:
        print("❌ PTOI failed")
    
    try:
        kpis['SAS'] = calculate_spatial_advantage_score(df)
        print("✅ SAS calculated")
    except:
        print("❌ SAS failed")
    
    try:
        kpis['FPI'] = calculate_formation_predictability_index(df)
        print("✅ FPI calculated")
    except:
        print("❌ FPI failed")
    
    try:
        kpis['WPL'] = calculate_win_probability_leverage(df)
        print("✅ WPL calculated")
    except:
        print("❌ WPL failed")
    
    try:
        kpis['DRT'] = calculate_defensive_reaction_time(df)
        print("✅ DRT calculated")
    except:
        print("❌ DRT failed")
    
    try:
        kpis['RZCE'] = calculate_red_zone_conversion_efficiency(df)
        print("✅ RZCE calculated")
    except:
        print("❌ RZCE failed")
    
    try:
        kpis['TIS'] = calculate_tempo_impact_score(df)
        print("✅ TIS calculated")
    except:
        print("❌ TIS failed")
    
    print(f"\n🎉 Successfully calculated {len(kpis)} strategic KPIs")
    
    return kpis
