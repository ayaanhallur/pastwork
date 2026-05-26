
import numpy as np
import pandas as pd
from scipy.special import expit
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class UltimateGlickoHockey:
    """
    Comprehensive Glicko-2 system with 18 components
    """
    
    def __init__(self, tau=0.5):
        self.tau = tau
        self.epsilon = 0.000001
        
        # Initial ratings
        self.initial_rating = 1500
        self.initial_rd = 350
        self.initial_volatility = 0.06
        
        # RATINGS
        self.even_strength_offense = defaultdict(lambda: self._init_rating())
        self.even_strength_defense = defaultdict(lambda: self._init_rating())
        self.goalie_ratings = defaultdict(lambda: self._init_rating())
        self.power_play_offense = defaultdict(lambda: self._init_rating())
        self.penalty_kill_defense = defaultdict(lambda: self._init_rating())
        self.penalty_discipline = defaultdict(lambda: self._init_rating())
        self.shooting_efficiency = defaultdict(lambda: self._init_rating())
        self.empty_net_performance = defaultdict(lambda: self._init_rating())
        self.line_depth_quality = defaultdict(lambda: self._init_rating())
        self.ot_clutch_performance = defaultdict(lambda: self._init_rating())
        self.shot_quality_differential = defaultdict(lambda: self._init_rating())
        self.pdo_sustainability = defaultdict(lambda: self._init_rating())
        self.playmaking_ability = defaultdict(lambda: self._init_rating())
        self.offensive_pressure = defaultdict(lambda: self._init_rating())
        self.xg_finishing_skill = defaultdict(lambda: self._init_rating())
        self.home_away_consistency = defaultdict(lambda: self._init_rating())
        self.performance_consistency = defaultdict(lambda: self._init_rating())
        self.st_goal_impact = defaultdict(lambda: self._init_rating())
        self.high_danger_creation = defaultdict(lambda: self._init_rating())
        
        # Statistics tracking
        self.team_stats = defaultdict(lambda: {
            'games': 0,
            'home_games': 0,
            'away_games': 0,
            'goals': 0,
            'goals_against': 0,
            'shots': 0,
            'shots_against': 0,
            'xg': 0,
            'xg_against': 0,
            'assists': 0,
            'max_xg_total': 0,
            'ot_games': 0,
            'ot_wins': 0,
            'home_xg': 0,
            'away_xg': 0,
            'home_goals': 0,
            'away_goals': 0,
            'pp_opportunities': 0,
            'pp_goals': 0,
            'pp_xg': 0,
            'pk_situations': 0,
            'pk_goals_against': 0,
            'pk_xg_against': 0,
            'st_goals_for': 0,
            'st_goals_against': 0,
            'penalties_taken': 0,
            'penalty_minutes': 0,
            'penalties_drawn': 0,
            'empty_net_goals_for': 0,
            'empty_net_goals_against': 0,
            'first_line_toi': 0,
            'second_line_toi': 0,
            'first_line_xg': 0,
            'second_line_xg': 0,
            'wins': 0,
            'losses': 0,
            'ot_losses': 0,
            'xg_per_game_list': [],
            'goals_per_game_list': []
        })
    
    def _init_rating(self):
        return {
            'rating': self.initial_rating,
            'rd': self.initial_rd,
            'volatility': self.initial_volatility
        }
    
    def g(self, rd):
        return 1 / np.sqrt(1 + 3 * rd**2 / (np.pi**2))
    
    def E(self, rating, opp_rating, opp_rd):
        return 1 / (1 + 10 ** (-self.g(opp_rd) * (rating - opp_rating) / 400))
    
    def _update_volatility(self, sigma, delta, rd, v):
        a = np.log(sigma**2)
        
        def f(x):
            ex = np.exp(x)
            return (ex * (delta**2 - rd**2 - v - ex) / 
                   (2 * (rd**2 + v + ex)**2) - 
                   (x - a) / self.tau**2)
        
        A = a
        if delta**2 > rd**2 + v:
            B = np.log(delta**2 - rd**2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau
        
        fA, fB = f(A), f(B)
        while abs(B - A) > self.epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            
            if fC * fB < 0:
                A, fA = B, fB
            else:
                fA /= 2
            
            B, fB = C, fC
        
        return np.exp(A / 2)
    
    def update_rating(self, rating_dict, opponents, outcomes, opp_rds):
        if len(opponents) == 0:
            return
            
        rating = rating_dict['rating']
        rd = rating_dict['rd']
        sigma = rating_dict['volatility']
        
        v_inv = 0
        delta_sum = 0
        
        for opp_rating, outcome, opp_rd in zip(opponents, outcomes, opp_rds):
            E_val = self.E(rating, opp_rating, opp_rd)
            g_val = self.g(opp_rd)
            
            v_inv += g_val**2 * E_val * (1 - E_val)
            delta_sum += g_val * (outcome - E_val)
        
        v = 1 / v_inv if v_inv > 0 else float('inf')
        delta = v * delta_sum
        
        sigma_new = self._update_volatility(sigma, delta, rd, v)
        rd_star = np.sqrt(rd**2 + sigma_new**2)
        rd_new = 1 / np.sqrt(1/rd_star**2 + 1/v)
        rating_new = rating + rd_new**2 * delta_sum
        
        rating_dict['rating'] = rating_new
        rating_dict['rd'] = rd_new
        rating_dict['volatility'] = sigma_new


def process_comprehensive_data(csv_path):
    """
    Process WHL data with ALL metrics
    """
    print("Loading WHL data with comprehensive metrics...")
    df = pd.read_csv(csv_path)
    
    # Separate by situation
    even_strength = df[df['home_off_line'].isin(['first_off', 'second_off'])]
    empty_net = df[df['home_off_line'] == 'empty_net_line']
    
    # EVEN STRENGTH aggregate
    es_summary = even_strength.groupby(['game_id', 'home_team', 'away_team']).agg({
        'home_xg': 'sum',
        'away_xg': 'sum',
        'home_goals': 'sum',
        'away_goals': 'sum',
        'home_shots': 'sum',
        'away_shots': 'sum',
        'home_assists': 'sum',
        'away_assists': 'sum',
        'home_max_xg': 'max',
        'away_max_xg': 'max',
        'toi': 'sum',
        'home_goalie': 'first',
        'away_goalie': 'first',
        'went_ot': 'first'
    }).reset_index()
    
    # LINE-SPECIFIC (first vs second)
    first_line = df[df['home_off_line'] == 'first_off'].groupby(['game_id', 'home_team', 'away_team']).agg({
        'home_xg': 'sum',
        'home_goals': 'sum',
        'home_shots': 'sum',
        'toi': 'sum'
    }).reset_index()
    first_line.columns = ['game_id', 'home_team', 'away_team', 
                          'home_first_line_xg', 'home_first_line_goals', 
                          'home_first_line_shots', 'home_first_line_toi']
    
    second_line = df[df['home_off_line'] == 'second_off'].groupby(['game_id', 'home_team', 'away_team']).agg({
        'home_xg': 'sum',
        'home_goals': 'sum',
        'home_shots': 'sum',
        'toi': 'sum'
    }).reset_index()
    second_line.columns = ['game_id', 'home_team', 'away_team',
                           'home_second_line_xg', 'home_second_line_goals',
                           'home_second_line_shots', 'home_second_line_toi']
    
    # Away team lines
    first_line_away = df[df['away_off_line'] == 'first_off'].groupby(['game_id', 'home_team', 'away_team']).agg({
        'away_xg': 'sum',
        'away_goals': 'sum',
        'away_shots': 'sum',
        'toi': 'sum'
    }).reset_index()
    first_line_away.columns = ['game_id', 'home_team', 'away_team',
                               'away_first_line_xg', 'away_first_line_goals',
                               'away_first_line_shots', 'away_first_line_toi']
    
    second_line_away = df[df['away_off_line'] == 'second_off'].groupby(['game_id', 'home_team', 'away_team']).agg({
        'away_xg': 'sum',
        'away_goals': 'sum',
        'away_shots': 'sum',
        'toi': 'sum'
    }).reset_index()
    second_line_away.columns = ['game_id', 'home_team', 'away_team',
                                'away_second_line_xg', 'away_second_line_goals',
                                'away_second_line_shots', 'away_second_line_toi']
    
    # EMPTY NET
    empty_net_summary = empty_net.groupby(['game_id', 'home_team', 'away_team']).agg({
        'home_goals': 'sum',
        'away_goals': 'sum',
        'toi': 'sum'
    }).reset_index()
    empty_net_summary.columns = ['game_id', 'home_team', 'away_team',
                                  'en_home_goals', 'en_away_goals', 'en_toi']
    
    # POWER PLAY
    pp_home = df[df['home_off_line'] == 'PP_up'].groupby(['game_id', 'home_team', 'away_team']).agg({
        'home_xg': 'sum',
        'home_goals': 'sum',
        'home_shots': 'sum',
        'toi': 'sum'
    }).reset_index()
    pp_home.columns = ['game_id', 'home_team', 'away_team', 'home_pp_xg', 
                        'home_pp_goals', 'home_pp_shots', 'home_pp_toi']
    
    pp_away = df[df['away_off_line'] == 'PP_up'].groupby(['game_id', 'home_team', 'away_team']).agg({
        'away_xg': 'sum',
        'away_goals': 'sum',
        'away_shots': 'sum',
        'toi': 'sum'
    }).reset_index()
    pp_away.columns = ['game_id', 'home_team', 'away_team', 'away_pp_xg', 
                        'away_pp_goals', 'away_pp_shots', 'away_pp_toi']
    
    # PENALTY KILL
    pk_home = df[df['home_off_line'] == 'PP_kill_dwn'].groupby(['game_id', 'home_team', 'away_team']).agg({
        'away_xg': 'sum',
        'away_goals': 'sum',
        'toi': 'sum'
    }).reset_index()
    pk_home.columns = ['game_id', 'home_team', 'away_team', 'home_pk_xg_against', 
                        'home_pk_goals_against', 'home_pk_toi']
    
    pk_away = df[df['away_off_line'] == 'PP_kill_dwn'].groupby(['game_id', 'home_team', 'away_team']).agg({
        'home_xg': 'sum',
        'home_goals': 'sum',
        'toi': 'sum'
    }).reset_index()
    pk_away.columns = ['game_id', 'home_team', 'away_team', 'away_pk_xg_against', 
                        'away_pk_goals_against', 'away_pk_toi']
    
    # PENALTIES
    penalty_summary = df.groupby(['game_id', 'home_team', 'away_team']).agg({
        'home_penalties_committed': 'sum',
        'home_penalty_minutes': 'sum',
        'away_penalties_committed': 'sum',
        'away_penalty_minutes': 'sum'
    }).reset_index()
    
    # MERGE ALL
    games = es_summary.merge(first_line, on=['game_id', 'home_team', 'away_team'], how='left')
    games = games.merge(second_line, on=['game_id', 'home_team', 'away_team'], how='left')
    games = games.merge(first_line_away, on=['game_id', 'home_team', 'away_team'], how='left')
    games = games.merge(second_line_away, on=['game_id', 'home_team', 'away_team'], how='left')
    games = games.merge(empty_net_summary, on=['game_id', 'home_team', 'away_team'], how='left')
    games = games.merge(pp_home, on=['game_id', 'home_team', 'away_team'], how='left')
    games = games.merge(pp_away, on=['game_id', 'home_team', 'away_team'], how='left')
    games = games.merge(pk_home, on=['game_id', 'home_team', 'away_team'], how='left')
    games = games.merge(pk_away, on=['game_id', 'home_team', 'away_team'], how='left')
    games = games.merge(penalty_summary, on=['game_id', 'home_team', 'away_team'], how='left')
    
    games = games.fillna(0)
    
    print(f"Processed {len(games)} games with ALL comprehensive metrics")
    
    return games


def train_ultimate_system(csv_path):
    """
    Train complete system with ALL 18 components
    """
    games = process_comprehensive_data(csv_path)
    system = UltimateGlickoHockey()
    
    print("\nTraining ULTIMATE rating system (18 components)...")
    
    for idx, game in games.iterrows():
        if idx % 200 == 0:
            print(f"  Processing game {idx}/{len(games)}...")
        
        home_team = game['home_team']
        away_team = game['away_team']
        home_goalie = game['home_goalie']
        away_goalie = game['away_goalie']
        
        is_home_game_for_home = True
        
        # Track statistics
        system.team_stats[home_team]['games'] += 1
        system.team_stats[away_team]['games'] += 1
        system.team_stats[home_team]['home_games'] += 1
        system.team_stats[away_team]['away_games'] += 1
        
        # Win tracking
        home_won = game['home_goals'] > game['away_goals']
        is_ot = game['went_ot'] == 1
        
        if home_won:
            system.team_stats[home_team]['wins'] += 1
            system.team_stats[away_team]['losses'] += 1
            if is_ot:
                system.team_stats[away_team]['ot_losses'] += 1
                system.team_stats[home_team]['ot_wins'] += 1
                system.team_stats[home_team]['ot_games'] += 1
                system.team_stats[away_team]['ot_games'] += 1
        elif game['away_goals'] > game['home_goals']:
            system.team_stats[away_team]['wins'] += 1
            system.team_stats[home_team]['losses'] += 1
            if is_ot:
                system.team_stats[home_team]['ot_losses'] += 1
                system.team_stats[away_team]['ot_wins'] += 1
                system.team_stats[home_team]['ot_games'] += 1
                system.team_stats[away_team]['ot_games'] += 1
        
        # Basic stats
        system.team_stats[home_team]['goals'] += game['home_goals']
        system.team_stats[home_team]['goals_against'] += game['away_goals']
        system.team_stats[home_team]['shots'] += game['home_shots']
        system.team_stats[home_team]['shots_against'] += game['away_shots']
        system.team_stats[home_team]['xg'] += game['home_xg']
        system.team_stats[home_team]['xg_against'] += game['away_xg']
        system.team_stats[home_team]['assists'] += game['home_assists']
        system.team_stats[home_team]['max_xg_total'] += game['home_max_xg']
        system.team_stats[home_team]['home_xg'] += game['home_xg']
        system.team_stats[home_team]['home_goals'] += game['home_goals']
        
        system.team_stats[away_team]['goals'] += game['away_goals']
        system.team_stats[away_team]['goals_against'] += game['home_goals']
        system.team_stats[away_team]['shots'] += game['away_shots']
        system.team_stats[away_team]['shots_against'] += game['home_shots']
        system.team_stats[away_team]['xg'] += game['away_xg']
        system.team_stats[away_team]['xg_against'] += game['home_xg']
        system.team_stats[away_team]['assists'] += game['away_assists']
        system.team_stats[away_team]['max_xg_total'] += game['away_max_xg']
        system.team_stats[away_team]['away_xg'] += game['away_xg']
        system.team_stats[away_team]['away_goals'] += game['away_goals']
        
        # For consistency tracking
        system.team_stats[home_team]['xg_per_game_list'].append(game['home_xg'])
        system.team_stats[home_team]['goals_per_game_list'].append(game['home_goals'])
        system.team_stats[away_team]['xg_per_game_list'].append(game['away_xg'])
        system.team_stats[away_team]['goals_per_game_list'].append(game['away_goals'])
        
        # EVEN STRENGTH OFFENSE/DEFENSE
        home_xg = game['home_xg']
        away_xg = game['away_xg']
        total_xg = home_xg + away_xg
        
        if total_xg > 0:
            es_home_out = home_xg / total_xg
            es_away_out = away_xg / total_xg
        else:
            es_home_out = es_away_out = 0.5
        
        system.update_rating(
            system.even_strength_offense[home_team],
            [system.even_strength_defense[away_team]['rating']],
            [es_home_out],
            [system.even_strength_defense[away_team]['rd']]
        )
        
        system.update_rating(
            system.even_strength_offense[away_team],
            [system.even_strength_defense[home_team]['rating']],
            [es_away_out],
            [system.even_strength_defense[home_team]['rd']]
        )
        
        system.update_rating(
            system.even_strength_defense[home_team],
            [system.even_strength_offense[away_team]['rating']],
            [es_home_out],
            [system.even_strength_offense[away_team]['rd']]
        )
        
        system.update_rating(
            system.even_strength_defense[away_team],
            [system.even_strength_offense[home_team]['rating']],
            [es_away_out],
            [system.even_strength_offense[home_team]['rd']]
        )
        
        # GOALIE (GSAE) 
        home_xg_against = game['away_xg']
        home_goals_against = game['away_goals']
        
        if home_xg_against > 0:
            home_gsae_rate = (home_xg_against - home_goals_against) / home_xg_against
            home_goalie_out = np.clip(0.5 + 2.5 * home_gsae_rate, 0, 1)
        else:
            home_goalie_out = 0.5
        
        away_xg_against = game['home_xg']
        away_goals_against = game['home_goals']
        
        if away_xg_against > 0:
            away_gsae_rate = (away_xg_against - away_goals_against) / away_xg_against
            away_goalie_out = np.clip(0.5 + 2.5 * away_gsae_rate, 0, 1)
        else:
            away_goalie_out = 0.5
        
        system.update_rating(system.goalie_ratings[home_goalie], [1500], [home_goalie_out], [100])
        system.update_rating(system.goalie_ratings[away_goalie], [1500], [away_goalie_out], [100])
        
        # SHOOTING EFFICIENCY (Conversion Rate)
        home_shots = game['home_shots']
        home_goals = game['home_goals']
        
        if home_shots > 0:
            home_conversion_rate = home_goals / home_shots
            home_efficiency_out = np.clip(home_conversion_rate / 0.12 * 0.5 + 0.25, 0, 1)
        else:
            home_efficiency_out = 0.5
        
        away_shots = game['away_shots']
        away_goals = game['away_goals']
        
        if away_shots > 0:
            away_conversion_rate = away_goals / away_shots
            away_efficiency_out = np.clip(away_conversion_rate / 0.12 * 0.5 + 0.25, 0, 1)
        else:
            away_efficiency_out = 0.5
        
        system.update_rating(system.shooting_efficiency[home_team], [1500], [home_efficiency_out], [100])
        system.update_rating(system.shooting_efficiency[away_team], [1500], [away_efficiency_out], [100])
        
        # EMPTY NET PERFORMANCE 
        if game['en_toi'] > 0:
            en_home_goals = game['en_home_goals']
            en_away_goals = game['en_away_goals']
            
            if en_home_goals + en_away_goals > 0:
                en_home_out = en_home_goals / (en_home_goals + en_away_goals)
                en_away_out = en_away_goals / (en_home_goals + en_away_goals)
            else:
                en_home_out = en_away_out = 0.5
            
            system.update_rating(system.empty_net_performance[home_team], [1500], [en_home_out], [100])
            system.update_rating(system.empty_net_performance[away_team], [1500], [en_away_out], [100])
            
            system.team_stats[home_team]['empty_net_goals_for'] += en_home_goals
            system.team_stats[home_team]['empty_net_goals_against'] += en_away_goals
            system.team_stats[away_team]['empty_net_goals_for'] += en_away_goals
            system.team_stats[away_team]['empty_net_goals_against'] += en_home_goals
        
        # LINE DEPTH QUALITY
        home_first_toi = game['home_first_line_toi']
        home_second_toi = game['home_second_line_toi']
        home_first_xg = game['home_first_line_xg']
        home_second_xg = game['home_second_line_xg']
        
        away_first_toi = game['away_first_line_toi']
        away_second_toi = game['away_second_line_toi']
        away_first_xg = game['away_first_line_xg']
        away_second_xg = game['away_second_line_xg']
        
        system.team_stats[home_team]['first_line_toi'] += home_first_toi
        system.team_stats[home_team]['second_line_toi'] += home_second_toi
        system.team_stats[home_team]['first_line_xg'] += home_first_xg
        system.team_stats[home_team]['second_line_xg'] += home_second_xg
        
        system.team_stats[away_team]['first_line_toi'] += away_first_toi
        system.team_stats[away_team]['second_line_toi'] += away_second_toi
        system.team_stats[away_team]['first_line_xg'] += away_first_xg
        system.team_stats[away_team]['second_line_xg'] += away_second_xg
        
        if home_second_toi > 0 and home_first_toi > 0:
            home_first_xg_per_60 = (home_first_xg / home_first_toi) * 3600
            home_second_xg_per_60 = (home_second_xg / home_second_toi) * 3600
            
            if home_first_xg_per_60 > 0:
                home_depth_ratio = home_second_xg_per_60 / home_first_xg_per_60
                home_depth_out = np.clip(home_depth_ratio, 0, 1)
            else:
                home_depth_out = 0.5
        else:
            home_depth_out = 0.5
        
        if away_second_toi > 0 and away_first_toi > 0:
            away_first_xg_per_60 = (away_first_xg / away_first_toi) * 3600
            away_second_xg_per_60 = (away_second_xg / away_second_toi) * 3600
            
            if away_first_xg_per_60 > 0:
                away_depth_ratio = away_second_xg_per_60 / away_first_xg_per_60
                away_depth_out = np.clip(away_depth_ratio, 0, 1)
            else:
                away_depth_out = 0.5
        else:
            away_depth_out = 0.5
        
        system.update_rating(system.line_depth_quality[home_team], [1500], [home_depth_out], [100])
        system.update_rating(system.line_depth_quality[away_team], [1500], [away_depth_out], [100])
        
        # OVERTIME/CLUTCH PERFORMANCE
        if is_ot:
            ot_home_out = 1.0 if home_won else 0.0
            ot_away_out = 1.0 if not home_won else 0.0
            
            system.update_rating(system.ot_clutch_performance[home_team], [1500], [ot_home_out], [100])
            system.update_rating(system.ot_clutch_performance[away_team], [1500], [ot_away_out], [100])
        
        # SHOT QUALITY DIFFERENTIAL
        if home_shots > 0 and away_shots > 0:
            home_shot_quality = home_xg / home_shots
            away_shot_quality_against = away_xg / away_shots
            
            # Home team's quality vs opponent's quality
            total_quality = home_shot_quality + away_shot_quality_against
            if total_quality > 0:
                sq_home_out = home_shot_quality / total_quality
                sq_away_out = away_shot_quality_against / total_quality
            else:
                sq_home_out = sq_away_out = 0.5
            
            system.update_rating(system.shot_quality_differential[home_team], [1500], [sq_home_out], [100])
            system.update_rating(system.shot_quality_differential[away_team], [1500], [sq_away_out], [100])
        
        # PDO SUSTAINABILITY
        if home_shots > 0 and away_shots > 0:
            home_shooting_pct = home_goals / home_shots
            home_save_pct = 1 - (away_goals / away_shots)
            home_pdo = home_shooting_pct + home_save_pct
            
            # Closer to 1.0 = more sustainable
            home_pdo_out = np.clip(1.0 - abs(home_pdo - 1.0), 0, 1)
            
            away_shooting_pct = away_goals / away_shots
            away_save_pct = 1 - (home_goals / home_shots)
            away_pdo = away_shooting_pct + away_save_pct
            
            away_pdo_out = np.clip(1.0 - abs(away_pdo - 1.0), 0, 1)
            
            system.update_rating(system.pdo_sustainability[home_team], [1500], [home_pdo_out], [100])
            system.update_rating(system.pdo_sustainability[away_team], [1500], [away_pdo_out], [100])
        
        # PLAYMAKING ABILITY (Assists)
        home_assists = game['home_assists']
        away_assists = game['away_assists']
        
        if home_goals > 0:
            home_playmaking_ratio = home_assists / home_goals
            # Ratio > 1.5 = good team play
            home_playmaking_out = np.clip(home_playmaking_ratio / 2.0, 0, 1)
        else:
            home_playmaking_out = 0.5
        
        if away_goals > 0:
            away_playmaking_ratio = away_assists / away_goals
            away_playmaking_out = np.clip(away_playmaking_ratio / 2.0, 0, 1)
        else:
            away_playmaking_out = 0.5
        
        system.update_rating(system.playmaking_ability[home_team], [1500], [home_playmaking_out], [100])
        system.update_rating(system.playmaking_ability[away_team], [1500], [away_playmaking_out], [100])
        
        # OFFENSIVE PRESSURE (Shot Volume)
        total_shots = home_shots + away_shots
        if total_shots > 0:
            shot_share_home = home_shots / total_shots
            shot_share_away = away_shots / total_shots
        else:
            shot_share_home = shot_share_away = 0.5
        
        system.update_rating(system.offensive_pressure[home_team], [1500], [shot_share_home], [100])
        system.update_rating(system.offensive_pressure[away_team], [1500], [shot_share_away], [100])
        
        # xG FINISHING SKILL (Over/Underperformance)
        if home_xg > 0:
            home_xg_diff = home_goals - home_xg
            # Positive = overperforming (good finishing)
            home_finishing_out = np.clip(0.5 + (home_xg_diff / 3.0), 0, 1)
        else:
            home_finishing_out = 0.5
        
        if away_xg > 0:
            away_xg_diff = away_goals - away_xg
            away_finishing_out = np.clip(0.5 + (away_xg_diff / 3.0), 0, 1)
        else:
            away_finishing_out = 0.5
        
        system.update_rating(system.xg_finishing_skill[home_team], [1500], [home_finishing_out], [100])
        system.update_rating(system.xg_finishing_skill[away_team], [1500], [away_finishing_out], [100])
        
        # HIGH DANGER CREATION (Max xG)
        home_max_xg = game['home_max_xg']
        away_max_xg = game['away_max_xg']
        
        total_max_xg = home_max_xg + away_max_xg
        if total_max_xg > 0:
            hd_home_out = home_max_xg / total_max_xg
            hd_away_out = away_max_xg / total_max_xg
        else:
            hd_home_out = hd_away_out = 0.5
        
        system.update_rating(system.high_danger_creation[home_team], [1500], [hd_home_out], [100])
        system.update_rating(system.high_danger_creation[away_team], [1500], [hd_away_out], [100])
        
        # POWER PLAY
        if game['home_pp_toi'] > 0:
            home_pp_xg = game['home_pp_xg']
            home_pp_goals = game['home_pp_goals']
            
            if home_pp_xg > 0:
                conversion_ratio = home_pp_goals / home_pp_xg
                home_pp_out = np.clip(0.3 + 0.4 * conversion_ratio, 0, 1)
            else:
                home_pp_out = 0.5
            
            system.update_rating(
                system.power_play_offense[home_team],
                [system.penalty_kill_defense[away_team]['rating']],
                [home_pp_out],
                [system.penalty_kill_defense[away_team]['rd']]
            )
            
            system.update_rating(
                system.penalty_kill_defense[away_team],
                [system.power_play_offense[home_team]['rating']],
                [1 - home_pp_out],
                [system.power_play_offense[home_team]['rd']]
            )
            
            system.team_stats[home_team]['pp_opportunities'] += 1
            system.team_stats[home_team]['pp_goals'] += home_pp_goals
            system.team_stats[home_team]['pp_xg'] += home_pp_xg
            system.team_stats[home_team]['st_goals_for'] += home_pp_goals
        
        if game['away_pp_toi'] > 0:
            away_pp_xg = game['away_pp_xg']
            away_pp_goals = game['away_pp_goals']
            
            if away_pp_xg > 0:
                conversion_ratio = away_pp_goals / away_pp_xg
                away_pp_out = np.clip(0.3 + 0.4 * conversion_ratio, 0, 1)
            else:
                away_pp_out = 0.5
            
            system.update_rating(
                system.power_play_offense[away_team],
                [system.penalty_kill_defense[home_team]['rating']],
                [away_pp_out],
                [system.penalty_kill_defense[home_team]['rd']]
            )
            
            system.update_rating(
                system.penalty_kill_defense[home_team],
                [system.power_play_offense[away_team]['rating']],
                [1 - away_pp_out],
                [system.power_play_offense[away_team]['rd']]
            )
            
            system.team_stats[away_team]['pp_opportunities'] += 1
            system.team_stats[away_team]['pp_goals'] += away_pp_goals
            system.team_stats[away_team]['pp_xg'] += away_pp_xg
            system.team_stats[away_team]['st_goals_for'] += away_pp_goals
        
        # PENALTY KILL
        if game['home_pk_toi'] > 0:
            system.team_stats[home_team]['pk_situations'] += 1
            system.team_stats[home_team]['pk_goals_against'] += game['home_pk_goals_against']
            system.team_stats[home_team]['pk_xg_against'] += game['home_pk_xg_against']
            system.team_stats[home_team]['st_goals_against'] += game['home_pk_goals_against']
        
        if game['away_pk_toi'] > 0:
            system.team_stats[away_team]['pk_situations'] += 1
            system.team_stats[away_team]['pk_goals_against'] += game['away_pk_goals_against']
            system.team_stats[away_team]['pk_xg_against'] += game['away_pk_xg_against']
            system.team_stats[away_team]['st_goals_against'] += game['away_pk_goals_against']
        
        # ST GOAL DIFFERENTIAL
        home_st_goals_this_game = game.get('home_pp_goals', 0)
        away_st_goals_this_game = game.get('away_pp_goals', 0)
        home_st_goals_against_this_game = game.get('home_pk_goals_against', 0)
        away_st_goals_against_this_game = game.get('away_pk_goals_against', 0)
        
        home_st_diff = home_st_goals_this_game - home_st_goals_against_this_game
        away_st_diff = away_st_goals_this_game - away_st_goals_against_this_game
        
        # Outcome based on ST goal differential
        total_st_diff = abs(home_st_diff) + abs(away_st_diff)
        if total_st_diff > 0:
            st_home_out = np.clip(0.5 + (home_st_diff / 4.0), 0, 1)
            st_away_out = np.clip(0.5 + (away_st_diff / 4.0), 0, 1)
        else:
            st_home_out = st_away_out = 0.5
        
        system.update_rating(system.st_goal_impact[home_team], [1500], [st_home_out], [100])
        system.update_rating(system.st_goal_impact[away_team], [1500], [st_away_out], [100])
        
        # PENALTY DISCIPLINE
        home_pim = game['home_penalty_minutes']
        away_pim = game['away_penalty_minutes']
        total_pim = home_pim + away_pim
        
        if total_pim > 0:
            home_disc_out = away_pim / total_pim
            away_disc_out = home_pim / total_pim
        else:
            home_disc_out = away_disc_out = 0.5
        
        system.update_rating(system.penalty_discipline[home_team], [1500], [home_disc_out], [100])
        system.update_rating(system.penalty_discipline[away_team], [1500], [away_disc_out], [100])
        
        system.team_stats[home_team]['penalties_taken'] += game['home_penalties_committed']
        system.team_stats[home_team]['penalty_minutes'] += home_pim
        system.team_stats[home_team]['penalties_drawn'] += game['away_penalties_committed']
        
        system.team_stats[away_team]['penalties_taken'] += game['away_penalties_committed']
        system.team_stats[away_team]['penalty_minutes'] += away_pim
        system.team_stats[away_team]['penalties_drawn'] += game['home_penalties_committed']
    
    # Calculate consistency and home/away splits
    print("\nCalculating post-game metrics (consistency, home/away)...")
    
    for team in system.even_strength_offense.keys():
        stats = system.team_stats[team]
        
        # PERFORMANCE CONSISTENCY 
        if len(stats['xg_per_game_list']) > 1:
            xg_std = np.std(stats['xg_per_game_list'])
            if xg_std > 0:
                # Lower std = more consistent = higher rating
                consistency_score = 1 / xg_std
                # Normalize to 0-1 range
                consistency_out = np.clip(consistency_score / 2.0, 0, 1)
            else:
                consistency_out = 1.0  # Perfect consistency
        else:
            consistency_out = 0.5
        
        system.update_rating(system.performance_consistency[team], [1500], [consistency_out], [100])
        
        # HOME/AWAY CONSISTENCY
        if stats['home_games'] > 0 and stats['away_games'] > 0:
            home_xg_per_game = stats['home_xg'] / stats['home_games']
            away_xg_per_game = stats['away_xg'] / stats['away_games']
            
            total_xg = home_xg_per_game + away_xg_per_game
            if total_xg > 0:
                # More balanced = better
                balance = min(home_xg_per_game, away_xg_per_game) / max(home_xg_per_game, away_xg_per_game)
                home_away_out = balance  # Closer to 1.0 = more balanced
            else:
                home_away_out = 0.5
        else:
            home_away_out = 0.5
        
        system.update_rating(system.home_away_consistency[team], [1500], [home_away_out], [100])
    
    print("✓ Training complete!")
    return system, games


def calculate_ultimate_rankings(system, games):
    """
    Calculate composite rankings with ALL 18 components
    
    Weight distribution (Total = 100%):
    ES Offense: 14%
    ES Defense: 12%
    Goalie: 18%
    Shooting Efficiency: 4%
    Empty Net: 2%
    Line Depth: 5%
    OT/Clutch: 4%
    Shot Quality Diff: 5%
    PDO Sustainability: 3%
    Playmaking: 2%
    Offensive Pressure: 3%
    xG Finishing: 2%
    High Danger: 3%
    Power Play: 9%
    Penalty Kill: 6%
    ST Goal Impact: 3%
    Discipline: 3%
    Consistency: 2%
    """
    
    print("\nCalculating ULTIMATE power rankings (18 components)...")
    
    # Get primary goalies
    home_goalie_usage = games.groupby(['home_team', 'home_goalie']).size().reset_index()
    home_goalie_usage.columns = ['team', 'goalie', 'games']
    
    away_goalie_usage = games.groupby(['away_team', 'away_goalie']).size().reset_index()
    away_goalie_usage.columns = ['team', 'goalie', 'games']
    
    all_goalie_usage = pd.concat([home_goalie_usage, away_goalie_usage])
    all_goalie_usage = all_goalie_usage.groupby(['team', 'goalie']).sum().reset_index()
    primary_goalies = all_goalie_usage.loc[all_goalie_usage.groupby('team')['games'].idxmax()]
    goalie_map = dict(zip(primary_goalies['team'], primary_goalies['goalie']))
    
    results = []
    
    for team in sorted(system.even_strength_offense.keys()):
        # Get all ratings
        es_off = system.even_strength_offense[team]['rating']
        es_def = system.even_strength_defense[team]['rating']
        shoot_eff = system.shooting_efficiency[team]['rating']
        empty_net = system.empty_net_performance[team]['rating']
        line_depth = system.line_depth_quality[team]['rating']
        ot_clutch = system.ot_clutch_performance[team]['rating']
        shot_quality = system.shot_quality_differential[team]['rating']
        pdo = system.pdo_sustainability[team]['rating']
        playmaking = system.playmaking_ability[team]['rating']
        pressure = system.offensive_pressure[team]['rating']
        xg_finish = system.xg_finishing_skill[team]['rating']
        high_danger = system.high_danger_creation[team]['rating']
        pp_off = system.power_play_offense[team]['rating']
        pk_def = system.penalty_kill_defense[team]['rating']
        st_impact = system.st_goal_impact[team]['rating']
        discipline = system.penalty_discipline[team]['rating']
        consistency = system.performance_consistency[team]['rating']
        home_away = system.home_away_consistency[team]['rating']
        
        goalie_id = goalie_map.get(team, 'unknown')
        goalie = system.goalie_ratings[goalie_id]['rating']
        
        # Composite calculation
        composite = (
            0.14 * es_off +
            0.12 * es_def +
            0.18 * goalie +
            0.04 * shoot_eff +
            0.02 * empty_net +
            0.05 * line_depth +
            0.04 * ot_clutch +
            0.05 * shot_quality +
            0.03 * pdo +
            0.02 * playmaking +
            0.03 * pressure +
            0.02 * xg_finish +
            0.03 * high_danger +
            0.09 * pp_off +
            0.06 * pk_def +
            0.03 * st_impact +
            0.03 * discipline +
            0.02 * consistency
            # Note: home_away not included in composite to keep at 100%
        )
        
        # Calculate statistics
        stats = system.team_stats[team]
        
        shooting_pct = (stats['goals'] / stats['shots'] * 100 
                       if stats['shots'] > 0 else 0)
        
        pp_pct = (stats['pp_goals'] / stats['pp_xg'] * 100 
                  if stats['pp_xg'] > 0 else 0)
        
        pk_pct = ((stats['pk_xg_against'] - stats['pk_goals_against']) / 
                 stats['pk_xg_against'] * 100
                 if stats['pk_xg_against'] > 0 else 0)
        
        ot_win_rate = (stats['ot_wins'] / stats['ot_games'] * 100
                      if stats['ot_games'] > 0 else 0)
        
        assists_per_goal = (stats['assists'] / stats['goals']
                           if stats['goals'] > 0 else 0)
        
        xg_diff = (stats['goals'] - stats['xg']) / stats['games'] if stats['games'] > 0 else 0
        
        if stats['first_line_toi'] > 0 and stats['second_line_toi'] > 0:
            first_xg_per_60 = (stats['first_line_xg'] / stats['first_line_toi']) * 3600
            second_xg_per_60 = (stats['second_line_xg'] / stats['second_line_toi']) * 3600
            line_disparity = (first_xg_per_60 / second_xg_per_60 
                             if second_xg_per_60 > 0 else 999)
        else:
            line_disparity = 0
        
        pdo_actual = ((stats['goals'] / stats['shots']) + 
                     (1 - stats['goals_against'] / stats['shots_against'])
                     if stats['shots'] > 0 and stats['shots_against'] > 0 else 1.0)
        
        st_goal_diff = stats['st_goals_for'] - stats['st_goals_against']
        
        xg_std = np.std(stats['xg_per_game_list']) if len(stats['xg_per_game_list']) > 0 else 0
        
        win_pct = stats['wins'] / stats['games'] * 100 if stats['games'] > 0 else 0
        
        results.append({
            'team': team,
            'composite_rating': composite,
            'es_offense': es_off,
            'es_defense': es_def,
            'goalie': goalie,
            'shooting_efficiency': shoot_eff,
            'empty_net': empty_net,
            'line_depth': line_depth,
            'ot_clutch': ot_clutch,
            'shot_quality': shot_quality,
            'pdo': pdo,
            'playmaking': playmaking,
            'offensive_pressure': pressure,
            'xg_finishing': xg_finish,
            'high_danger': high_danger,
            'pp_offense': pp_off,
            'pk_defense': pk_def,
            'st_impact': st_impact,
            'discipline': discipline,
            'consistency': consistency,
            'home_away': home_away,
            'shooting_pct': shooting_pct,
            'pp_pct': pp_pct,
            'pk_pct': pk_pct,
            'ot_win_rate': ot_win_rate,
            'assists_per_goal': assists_per_goal,
            'xg_over_under': xg_diff,
            'line_disparity': line_disparity,
            'pdo_actual': pdo_actual,
            'st_goal_diff': st_goal_diff,
            'xg_std': xg_std,
            'win_pct': win_pct,
            'wins': stats['wins'],
            'losses': stats['losses']
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('composite_rating', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def main():
    """
    Run ULTIMATE comprehensive system
    """
    
    # Train
    system, games = train_ultimate_system('/mnt/user-data/uploads/whl_2025.csv')
    
    # Rankings
    rankings = calculate_ultimate_rankings(system, games)
    
    print("\n" + "="*100)
    print("TOP 10 POWER RANKINGS (ULTIMATE - 18 COMPONENTS)")
    print("="*100)
    print(rankings[['rank', 'team', 'composite_rating', 'win_pct', 
                    'ot_win_rate', 'pdo_actual', 'xg_over_under']].head(10).to_string(index=False))
    
    print("\n" + "="*100)
    print("WEIGHT DISTRIBUTION:")
    print("="*100)
    print("ES Offense: 14%, ES Defense: 12%, Goalie: 18%")
    print("Shooting Eff: 4%, Empty Net: 2%, Line Depth: 5%")
    print("OT/Clutch: 4%, Shot Quality: 5%, PDO: 3%")
    print("Playmaking: 2%, Pressure: 3%, xG Finishing: 2%")
    print("High Danger: 3%, Power Play: 9%, Penalty Kill: 6%")
    print("ST Impact: 3%, Discipline: 3%, Consistency: 2%")
    print("TOTAL: 100%")
    
    # Save
    rankings.to_csv('/mnt/user-data/outputs/ultimate_power_rankings.csv', index=False)
    
    # Predict matchups
    matchups = pd.read_excel('/mnt/user-data/uploads/WHSDSC_Rnd1_matchups.xlsx')
    
    print("\n" + "="*100)
    print("PLAYOFF PREDICTIONS")
    print("="*100)
    
    for _, matchup in matchups.iterrows():
        home = matchup['home_team']
        away = matchup['away_team']
        
        home_rating = rankings[rankings['team'] == home]['composite_rating'].values[0]
        away_rating = rankings[rankings['team'] == away]['composite_rating'].values[0]
        
        # Simple win probability from rating differential
        rating_diff = home_rating - away_rating + 50  # Home advantage
        win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
        
        print(f"{home} vs {away}: {win_prob:.1%}")
    
    print("\n✓ ULTIMATE SYSTEM COMPLETE!")
    print(f"✓ All 18 components implemented")
    print(f"✓ Rankings saved to ultimate_power_rankings.csv")
    
    return system, rankings


if __name__ == "__main__":
    system, rankings = main()
