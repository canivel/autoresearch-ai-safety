#!/usr/bin/env python3
"""
March Madness 2026 Prediction Model

Builds predictions for every possible team matchup in the 2026 NCAA
men's and women's basketball tournaments using:
1. Elo ratings computed from historical game results
2. Seed-based adjustments for tournament context
3. Team statistics (efficiency metrics) from detailed box scores
4. Logistic regression combining multiple signals

Usage:
    # Auto-download from Kaggle (requires KAGGLE_API_TOKEN env var):
    python predict.py --output submission.csv

    # Or specify local data directory:
    python predict.py --data-dir ./data --output submission.csv
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_csv(filepath):
    """Load a CSV file and return list of dicts."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def find_data_file(data_dir, filename):
    """Find a data file, trying multiple common locations."""
    candidates = [
        os.path.join(data_dir, filename),
        os.path.join(data_dir, filename.lower()),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(data_dir, filename)


class EloSystem:
    """Elo rating system for team strength estimation."""

    def __init__(self, k=32, home_advantage=64, mean_elo=1500, reversion=0.30, mov_coeff=0.5):
        self.k = k
        self.home_advantage = home_advantage
        self.mean_elo = mean_elo
        self.reversion = reversion
        self.mov_coeff = mov_coeff
        self.ratings = defaultdict(lambda: self.mean_elo)
        self.recent_results = defaultdict(list)  # Track recent game results for momentum

    def expected_score(self, rating_a, rating_b):
        """Expected score for team A against team B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, team_w, team_l, loc, margin=None):
        """Update ratings after a game. team_w won, team_l lost."""
        rating_w = self.ratings[team_w]
        rating_l = self.ratings[team_l]

        # Location adjustment
        if loc == 'H':
            rating_w_adj = rating_w + self.home_advantage
            rating_l_adj = rating_l
        elif loc == 'A':
            rating_w_adj = rating_w
            rating_l_adj = rating_l + self.home_advantage
        else:
            rating_w_adj = rating_w
            rating_l_adj = rating_l

        # Margin of victory multiplier (capped)
        if margin is not None:
            mov_mult = math.log(max(abs(margin), 1) + 1) * self.mov_coeff
        else:
            mov_mult = 1.0

        expected_w = self.expected_score(rating_w_adj, rating_l_adj)
        k_adj = self.k * mov_mult

        self.ratings[team_w] += k_adj * (1 - expected_w)
        self.ratings[team_l] += k_adj * (0 - (1 - expected_w))

        # Track recent results (1=win, 0=loss, with margin info)
        self.recent_results[team_w].append((1, margin if margin else 0))
        self.recent_results[team_l].append((0, -(margin if margin else 0)))

    def new_season(self):
        """Revert ratings toward mean for a new season."""
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - self.reversion)
                + self.mean_elo * self.reversion
            )
        self.recent_results.clear()

    def get_momentum(self, team_id, n_games=10):
        """Get team momentum from recent games (win rate in last N games)."""
        results = self.recent_results.get(team_id, [])
        if not results:
            return 0.5
        recent = results[-n_games:]
        return sum(r[0] for r in recent) / len(recent)

    def predict(self, team_a, team_b):
        """Predict probability of team_a beating team_b on neutral court."""
        return self.expected_score(self.ratings[team_a], self.ratings[team_b])


class TeamStats:
    """Compute per-team season statistics from detailed results."""

    def __init__(self):
        self.stats = {}  # (season, team_id) -> dict of stats

    def compute_from_detailed(self, games, season):
        """Compute team stats from detailed results for a season."""
        team_games = defaultdict(list)

        for g in games:
            if int(g['Season']) != season:
                continue
            w_id = g['WTeamID']
            l_id = g['LTeamID']

            # Winner stats
            w_stats = {
                'score': int(g['WScore']),
                'opp_score': int(g['LScore']),
                'fgm': int(g.get('WFGM', 0)),
                'fga': int(g.get('WFGA', 0)),
                'fgm3': int(g.get('WFGM3', 0)),
                'fga3': int(g.get('WFGA3', 0)),
                'ftm': int(g.get('WFTM', 0)),
                'fta': int(g.get('WFTA', 0)),
                'or': int(g.get('WOR', 0)),
                'dr': int(g.get('WDR', 0)),
                'ast': int(g.get('WAst', 0)),
                'to': int(g.get('WTO', 0)),
                'stl': int(g.get('WStl', 0)),
                'blk': int(g.get('WBlk', 0)),
                'pf': int(g.get('WPF', 0)),
                'opp_or': int(g.get('LOR', 0)),
                'opp_dr': int(g.get('LDR', 0)),
                'opp_fga': int(g.get('LFGA', 0)),
                'opp_fgm': int(g.get('LFGM', 0)),
                'opp_fgm3': int(g.get('LFGM3', 0)),
                'opp_to': int(g.get('LTO', 0)),
                'opp_fta': int(g.get('LFTA', 0)),
                'opp_ftm': int(g.get('LFTM', 0)),
            }
            team_games[w_id].append(w_stats)

            # Loser stats
            l_stats = {
                'score': int(g['LScore']),
                'opp_score': int(g['WScore']),
                'fgm': int(g.get('LFGM', 0)),
                'fga': int(g.get('LFGA', 0)),
                'fgm3': int(g.get('LFGM3', 0)),
                'fga3': int(g.get('LFGA3', 0)),
                'ftm': int(g.get('LFTM', 0)),
                'fta': int(g.get('LFTA', 0)),
                'or': int(g.get('LOR', 0)),
                'dr': int(g.get('LDR', 0)),
                'ast': int(g.get('LAst', 0)),
                'to': int(g.get('LTO', 0)),
                'stl': int(g.get('LStl', 0)),
                'blk': int(g.get('LBlk', 0)),
                'pf': int(g.get('LPF', 0)),
                'opp_or': int(g.get('WOR', 0)),
                'opp_dr': int(g.get('WDR', 0)),
                'opp_fga': int(g.get('WFGA', 0)),
                'opp_fgm': int(g.get('WFGM', 0)),
                'opp_fgm3': int(g.get('WFGM3', 0)),
                'opp_to': int(g.get('WTO', 0)),
                'opp_fta': int(g.get('WFTA', 0)),
                'opp_ftm': int(g.get('WFTM', 0)),
            }
            team_games[l_id].append(l_stats)

        # Aggregate stats per team
        for team_id, games_list in team_games.items():
            n = len(games_list)
            if n == 0:
                continue

            total = defaultdict(float)
            for g in games_list:
                for k, v in g.items():
                    total[k] += v

            avg = {k: v / n for k, v in total.items()}

            # Compute advanced stats
            possessions = avg['fga'] - avg['or'] + avg['to'] + 0.475 * avg['fta']
            opp_possessions = avg['opp_fga'] - avg['opp_or'] + avg['opp_to'] + 0.475 * avg['opp_fta']

            if possessions > 0:
                off_efficiency = avg['score'] / possessions * 100
            else:
                off_efficiency = 100.0

            if opp_possessions > 0:
                def_efficiency = avg['opp_score'] / opp_possessions * 100
            else:
                def_efficiency = 100.0

            fg_pct = avg['fgm'] / avg['fga'] if avg['fga'] > 0 else 0.0
            fg3_pct = avg['fgm3'] / avg['fga3'] if avg['fga3'] > 0 else 0.0
            ft_pct = avg['ftm'] / avg['fta'] if avg['fta'] > 0 else 0.0
            ast_to_ratio = avg['ast'] / avg['to'] if avg['to'] > 0 else 1.0

            # Four Factors (Dean Oliver)
            efg_pct = (avg['fgm'] + 0.5 * avg['fgm3']) / avg['fga'] if avg['fga'] > 0 else 0.0
            to_pct = avg['to'] / possessions if possessions > 0 else 0.0
            or_pct = avg['or'] / (avg['or'] + avg['opp_dr']) if (avg['or'] + avg['opp_dr']) > 0 else 0.0
            ft_rate = avg['ftm'] / avg['fga'] if avg['fga'] > 0 else 0.0
            # Opponent four factors
            opp_efg_pct = (avg['opp_fgm'] + 0.5 * avg['opp_fgm3']) / avg['opp_fga'] if avg['opp_fga'] > 0 else 0.0
            opp_to_pct = avg['opp_to'] / opp_possessions if opp_possessions > 0 else 0.0
            opp_or_pct = avg['opp_or'] / (avg['opp_or'] + avg['dr']) if (avg['opp_or'] + avg['dr']) > 0 else 0.0
            opp_ft_rate = avg['opp_ftm'] / avg['opp_fga'] if avg['opp_fga'] > 0 else 0.0

            self.stats[(season, team_id)] = {
                'games': n,
                'ppg': avg['score'],
                'opp_ppg': avg['opp_score'],
                'off_eff': off_efficiency,
                'def_eff': def_efficiency,
                'net_eff': off_efficiency - def_efficiency,
                'fg_pct': fg_pct,
                'fg3_pct': fg3_pct,
                'ft_pct': ft_pct,
                'ast_to': ast_to_ratio,
                'reb_margin': (avg['or'] + avg['dr']) - (avg['opp_or'] + avg['opp_dr']),
                'possessions': possessions,
                # Four Factors
                'efg_pct': efg_pct,
                'to_pct': to_pct,
                'or_pct': or_pct,
                'ft_rate': ft_rate,
                'opp_efg_pct': opp_efg_pct,
                'opp_to_pct': opp_to_pct,
                'opp_or_pct': opp_or_pct,
                'opp_ft_rate': opp_ft_rate,
            }

    def get_stats(self, season, team_id):
        """Get computed stats for a team in a season."""
        return self.stats.get((season, team_id))


def parse_seed_number(seed_str):
    """Extract numeric seed from seed string like 'W01' or 'X16a'."""
    if seed_str is None:
        return 16  # Default for unseeded teams
    # Remove region letter and any play-in suffix
    num_str = seed_str[1:3]
    try:
        return int(num_str)
    except ValueError:
        return 16


def seed_probability(seed_a, seed_b, coeff=0.30):
    """Probability estimate based purely on seeds."""
    diff = seed_b - seed_a  # Positive means team A has better (lower) seed
    return 1.0 / (1.0 + 10 ** (-diff * coeff))


def build_elo_ratings(data_dir):
    """Build Elo ratings from all historical game data."""
    print("Building Elo ratings from historical data...")

    men_elo = EloSystem(k=32, home_advantage=48, reversion=0.20, mov_coeff=0.7)
    women_elo = EloSystem(k=32, home_advantage=48, reversion=0.20, mov_coeff=0.7)

    # Load compact results
    m_reg = load_csv(find_data_file(data_dir, 'MRegularSeasonCompactResults.csv'))
    w_reg = load_csv(find_data_file(data_dir, 'WRegularSeasonCompactResults.csv'))
    m_tourney = load_csv(find_data_file(data_dir, 'MNCAATourneyCompactResults.csv'))
    w_tourney = load_csv(find_data_file(data_dir, 'WNCAATourneyCompactResults.csv'))

    if m_reg is None:
        print("ERROR: Could not find MRegularSeasonCompactResults.csv")
        print(f"  Looked in: {data_dir}")
        sys.exit(1)

    def process_games(games, elo_system, is_tourney=False):
        """Process games in chronological order."""
        if games is None:
            return

        # Sort by season and day
        sorted_games = sorted(games, key=lambda g: (int(g['Season']), int(g['DayNum'])))

        current_season = None
        for g in sorted_games:
            season = int(g['Season'])
            if season != current_season:
                if current_season is not None and not is_tourney:
                    elo_system.new_season()
                current_season = season

            w_id = g['WTeamID']
            l_id = g['LTeamID']
            w_score = int(g['WScore'])
            l_score = int(g['LScore'])
            loc = g.get('WLoc', 'N')
            margin = w_score - l_score

            elo_system.update(w_id, l_id, loc, margin)

    # Process regular season then tourney games
    process_games(m_reg, men_elo)
    process_games(m_tourney, men_elo, is_tourney=True)
    process_games(w_reg, women_elo)
    process_games(w_tourney, women_elo, is_tourney=True)

    print(f"  Men's teams rated: {len([k for k in men_elo.ratings if int(k) < 3000])}")
    print(f"  Women's teams rated: {len([k for k in women_elo.ratings if int(k) >= 3000])}")

    return men_elo, women_elo


def build_team_stats(data_dir, season=2026):
    """Build team statistics from detailed results."""
    print(f"Building team stats for season {season}...")

    stats = TeamStats()

    m_detailed = load_csv(find_data_file(data_dir, 'MRegularSeasonDetailedResults.csv'))
    w_detailed = load_csv(find_data_file(data_dir, 'WRegularSeasonDetailedResults.csv'))

    if m_detailed:
        stats.compute_from_detailed(m_detailed, season)
    if w_detailed:
        stats.compute_from_detailed(w_detailed, season)

    # Also compute for recent seasons for model training
    for s in range(max(2003, season - 10), season):
        if m_detailed:
            stats.compute_from_detailed(m_detailed, s)
        if w_detailed and s >= 2010:
            stats.compute_from_detailed(w_detailed, s)

    print(f"  Teams with stats: {len(stats.stats)}")
    return stats


def load_seeds(data_dir):
    """Load tournament seeds."""
    seeds = {}
    m_seeds = load_csv(find_data_file(data_dir, 'MNCAATourneySeeds.csv'))
    w_seeds = load_csv(find_data_file(data_dir, 'WNCAATourneySeeds.csv'))

    for seed_file in [m_seeds, w_seeds]:
        if seed_file is None:
            continue
        for row in seed_file:
            season = int(row['Season'])
            team_id = row['TeamID']
            seed = row['Seed']
            seeds[(season, team_id)] = seed

    return seeds


def load_ordinals(data_dir, season=2026):
    """Load Massey ordinal rankings for the current season."""
    print("Loading ordinal rankings...")
    ordinals = {}

    m_ordinals = load_csv(find_data_file(data_dir, 'MMasseyOrdinals.csv'))
    if m_ordinals is None:
        print("  No ordinal rankings found")
        return ordinals

    # Get the latest rankings for the season
    latest_day = defaultdict(lambda: defaultdict(lambda: -1))
    latest_rank = defaultdict(lambda: defaultdict(dict))

    for row in m_ordinals:
        s = int(row['Season'])
        if s != season:
            continue
        day = int(row['RankingDayNum'])
        system = row['SystemName']
        team_id = row['TeamID']
        rank = int(row['OrdinalRank'])

        if day > latest_day[system][team_id]:
            latest_day[system][team_id] = day
            latest_rank[system][team_id] = rank

    # Use average rank across systems for each team
    team_ranks = defaultdict(list)
    for system in latest_rank:
        for team_id, rank in latest_rank[system].items():
            team_ranks[team_id].append(rank)

    for team_id, ranks in team_ranks.items():
        ordinals[team_id] = sum(ranks) / len(ranks)

    print(f"  Teams with ordinal rankings: {len(ordinals)}")
    return ordinals


def compute_win_probability(elo_pred, seed_pred=None, stats_pred=None, ordinal_pred=None,
                            w_elo=0.35, w_seed=0.15, w_stats=0.30, w_ordinal=0.20):
    """Combine multiple prediction signals into final probability."""
    weights = []
    preds = []

    # Elo is always available and gets highest weight
    weights.append(w_elo)
    preds.append(elo_pred)

    if seed_pred is not None:
        weights.append(w_seed)
        preds.append(seed_pred)

    if stats_pred is not None:
        weights.append(w_stats)
        preds.append(stats_pred)

    if ordinal_pred is not None:
        weights.append(w_ordinal)
        preds.append(ordinal_pred)

    # Normalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # Weighted average in log-odds space for better calibration
    log_odds = 0.0
    for w, p in zip(weights, preds):
        p_clipped = max(0.001, min(0.999, p))
        log_odds += w * math.log(p_clipped / (1 - p_clipped))

    final_prob = 1.0 / (1.0 + math.exp(-log_odds))
    # Clip to reasonable bounds
    return max(0.01, min(0.99, final_prob))


def stats_prediction(stats_a, stats_b):
    """Predict based on team statistics comparison using Four Factors."""
    if stats_a is None or stats_b is None:
        return None

    # Combine net efficiency with Four Factors differential
    net_eff_diff = stats_a['net_eff'] - stats_b['net_eff']

    # Four Factors differentials (offensive advantage - defensive disadvantage)
    efg_diff = (stats_a.get('efg_pct', 0.45) - stats_b.get('opp_efg_pct', 0.45)) - \
               (stats_b.get('efg_pct', 0.45) - stats_a.get('opp_efg_pct', 0.45))
    to_diff = (stats_b.get('to_pct', 0.18) - stats_a.get('to_pct', 0.18)) + \
              (stats_a.get('opp_to_pct', 0.18) - stats_b.get('opp_to_pct', 0.18))
    or_diff = (stats_a.get('or_pct', 0.30) - stats_b.get('or_pct', 0.30)) + \
              (stats_b.get('opp_or_pct', 0.30) - stats_a.get('opp_or_pct', 0.30))
    ft_diff = (stats_a.get('ft_rate', 0.20) - stats_b.get('ft_rate', 0.20)) + \
              (stats_b.get('opp_ft_rate', 0.20) - stats_a.get('opp_ft_rate', 0.20))

    # Weighted composite: net efficiency is primary, four factors are secondary
    # Weights reflect importance in Dean Oliver's framework
    composite = (net_eff_diff * 0.06 +
                 efg_diff * 8.0 +    # eFG% most important
                 to_diff * 4.0 +     # Turnover rate
                 or_diff * 3.0 +     # Offensive rebounding
                 ft_diff * 2.0)      # Free throw rate

    return 1.0 / (1.0 + math.exp(-composite))


def ordinal_prediction(rank_a, rank_b, n_teams=360):
    """Predict based on ordinal rankings."""
    if rank_a is None or rank_b is None:
        return None

    # Normalize ranks and convert to probability
    # Higher rank (closer to 1) is better
    diff = rank_b - rank_a  # Positive means A is better ranked
    return 1.0 / (1.0 + 10 ** (-diff * 0.005))


class BradleyTerryModel:
    """Bradley-Terry model for team quality estimation.

    Fits team strength parameters from regular season results using
    iterative logistic regression (equivalent to GLMM random effects).
    This is the core of the RADDAR winning approach.
    """

    def __init__(self, lr=0.05, n_iterations=200, regularization=0.01):
        self.lr = lr
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.quality = {}  # (season, team_id) -> quality score

    def fit_season(self, games, season, tournament_teams=None):
        """Fit team qualities for a single season from game results.

        Args:
            games: List of game dicts with Season, WTeamID, LTeamID, WScore, LScore, NumOT
            season: Season to fit
            tournament_teams: If provided, only fit these teams (like RADDAR approach)
        """
        # Collect season games
        season_games = []
        teams_in_season = set()
        for g in games:
            if int(g['Season']) != season:
                continue
            w_id = g['WTeamID']
            l_id = g['LTeamID']
            num_ot = int(g.get('NumOT', 0))

            # Filter to tournament teams if specified
            if tournament_teams is not None:
                if w_id not in tournament_teams or l_id not in tournament_teams:
                    continue

            teams_in_season.add(w_id)
            teams_in_season.add(l_id)

            # Only use regulation-time results for quality (like RADDAR)
            if num_ot == 0:
                season_games.append((w_id, l_id))

        if not season_games:
            return

        # Initialize qualities to 0 (log-scale)
        q = {t: 0.0 for t in teams_in_season}

        # Iterative gradient descent on Bradley-Terry log-likelihood
        for _ in range(self.n_iterations):
            grad = {t: 0.0 for t in teams_in_season}

            for w_id, l_id in season_games:
                # P(w beats l) = sigmoid(q_w - q_l)
                diff = q[w_id] - q[l_id]
                prob = 1.0 / (1.0 + math.exp(-diff))

                # Gradient: for winner, we want to increase prob
                grad[w_id] += (1.0 - prob)
                grad[l_id] -= (1.0 - prob)

            # Update with gradient + L2 regularization
            for t in teams_in_season:
                q[t] += self.lr * (grad[t] / max(len(season_games), 1) - self.regularization * q[t])

        # Store as exponentiated quality (like RADDAR)
        for t in teams_in_season:
            self.quality[(season, t)] = math.exp(q[t])

    def get_quality(self, season, team_id):
        """Get team quality for prediction."""
        return self.quality.get((season, team_id))

    def predict(self, season, team_a, team_b):
        """Predict P(team_a beats team_b) using Bradley-Terry qualities."""
        q_a = self.quality.get((season, team_a))
        q_b = self.quality.get((season, team_b))
        if q_a is None or q_b is None:
            return None
        return q_a / (q_a + q_b)


class TrainedModel:
    """Logistic regression trained on historical tournament game features.

    Similar to the RADDAR XGBoost approach but using pure logistic regression
    (no external dependencies) with feature differentials.
    """

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.feature_names = []

    def _extract_features(self, team_stats_a, team_stats_b, seed_a, seed_b,
                          elo_pred, quality_a, quality_b, ordinal_a, ordinal_b,
                          momentum_a=None, momentum_b=None):
        """Extract differential features for a matchup."""
        features = {}

        # Elo prediction (log-odds)
        elo_clipped = max(0.01, min(0.99, elo_pred))
        features['elo_logodds'] = math.log(elo_clipped / (1 - elo_clipped))

        # Seed differential
        s_a = parse_seed_number(seed_a) if seed_a else 8.5
        s_b = parse_seed_number(seed_b) if seed_b else 8.5
        features['seed_diff'] = s_b - s_a  # Positive = A has better seed

        # Bradley-Terry quality ratio
        if quality_a is not None and quality_b is not None:
            features['bt_quality_ratio'] = math.log(max(quality_a, 0.01) / max(quality_b, 0.01))
        else:
            features['bt_quality_ratio'] = 0.0

        # Ordinal ranking differential
        if ordinal_a is not None and ordinal_b is not None:
            features['ordinal_diff'] = ordinal_b - ordinal_a
        else:
            features['ordinal_diff'] = 0.0

        # Momentum differential
        if momentum_a is not None and momentum_b is not None:
            features['momentum_diff'] = momentum_a - momentum_b
        else:
            features['momentum_diff'] = 0.0

        # Team stats differentials
        if team_stats_a is not None and team_stats_b is not None:
            features['net_eff_diff'] = team_stats_a['net_eff'] - team_stats_b['net_eff']
            features['off_eff_diff'] = team_stats_a['off_eff'] - team_stats_b['off_eff']
            features['def_eff_diff'] = team_stats_b['def_eff'] - team_stats_a['def_eff']  # Lower is better for defense
            features['ppg_diff'] = team_stats_a['ppg'] - team_stats_b['ppg']
            features['fg_pct_diff'] = team_stats_a['fg_pct'] - team_stats_b['fg_pct']
            features['fg3_pct_diff'] = team_stats_a['fg3_pct'] - team_stats_b['fg3_pct']
            features['ft_pct_diff'] = team_stats_a['ft_pct'] - team_stats_b['ft_pct']
            features['reb_margin_diff'] = team_stats_a['reb_margin'] - team_stats_b['reb_margin']
            features['ast_to_diff'] = team_stats_a['ast_to'] - team_stats_b['ast_to']
            # Four factors
            features['efg_diff'] = team_stats_a.get('efg_pct', 0.45) - team_stats_b.get('efg_pct', 0.45)
            features['to_pct_diff'] = team_stats_b.get('to_pct', 0.18) - team_stats_a.get('to_pct', 0.18)
            features['or_pct_diff'] = team_stats_a.get('or_pct', 0.30) - team_stats_b.get('or_pct', 0.30)
        else:
            for k in ['net_eff_diff', 'off_eff_diff', 'def_eff_diff', 'ppg_diff',
                       'fg_pct_diff', 'fg3_pct_diff', 'ft_pct_diff', 'reb_margin_diff',
                       'ast_to_diff', 'efg_diff', 'to_pct_diff', 'or_pct_diff']:
                features[k] = 0.0

        return features

    def train(self, training_data, lr=0.01, n_epochs=100, l2_reg=0.001):
        """Train logistic regression on feature vectors.

        Args:
            training_data: List of (features_dict, outcome) tuples
                           outcome is 1.0 if team_a won, 0.0 otherwise
        """
        if not training_data:
            return

        self.feature_names = sorted(training_data[0][0].keys())
        n_features = len(self.feature_names)
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for epoch in range(n_epochs):
            total_loss = 0.0

            for features, outcome in training_data:
                # Compute prediction
                z = self.bias
                for i, name in enumerate(self.feature_names):
                    z += self.weights[i] * features[name]

                pred = 1.0 / (1.0 + math.exp(-max(-20, min(20, z))))

                # Gradient
                error = pred - outcome
                total_loss += -(outcome * math.log(max(pred, 1e-10)) +
                                (1 - outcome) * math.log(max(1 - pred, 1e-10)))

                # Update weights
                self.bias -= lr * error
                for i, name in enumerate(self.feature_names):
                    self.weights[i] -= lr * (error * features[name] + l2_reg * self.weights[i])

        return total_loss / len(training_data)

    def predict(self, features):
        """Predict probability from features."""
        if self.weights is None:
            return 0.5

        z = self.bias
        for i, name in enumerate(self.feature_names):
            z += self.weights[i] * features.get(name, 0.0)

        pred = 1.0 / (1.0 + math.exp(-max(-20, min(20, z))))
        return max(0.01, min(0.99, pred))

    def get_feature_importance(self):
        """Return feature importances."""
        if self.weights is None:
            return {}
        return {name: abs(w) for name, w in zip(self.feature_names, self.weights)}


def generate_submission(data_dir, output_file, current_season=2026):
    """Generate the full submission file using trained model + ensemble blend."""
    print(f"\n=== March Madness {current_season} Prediction Generator ===\n")

    # Load sample submission to get required matchups
    sample_file = find_data_file(data_dir, 'SampleSubmissionStage2.csv')
    sample = load_csv(sample_file)
    if sample is None:
        print(f"ERROR: Could not find SampleSubmissionStage2.csv in {data_dir}")
        print("Trying to generate matchups from team lists...")
        sample = generate_matchups_from_teams(data_dir, current_season)

    if sample is None or len(sample) == 0:
        print("ERROR: No matchups to predict. Please ensure data files are available.")
        sys.exit(1)

    print(f"Total matchups to predict: {len(sample)}")

    # Build prediction components
    men_elo, women_elo = build_elo_ratings(data_dir)
    team_stats = build_team_stats(data_dir, current_season)
    seeds = load_seeds(data_dir)
    ordinals = load_ordinals(data_dir, current_season)

    # Build Bradley-Terry model
    print("Building Bradley-Terry team qualities...")
    m_reg = load_csv(find_data_file(data_dir, 'MRegularSeasonCompactResults.csv'))
    w_reg = load_csv(find_data_file(data_dir, 'WRegularSeasonCompactResults.csv'))
    bt_model = BradleyTerryModel()
    for reg_data in [m_reg, w_reg]:
        if reg_data:
            bt_model.fit_season(reg_data, current_season)

    # Train logistic regression on historical tournament data
    print("Training model on historical tournament data...")
    m_tourney = load_csv(find_data_file(data_dir, 'MNCAATourneyCompactResults.csv'))
    w_tourney = load_csv(find_data_file(data_dir, 'WNCAATourneyCompactResults.csv'))
    m_detailed = load_csv(find_data_file(data_dir, 'MRegularSeasonDetailedResults.csv'))
    w_detailed = load_csv(find_data_file(data_dir, 'WRegularSeasonDetailedResults.csv'))
    m_ordinals = load_csv(find_data_file(data_dir, 'MMasseyOrdinals.csv'))

    # Load ordinals for all seasons
    ordinals_by_season = {}
    if m_ordinals:
        latest_day = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: -1)))
        latest_rank = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for row in m_ordinals:
            s = int(row['Season'])
            day = int(row['RankingDayNum'])
            system = row['SystemName']
            team_id = row['TeamID']
            rank = int(row['OrdinalRank'])
            if day > latest_day[s][system][team_id]:
                latest_day[s][system][team_id] = day
                latest_rank[s][system][team_id] = rank
        for s in latest_rank:
            team_ranks = defaultdict(list)
            for system in latest_rank[s]:
                for team_id, rank in latest_rank[s][system].items():
                    team_ranks[team_id].append(rank)
            ordinals_by_season[s] = {tid: sum(r) / len(r) for tid, r in team_ranks.items()}

    trained_model = TrainedModel()
    training_data = []

    for train_s in range(max(2003, current_season - 15), current_season):
        # Build Elo for training season
        tr_men_elo = EloSystem(k=32, home_advantage=48, reversion=0.20, mov_coeff=0.7)
        tr_women_elo = EloSystem(k=32, home_advantage=48, reversion=0.20, mov_coeff=0.7)
        for reg_data, elo_sys in [(m_reg, tr_men_elo), (w_reg, tr_women_elo)]:
            if reg_data is None:
                continue
            tr_games = sorted([g for g in reg_data if int(g['Season']) <= train_s],
                              key=lambda g: (int(g['Season']), int(g['DayNum'])))
            cur_s = None
            for g in tr_games:
                s = int(g['Season'])
                if s != cur_s:
                    if cur_s is not None:
                        elo_sys.new_season()
                    cur_s = s
                margin = int(g['WScore']) - int(g['LScore'])
                elo_sys.update(g['WTeamID'], g['LTeamID'], g.get('WLoc', 'N'), margin)

        tr_stats = TeamStats()
        if m_detailed:
            tr_stats.compute_from_detailed(m_detailed, train_s)
        if w_detailed and train_s >= 2010:
            tr_stats.compute_from_detailed(w_detailed, train_s)

        tr_bt = BradleyTerryModel()
        tr_tourney_teams = set()
        for s_key in seeds:
            if s_key[0] == train_s:
                tr_tourney_teams.add(s_key[1])
        for reg_data in [m_reg, w_reg]:
            if reg_data:
                tr_bt.fit_season(reg_data, train_s,
                                 tournament_teams=tr_tourney_teams if tr_tourney_teams else None)

        tr_ordinals = ordinals_by_season.get(train_s, {})

        for tourney_data in [m_tourney, w_tourney]:
            if tourney_data is None:
                continue
            for g in tourney_data:
                if int(g['Season']) != train_s:
                    continue
                w_id, l_id = g['WTeamID'], g['LTeamID']
                if int(w_id) < int(l_id):
                    team_a, team_b, outcome = w_id, l_id, 1.0
                else:
                    team_a, team_b, outcome = l_id, w_id, 0.0

                is_women = int(team_a) >= 3000
                elo_sys = tr_women_elo if is_women else tr_men_elo
                elo_pred = elo_sys.predict(team_a, team_b)
                features = trained_model._extract_features(
                    tr_stats.get_stats(train_s, team_a), tr_stats.get_stats(train_s, team_b),
                    seeds.get((train_s, team_a)), seeds.get((train_s, team_b)),
                    elo_pred,
                    tr_bt.get_quality(train_s, team_a), tr_bt.get_quality(train_s, team_b),
                    tr_ordinals.get(team_a), tr_ordinals.get(team_b),
                    elo_sys.get_momentum(team_a), elo_sys.get_momentum(team_b),
                )
                training_data.append((features, outcome))

    if training_data:
        trained_model.train(training_data, lr=0.005, n_epochs=200, l2_reg=0.001)
        print(f"  Trained on {len(training_data)} historical tournament games")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    men_count = 0
    women_count = 0

    for row in sample:
        matchup_id = row['ID']
        parts = matchup_id.split('_')
        season = int(parts[0])
        team_a = parts[1]
        team_b = parts[2]

        is_women = int(team_a) >= 3000

        # Elo prediction
        elo_system = women_elo if is_women else men_elo
        elo_pred = elo_system.predict(team_a, team_b)

        # Seed prediction
        seed_a = seeds.get((season, team_a))
        seed_b = seeds.get((season, team_b))
        if seed_a and seed_b:
            seed_pred = seed_probability(parse_seed_number(seed_a), parse_seed_number(seed_b))
        else:
            seed_pred = None

        # Stats prediction
        s_a = team_stats.get_stats(season, team_a)
        s_b = team_stats.get_stats(season, team_b)
        stats_pred = stats_prediction(s_a, s_b)

        # Ordinal prediction
        ord_a = ordinals.get(team_a)
        ord_b = ordinals.get(team_b)
        ord_pred = ordinal_prediction(ord_a, ord_b) if (ord_a and ord_b) else None

        # Simple ensemble
        ensemble_pred = compute_win_probability(elo_pred, seed_pred, stats_pred, ord_pred)

        # Trained model prediction
        features = trained_model._extract_features(
            s_a, s_b, seed_a, seed_b, elo_pred,
            bt_model.get_quality(season, team_a), bt_model.get_quality(season, team_b),
            ordinals.get(team_a) if not is_women else None,
            ordinals.get(team_b) if not is_women else None,
            elo_system.get_momentum(team_a), elo_system.get_momentum(team_b),
        )
        trained_pred = trained_model.predict(features)

        # Blend: 20% trained model, 80% ensemble (optimized via backtest)
        final_prob = 0.20 * trained_pred + 0.80 * ensemble_pred

        predictions.append({'ID': matchup_id, 'Pred': f'{final_prob:.6f}'})

        if is_women:
            women_count += 1
        else:
            men_count += 1

    # Write submission file
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'Pred'])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"\nSubmission file written to: {output_file}")
    print(f"  Men's matchups: {men_count}")
    print(f"  Women's matchups: {women_count}")
    print(f"  Total predictions: {len(predictions)}")

    # Validation
    validate_submission(predictions)


def generate_matchups_from_teams(data_dir, season):
    """Generate all possible matchups from team lists if no sample submission exists."""
    m_teams = load_csv(find_data_file(data_dir, 'MTeams.csv'))
    w_teams = load_csv(find_data_file(data_dir, 'WTeams.csv'))

    if m_teams is None or w_teams is None:
        return None

    matchups = []

    # Men's teams - filter to current D1 teams
    men_ids = []
    for t in m_teams:
        last_season = int(t.get('LastD1Season', season))
        first_season = int(t.get('FirstD1Season', 1985))
        if first_season <= season <= last_season:
            men_ids.append(t['TeamID'])

    # Women's teams
    women_ids = [t['TeamID'] for t in w_teams]

    print(f"  Men's D1 teams: {len(men_ids)}")
    print(f"  Women's teams: {len(women_ids)}")

    # Generate all pairwise matchups
    for team_list in [men_ids, women_ids]:
        team_list_sorted = sorted(team_list, key=int)
        for i in range(len(team_list_sorted)):
            for j in range(i + 1, len(team_list_sorted)):
                matchup_id = f"{season}_{team_list_sorted[i]}_{team_list_sorted[j]}"
                matchups.append({'ID': matchup_id, 'Pred': '0.5'})

    print(f"  Total matchups generated: {len(matchups)}")
    return matchups


def validate_submission(predictions):
    """Validate the submission format."""
    print("\nValidation:")
    issues = []

    for p in predictions:
        pred = float(p['Pred'])
        if pred < 0 or pred > 1:
            issues.append(f"  Invalid probability {pred} for {p['ID']}")
        parts = p['ID'].split('_')
        if len(parts) != 3:
            issues.append(f"  Invalid ID format: {p['ID']}")
        elif int(parts[1]) >= int(parts[2]):
            issues.append(f"  Team IDs not in order: {p['ID']}")

    if issues:
        print(f"  ISSUES FOUND: {len(issues)}")
        for issue in issues[:10]:
            print(issue)
    else:
        print("  All predictions valid!")

    # Distribution stats
    probs = [float(p['Pred']) for p in predictions]
    print(f"  Prob range: [{min(probs):.4f}, {max(probs):.4f}]")
    print(f"  Mean prob: {sum(probs) / len(probs):.4f}")
    print(f"  Predictions near 0.5: {sum(1 for p in probs if 0.45 <= p <= 0.55)}")
    print(f"  Strong predictions (>0.8 or <0.2): {sum(1 for p in probs if p > 0.8 or p < 0.2)}")


def backtest(data_dir, test_seasons=None, elo_k=32, elo_home=48, elo_reversion=0.20,
             mov_coeff=0.7, seed_coeff=0.30, blend_trained=0.20, verbose=True,
             w_elo=0.35, w_seed=0.15, w_stats=0.30, w_ordinal=0.20):
    """Backtest the trained model against historical tournament results.

    Uses leave-one-season-out cross-validation: for each test season,
    trains on all other seasons' tournament data.
    """
    if test_seasons is None:
        test_seasons = [2022, 2023, 2024, 2025]

    if verbose:
        print(f"\n=== Backtesting on seasons {test_seasons} ===\n")

    # Load all data
    m_tourney = load_csv(find_data_file(data_dir, 'MNCAATourneyCompactResults.csv'))
    w_tourney = load_csv(find_data_file(data_dir, 'WNCAATourneyCompactResults.csv'))
    m_reg = load_csv(find_data_file(data_dir, 'MRegularSeasonCompactResults.csv'))
    w_reg = load_csv(find_data_file(data_dir, 'WRegularSeasonCompactResults.csv'))
    m_detailed = load_csv(find_data_file(data_dir, 'MRegularSeasonDetailedResults.csv'))
    w_detailed = load_csv(find_data_file(data_dir, 'WRegularSeasonDetailedResults.csv'))
    seeds = load_seeds(data_dir)

    if m_tourney is None:
        if verbose:
            print("ERROR: No tournament results found for backtesting")
        return None, None

    # Load ordinals for all seasons
    m_ordinals = load_csv(find_data_file(data_dir, 'MMasseyOrdinals.csv'))
    ordinals_by_season = {}
    if m_ordinals:
        latest_day = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: -1)))
        latest_rank = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for row in m_ordinals:
            s = int(row['Season'])
            day = int(row['RankingDayNum'])
            system = row['SystemName']
            team_id = row['TeamID']
            rank = int(row['OrdinalRank'])
            if day > latest_day[s][system][team_id]:
                latest_day[s][system][team_id] = day
                latest_rank[s][system][team_id] = rank
        for s in latest_rank:
            team_ranks = defaultdict(list)
            for system in latest_rank[s]:
                for team_id, rank in latest_rank[s][system].items():
                    team_ranks[team_id].append(rank)
            ordinals_by_season[s] = {tid: sum(r) / len(r) for tid, r in team_ranks.items()}

    total_brier = 0.0
    total_games = 0
    total_correct = 0

    for test_season in test_seasons:
        # Build Elo using data up to this season
        men_elo = EloSystem(k=elo_k, home_advantage=elo_home, reversion=elo_reversion, mov_coeff=mov_coeff)
        women_elo = EloSystem(k=elo_k, home_advantage=elo_home, reversion=elo_reversion, mov_coeff=mov_coeff)

        for reg_data, elo_sys in [(m_reg, men_elo), (w_reg, women_elo)]:
            if reg_data is None:
                continue
            train_games = [g for g in reg_data if int(g['Season']) <= test_season]
            train_games.sort(key=lambda g: (int(g['Season']), int(g['DayNum'])))
            current_season = None
            for g in train_games:
                s = int(g['Season'])
                if s != current_season:
                    if current_season is not None:
                        elo_sys.new_season()
                    current_season = s
                margin = int(g['WScore']) - int(g['LScore'])
                elo_sys.update(g['WTeamID'], g['LTeamID'], g.get('WLoc', 'N'), margin)

        # Build team stats for this season
        team_stats = TeamStats()
        if m_detailed:
            team_stats.compute_from_detailed(m_detailed, test_season)
        if w_detailed:
            team_stats.compute_from_detailed(w_detailed, test_season)

        # Build Bradley-Terry qualities for this season
        bt_model = BradleyTerryModel()
        # Get tournament teams for this season
        tourney_teams = set()
        for s_key, s_val in seeds.items():
            if s_key[0] == test_season:
                tourney_teams.add(s_key[1])

        for reg_data in [m_reg, w_reg]:
            if reg_data:
                bt_model.fit_season(reg_data, test_season,
                                    tournament_teams=tourney_teams if tourney_teams else None)

        # Get ordinals for this season
        ordinals = ordinals_by_season.get(test_season, {})

        # Collect training data from past tournament seasons (leave-one-out)
        training_data = []
        trained_model = TrainedModel()

        # Build features for historical tournament games (training)
        train_seasons = range(max(2003, test_season - 15), test_season)
        for train_s in train_seasons:
            # Build Elo for training season
            tr_men_elo = EloSystem(k=elo_k, home_advantage=elo_home, reversion=elo_reversion, mov_coeff=mov_coeff)
            tr_women_elo = EloSystem(k=elo_k, home_advantage=elo_home, reversion=elo_reversion, mov_coeff=mov_coeff)
            for reg_data, elo_sys in [(m_reg, tr_men_elo), (w_reg, tr_women_elo)]:
                if reg_data is None:
                    continue
                tr_games = [g for g in reg_data if int(g['Season']) <= train_s]
                tr_games.sort(key=lambda g: (int(g['Season']), int(g['DayNum'])))
                cur_s = None
                for g in tr_games:
                    s = int(g['Season'])
                    if s != cur_s:
                        if cur_s is not None:
                            elo_sys.new_season()
                        cur_s = s
                    margin = int(g['WScore']) - int(g['LScore'])
                    elo_sys.update(g['WTeamID'], g['LTeamID'], g.get('WLoc', 'N'), margin)

            # Build stats for training season
            tr_stats = TeamStats()
            if m_detailed:
                tr_stats.compute_from_detailed(m_detailed, train_s)
            if w_detailed and train_s >= 2010:
                tr_stats.compute_from_detailed(w_detailed, train_s)

            # Build BT for training season
            tr_bt = BradleyTerryModel()
            tr_tourney_teams = set()
            for s_key, s_val in seeds.items():
                if s_key[0] == train_s:
                    tr_tourney_teams.add(s_key[1])
            for reg_data in [m_reg, w_reg]:
                if reg_data:
                    tr_bt.fit_season(reg_data, train_s,
                                     tournament_teams=tr_tourney_teams if tr_tourney_teams else None)

            tr_ordinals = ordinals_by_season.get(train_s, {})

            # Extract features from tournament games
            for tourney_data in [m_tourney, w_tourney]:
                if tourney_data is None:
                    continue
                for g in tourney_data:
                    if int(g['Season']) != train_s:
                        continue
                    w_id = g['WTeamID']
                    l_id = g['LTeamID']

                    if int(w_id) < int(l_id):
                        team_a, team_b = w_id, l_id
                        outcome = 1.0
                    else:
                        team_a, team_b = l_id, w_id
                        outcome = 0.0

                    is_women = int(team_a) >= 3000
                    elo_sys = tr_women_elo if is_women else tr_men_elo
                    elo_pred = elo_sys.predict(team_a, team_b)

                    features = trained_model._extract_features(
                        tr_stats.get_stats(train_s, team_a),
                        tr_stats.get_stats(train_s, team_b),
                        seeds.get((train_s, team_a)),
                        seeds.get((train_s, team_b)),
                        elo_pred,
                        tr_bt.get_quality(train_s, team_a),
                        tr_bt.get_quality(train_s, team_b),
                        tr_ordinals.get(team_a),
                        tr_ordinals.get(team_b),
                        elo_sys.get_momentum(team_a),
                        elo_sys.get_momentum(team_b),
                    )
                    training_data.append((features, outcome))

        # Train the model
        if training_data:
            trained_model.train(training_data, lr=0.005, n_epochs=200, l2_reg=0.001)

        # Test on this season's tournament games
        season_brier = 0.0
        season_games = 0
        season_correct = 0

        for tourney_data in [m_tourney, w_tourney]:
            if tourney_data is None:
                continue
            for g in tourney_data:
                if int(g['Season']) != test_season:
                    continue

                w_id = g['WTeamID']
                l_id = g['LTeamID']

                if int(w_id) < int(l_id):
                    team_a, team_b = w_id, l_id
                    actual = 1.0
                else:
                    team_a, team_b = l_id, w_id
                    actual = 0.0

                is_women = int(team_a) >= 3000
                elo_system = women_elo if is_women else men_elo
                elo_pred = elo_system.predict(team_a, team_b)

                features = trained_model._extract_features(
                    team_stats.get_stats(test_season, team_a),
                    team_stats.get_stats(test_season, team_b),
                    seeds.get((test_season, team_a)),
                    seeds.get((test_season, team_b)),
                    elo_pred,
                    bt_model.get_quality(test_season, team_a),
                    bt_model.get_quality(test_season, team_b),
                    ordinals.get(team_a),
                    ordinals.get(team_b),
                    elo_system.get_momentum(team_a),
                    elo_system.get_momentum(team_b),
                )

                # Blend trained model with simple ensemble
                trained_pred = trained_model.predict(features)

                # Simple ensemble prediction (seed + elo + stats)
                seed_a_val = seeds.get((test_season, team_a))
                seed_b_val = seeds.get((test_season, team_b))
                seed_pred = None
                if seed_a_val and seed_b_val:
                    seed_pred = seed_probability(
                        parse_seed_number(seed_a_val), parse_seed_number(seed_b_val), coeff=seed_coeff)

                s_a = team_stats.get_stats(test_season, team_a)
                s_b = team_stats.get_stats(test_season, team_b)
                sp = stats_prediction(s_a, s_b)

                ord_a = ordinals.get(team_a)
                ord_b = ordinals.get(team_b)
                ord_pred = ordinal_prediction(ord_a, ord_b) if (ord_a and ord_b) else None

                ensemble_pred = compute_win_probability(elo_pred, seed_pred, sp, ord_pred,
                                                       w_elo=w_elo, w_seed=w_seed,
                                                       w_stats=w_stats, w_ordinal=w_ordinal)

                # Blend trained model with simple ensemble
                pred = blend_trained * trained_pred + (1 - blend_trained) * ensemble_pred

                brier = (pred - actual) ** 2
                season_brier += brier
                season_games += 1

                if (pred > 0.5) == (actual > 0.5):
                    season_correct += 1

        if season_games > 0:
            avg_brier = season_brier / season_games
            accuracy = season_correct / season_games
            if verbose:
                print(f"  Season {test_season}: Brier={avg_brier:.4f}, "
                      f"Accuracy={accuracy:.1%} ({season_correct}/{season_games})")
            total_brier += season_brier
            total_games += season_games
            total_correct += season_correct

    if total_games > 0:
        overall_brier = total_brier / total_games
        overall_accuracy = total_correct / total_games
        if verbose:
            print(f"\n  Overall: Brier={overall_brier:.4f}, "
                  f"Accuracy={overall_accuracy:.1%} ({total_correct}/{total_games})")
            print(f"  (Lower Brier score is better. Baseline 0.25 for 50/50 predictions)")
            if training_data and trained_model.weights is not None:
                importance = trained_model.get_feature_importance()
                sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
                print(f"\n  Feature importance (top 5):")
                for name, imp in sorted_imp[:5]:
                    print(f"    {name}: {imp:.4f}")
        return overall_brier, overall_accuracy
    return None, None


def download_data():
    """Download competition data using kagglehub."""
    try:
        import kagglehub
        print("Downloading data from Kaggle...")
        path = kagglehub.competition_download('march-machine-learning-mania-2026')
        print(f"Data downloaded to: {path}")
        return str(path)
    except ImportError:
        print("kagglehub not installed. Install with: pip install kagglehub")
        print("Then set KAGGLE_API_TOKEN environment variable.")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to download data: {e}")
        print("Set KAGGLE_API_TOKEN env var or use --data-dir to specify local data.")
        sys.exit(1)


def grid_search(data_dir):
    """Grid search over key hyperparameters to find optimal settings."""
    print("\n=== Grid Search ===\n")

    best_brier = 1.0
    best_params = {}

    # Coarse grid first
    param_grid = {
        'elo_k': [20, 28, 32, 40],
        'elo_home': [48, 64, 80, 100],
        'elo_reversion': [0.20, 0.25, 0.30, 0.35],
        'mov_coeff': [0.4, 0.5, 0.6, 0.7],
    }

    # Start with weight optimization using default Elo params
    print("Phase 1: Optimizing ensemble weights...")
    weight_configs = [
        (0.35, 0.10, 0.35, 0.20),
        (0.40, 0.10, 0.30, 0.20),
        (0.40, 0.05, 0.35, 0.20),
        (0.45, 0.10, 0.25, 0.20),
        (0.35, 0.05, 0.40, 0.20),
        (0.30, 0.10, 0.35, 0.25),
        (0.35, 0.15, 0.30, 0.20),
        (0.40, 0.10, 0.25, 0.25),
    ]

    for w_elo, w_seed, w_stats, w_ordinal in weight_configs:
        brier, acc = backtest(data_dir, verbose=False,
                              w_elo=w_elo, w_seed=w_seed, w_stats=w_stats, w_ordinal=w_ordinal)
        if brier is not None and brier < best_brier:
            best_brier = brier
            best_params = {'w_elo': w_elo, 'w_seed': w_seed, 'w_stats': w_stats, 'w_ordinal': w_ordinal}
            print(f"  New best: Brier={brier:.4f}, weights=({w_elo},{w_seed},{w_stats},{w_ordinal})")

    print(f"\nBest weights: {best_params}")
    w_elo = best_params.get('w_elo', 0.40)
    w_seed = best_params.get('w_seed', 0.10)
    w_stats = best_params.get('w_stats', 0.30)
    w_ordinal = best_params.get('w_ordinal', 0.20)

    # Phase 2: Optimize Elo parameters with best weights
    print("\nPhase 2: Optimizing Elo parameters...")
    for k in param_grid['elo_k']:
        for home in param_grid['elo_home']:
            for rev in param_grid['elo_reversion']:
                for mov in param_grid['mov_coeff']:
                    brier, acc = backtest(data_dir, verbose=False,
                                          elo_k=k, elo_home=home, elo_reversion=rev, mov_coeff=mov,
                                          w_elo=w_elo, w_seed=w_seed, w_stats=w_stats, w_ordinal=w_ordinal)
                    if brier is not None and brier < best_brier:
                        best_brier = brier
                        best_params.update({'elo_k': k, 'elo_home': home, 'elo_reversion': rev, 'mov_coeff': mov})
                        print(f"  New best: Brier={brier:.4f}, k={k}, home={home}, rev={rev}, mov={mov}")

    # Phase 3: Optimize seed coefficient
    print("\nPhase 3: Optimizing seed coefficient...")
    for seed_c in [0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]:
        brier, acc = backtest(data_dir, verbose=False, seed_coeff=seed_c, **{
            k: v for k, v in best_params.items() if k != 'seed_coeff'
        })
        if brier is not None and brier < best_brier:
            best_brier = brier
            best_params['seed_coeff'] = seed_c
            print(f"  New best: Brier={brier:.4f}, seed_coeff={seed_c}")

    print(f"\n=== Best parameters: Brier={best_brier:.4f} ===")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    # Run final backtest with best params, verbose
    print("\nFinal backtest with best parameters:")
    backtest(data_dir, verbose=True, **best_params)

    return best_params, best_brier


def main():
    parser = argparse.ArgumentParser(description='March Madness 2026 Predictions')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing competition data files. '
                             'If not specified, downloads from Kaggle via kagglehub.')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output submission CSV file')
    parser.add_argument('--season', type=int, default=2026,
                        help='Season to predict')
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtesting on historical data')
    parser.add_argument('--grid-search', action='store_true',
                        help='Run grid search over hyperparameters')

    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else download_data()

    if args.grid_search:
        grid_search(data_dir)
    elif args.backtest:
        backtest(data_dir)
    else:
        generate_submission(data_dir, args.output, args.season)


if __name__ == '__main__':
    main()
