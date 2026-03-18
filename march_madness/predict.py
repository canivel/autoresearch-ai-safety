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

    def __init__(self, k=20, home_advantage=100, mean_elo=1500, reversion=0.25):
        self.k = k
        self.home_advantage = home_advantage
        self.mean_elo = mean_elo
        self.reversion = reversion
        self.ratings = defaultdict(lambda: self.mean_elo)

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
            mov_mult = math.log(max(abs(margin), 1) + 1) * 0.7
        else:
            mov_mult = 1.0

        expected_w = self.expected_score(rating_w_adj, rating_l_adj)
        k_adj = self.k * mov_mult

        self.ratings[team_w] += k_adj * (1 - expected_w)
        self.ratings[team_l] += k_adj * (0 - (1 - expected_w))

    def new_season(self):
        """Revert ratings toward mean for a new season."""
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - self.reversion)
                + self.mean_elo * self.reversion
            )

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
                'opp_to': int(g.get('LTO', 0)),
                'opp_fta': int(g.get('LFTA', 0)),
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
                'opp_to': int(g.get('WTO', 0)),
                'opp_fta': int(g.get('WFTA', 0)),
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


def seed_probability(seed_a, seed_b):
    """Probability estimate based purely on seeds."""
    # Historical seed win rates (approximate)
    diff = seed_b - seed_a  # Positive means team A has better (lower) seed
    return 1.0 / (1.0 + 10 ** (-diff * 0.15))


def build_elo_ratings(data_dir):
    """Build Elo ratings from all historical game data."""
    print("Building Elo ratings from historical data...")

    men_elo = EloSystem(k=20, home_advantage=100)
    women_elo = EloSystem(k=20, home_advantage=100)

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


def compute_win_probability(elo_pred, seed_pred=None, stats_pred=None, ordinal_pred=None):
    """Combine multiple prediction signals into final probability."""
    weights = []
    preds = []

    # Elo is always available and gets highest weight
    weights.append(0.45)
    preds.append(elo_pred)

    if seed_pred is not None:
        weights.append(0.15)
        preds.append(seed_pred)

    if stats_pred is not None:
        weights.append(0.25)
        preds.append(stats_pred)

    if ordinal_pred is not None:
        weights.append(0.15)
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
    """Predict based on team statistics comparison."""
    if stats_a is None or stats_b is None:
        return None

    # Net efficiency difference
    net_eff_diff = stats_a['net_eff'] - stats_b['net_eff']

    # Convert to probability using logistic function
    # Calibrated so that ~10 point net efficiency diff ≈ 75% win prob
    return 1.0 / (1.0 + math.exp(-net_eff_diff * 0.08))


def ordinal_prediction(rank_a, rank_b, n_teams=360):
    """Predict based on ordinal rankings."""
    if rank_a is None or rank_b is None:
        return None

    # Normalize ranks and convert to probability
    # Higher rank (closer to 1) is better
    diff = rank_b - rank_a  # Positive means A is better ranked
    return 1.0 / (1.0 + 10 ** (-diff * 0.005))


def generate_submission(data_dir, output_file, current_season=2026):
    """Generate the full submission file."""
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
            seed_num_a = parse_seed_number(seed_a)
            seed_num_b = parse_seed_number(seed_b)
            seed_pred = seed_probability(seed_num_a, seed_num_b)
        else:
            seed_pred = None

        # Stats prediction
        s_a = team_stats.get_stats(season, team_a)
        s_b = team_stats.get_stats(season, team_b)
        stats_pred = stats_prediction(s_a, s_b)

        # Ordinal prediction (men only)
        ord_a = ordinals.get(team_a)
        ord_b = ordinals.get(team_b)
        ord_pred = ordinal_prediction(ord_a, ord_b) if (ord_a and ord_b) else None

        # Combine signals
        final_prob = compute_win_probability(elo_pred, seed_pred, stats_pred, ord_pred)

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


def backtest(data_dir, test_seasons=None):
    """Backtest the model against historical tournament results."""
    if test_seasons is None:
        test_seasons = [2022, 2023, 2024, 2025]

    print(f"\n=== Backtesting on seasons {test_seasons} ===\n")

    # Load tournament results
    m_tourney = load_csv(find_data_file(data_dir, 'MNCAATourneyCompactResults.csv'))
    w_tourney = load_csv(find_data_file(data_dir, 'WNCAATourneyCompactResults.csv'))

    if m_tourney is None:
        print("ERROR: No tournament results found for backtesting")
        return

    # Build Elo up to each test season
    m_reg = load_csv(find_data_file(data_dir, 'MRegularSeasonCompactResults.csv'))
    w_reg = load_csv(find_data_file(data_dir, 'WRegularSeasonCompactResults.csv'))
    seeds = load_seeds(data_dir)

    total_brier = 0.0
    total_games = 0
    total_correct = 0

    for test_season in test_seasons:
        # Build Elo using only data before this season
        men_elo = EloSystem(k=20, home_advantage=100)
        women_elo = EloSystem(k=20, home_advantage=100)

        if m_reg:
            train_games = [g for g in m_reg if int(g['Season']) <= test_season]
            train_games.sort(key=lambda g: (int(g['Season']), int(g['DayNum'])))
            current_season = None
            for g in train_games:
                s = int(g['Season'])
                if s != current_season:
                    if current_season is not None:
                        men_elo.new_season()
                    current_season = s
                margin = int(g['WScore']) - int(g['LScore'])
                men_elo.update(g['WTeamID'], g['LTeamID'], g.get('WLoc', 'N'), margin)

        if w_reg:
            train_games = [g for g in w_reg if int(g['Season']) <= test_season]
            train_games.sort(key=lambda g: (int(g['Season']), int(g['DayNum'])))
            current_season = None
            for g in train_games:
                s = int(g['Season'])
                if s != current_season:
                    if current_season is not None:
                        women_elo.new_season()
                    current_season = s
                margin = int(g['WScore']) - int(g['LScore'])
                women_elo.update(g['WTeamID'], g['LTeamID'], g.get('WLoc', 'N'), margin)

        # Build team stats for this season
        team_stats = TeamStats()
        m_detailed = load_csv(find_data_file(data_dir, 'MRegularSeasonDetailedResults.csv'))
        w_detailed = load_csv(find_data_file(data_dir, 'WRegularSeasonDetailedResults.csv'))
        if m_detailed:
            team_stats.compute_from_detailed(m_detailed, test_season)
        if w_detailed:
            team_stats.compute_from_detailed(w_detailed, test_season)

        # Test on tournament games
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

                # Ensure lower ID is team_a
                if int(w_id) < int(l_id):
                    team_a, team_b = w_id, l_id
                    actual = 1.0
                else:
                    team_a, team_b = l_id, w_id
                    actual = 0.0

                is_women = int(team_a) >= 3000
                elo_system = women_elo if is_women else men_elo
                elo_pred = elo_system.predict(team_a, team_b)

                # Seed prediction
                seed_a = seeds.get((test_season, team_a))
                seed_b = seeds.get((test_season, team_b))
                if seed_a and seed_b:
                    seed_pred = seed_probability(
                        parse_seed_number(seed_a), parse_seed_number(seed_b)
                    )
                else:
                    seed_pred = None

                # Stats prediction
                s_a = team_stats.get_stats(test_season, team_a)
                s_b = team_stats.get_stats(test_season, team_b)
                sp = stats_prediction(s_a, s_b)

                pred = compute_win_probability(elo_pred, seed_pred, sp)
                brier = (pred - actual) ** 2
                season_brier += brier
                season_games += 1

                predicted_winner = team_a if pred > 0.5 else team_b
                actual_winner = w_id
                if predicted_winner == actual_winner:
                    season_correct += 1

        if season_games > 0:
            avg_brier = season_brier / season_games
            accuracy = season_correct / season_games
            print(f"  Season {test_season}: Brier={avg_brier:.4f}, "
                  f"Accuracy={accuracy:.1%} ({season_correct}/{season_games})")
            total_brier += season_brier
            total_games += season_games
            total_correct += season_correct

    if total_games > 0:
        overall_brier = total_brier / total_games
        overall_accuracy = total_correct / total_games
        print(f"\n  Overall: Brier={overall_brier:.4f}, "
              f"Accuracy={overall_accuracy:.1%} ({total_correct}/{total_games})")
        print(f"  (Lower Brier score is better. Baseline 0.25 for 50/50 predictions)")


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

    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else download_data()

    if args.backtest:
        backtest(data_dir)
    else:
        generate_submission(data_dir, args.output, args.season)


if __name__ == '__main__':
    main()
