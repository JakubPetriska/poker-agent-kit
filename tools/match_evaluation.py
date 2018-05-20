import numpy as np
from scipy.stats import sem, norm

import acpc_python_client as acpc


def get_player_final_utilities_from_log_file(log_file_path):
    with open(log_file_path, 'r') as strategy_file:
        for line in map(lambda line: line.strip(), strategy_file):
            if line.startswith('SCORE'):
                line_segments = line.split(':')
                player_names = line_segments[2].split('|')
                scores = [float(score) for score in line_segments[1].split('|')]
                return scores, player_names
    raise AttributeError('Log file does not contain SCORE line')


def get_player_utilities_from_log_file(
        log_file_path,
        game_file_path=None,
        utility_estimator=None,
        player_strategies=None,
        evaluated_strategies=None):
    player_names = None
    num_hands = 0

    if utility_estimator is not None and game_file_path is None:
        raise AttributeError('Game file path must be provided with utility estimator')

    with open(log_file_path, 'r') as strategy_file:
        for line in map(lambda line: line.strip(), strategy_file):
            if line.startswith('STATE'):
                num_hands += 1
            if line.startswith('SCORE'):
                player_names = line.split(':')[2].split('|')
                break
    if not player_names:
        raise AttributeError('Log file does not contain SCORE line')

    num_evaluated_strategies = 1 if evaluated_strategies is None else len(evaluated_strategies)
    player_utilities = np.zeros([num_hands, len(player_names), num_evaluated_strategies])

    with open(log_file_path, 'r') as strategy_file:
        for line in map(lambda line: line.strip(), strategy_file):
            if line.startswith('STATE'):
                line_segments = line.split(':')
                hand_index = int(float(line_segments[1]))
                scores = [float(score) for score in line_segments[-2].split('|')]
                players = line_segments[-1].split('|')
                state = None
                if utility_estimator is not None:
                    state = acpc.parse_state(game_file_path, line)
                for i, player_name in enumerate(players):
                    player_index = player_names.index(player_name)
                    if utility_estimator is not None and player_name in player_strategies:
                        player_strategy = player_strategies[player_name]
                        utility_estimates = utility_estimator.get_utility_estimations(state, i, player_strategy, evaluated_strategies)
                        player_utilities[hand_index, player_index] = utility_estimates
                    else:
                        player_utilities[hand_index, player_index] = scores[i]

    return player_utilities, player_names

def get_logs_data(*log_readings):
    num_matches = len(log_readings)
    num_match_hands = None
    num_evaluated_strategies = None
    num_players = None
    player_names = None
    for log_reading in log_readings:
        utilities, log_player_names = log_reading
        if num_match_hands is None:
            num_match_hands = utilities.shape[0]
            num_players = utilities.shape[1]
            num_evaluated_strategies = utilities.shape[2]
            player_names = list(sorted(log_player_names))
        elif utilities.shape[0] != num_match_hands:
            raise AttributeError('Log readings must contain same number of hands')
        elif utilities.shape[1] != num_players:
            raise AttributeError('Log readings must contain same number of players')
        elif utilities.shape[2] != num_evaluated_strategies:
            raise AttributeError('Log readings must contain same number of evaluated strategies')
        elif list(sorted(log_player_names)) != player_names:
            raise AttributeError('Log readings must contain same set of players')

    num_total_hands = num_matches * num_match_hands
    data = np.empty([num_total_hands, num_players, num_evaluated_strategies])
    for i, log_reading in enumerate(log_readings):
        utilities, log_player_names = log_reading
        for p in range(num_players):
            player_name = log_player_names[p]
            start_hand_index = i * num_match_hands
            end_hand_index = start_hand_index + num_match_hands
            data[start_hand_index:end_hand_index, player_names.index(player_name)] = utilities[:, p]
    return data, player_names

def calculate_confidence_interval(data, confidence):
    num_total_hands = data.shape[0]

    means = np.mean(data, axis=0)
    standard_error = np.std(data, axis=0) / np.sqrt(num_total_hands)

    alpha = 1 - confidence
    phi = 1 - (alpha / 2)
    z = norm.ppf(phi)
    interval_half_size = z * standard_error
    lower_bounds = means - interval_half_size
    upper_bounds = means + interval_half_size

    return means, interval_half_size, lower_bounds, upper_bounds
