import unittest
import os
import shutil
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import acpc_python_client as acpc

from tools.constants import Action
from weak_agents.action_tilted_agent import create_agent_strategy_from_trained_strategy, TiltType
from tools.io_util import read_strategy_from_file, write_strategy_to_file
from implicit_modelling.build_portfolio import train_portfolio_responses, optimize_portfolio
from evaluation.exploitability import Exploitability


TEST_DIRECTORY = 'verification/implicit_agent'
TEST_OUTPUT_DIRECTORY = '%s/portfolios' % TEST_DIRECTORY

BASE_AGENT_SCRIPT_PATH = '%s/base_agent_script.sh' % TEST_DIRECTORY
BASE_OPPONENT_SCRIPT_PATH = '%s/base_opponent_script.sh' % TEST_DIRECTORY

REPLACE_STRING_COMMENT = '###COMMENT###'
REPLACE_STRING_GAME_FILE_PATH = '###GAME_FILE_PATH###'
REPLACE_STRING_ENVIRONMENT_ACTIVATION = '###ENVIRONMENT_ACTIVATION###'
OPPONENT_SCRIPT_REPLACE_STRING_COMMENT_FILE_PATH = '###STRATEGY_FILE_PATH###'
AGENT_SCRIPT_REPLACE_STRING_PORTFOLIO_STRATEGIES_PATHS = '###PORTFOLIO_STRATEGIES_PATHS###'
AGENT_SCRIPT_REPLACE_STRING_UTILITY_ESTIMATION_TYPE = '###UTILITY_ESTIMATION_TYPE###'

OPPONENT_SCRIPT_REPLACE_STRINGS = [
    REPLACE_STRING_COMMENT,
    REPLACE_STRING_GAME_FILE_PATH,
    OPPONENT_SCRIPT_REPLACE_STRING_COMMENT_FILE_PATH]

AGENT_SCRIPT_REPLACE_STRINGS = [
    REPLACE_STRING_COMMENT,
    REPLACE_STRING_GAME_FILE_PATH,
    AGENT_SCRIPT_REPLACE_STRING_UTILITY_ESTIMATION_TYPE,
    AGENT_SCRIPT_REPLACE_STRING_PORTFOLIO_STRATEGIES_PATHS]

WARNING_COMMENT = 'This file is generated. Do not edit!'


UTILITY_ESTIMATION_METHODS = ['none', 'imaginary_observations', 'aivat']


KUHN_EQUILIBRIUM_STRATEGY_PATH = 'strategies/kuhn.limit.2p-equilibrium.strategy'
LEDUC_EQUILIBRIUM_STRATEGY_PATH = 'strategies/leduc.limit.2p-equilibrium.strategy'


def _replace_in_file(filename, old_strings, new_strings):
    with open(filename) as f:
        s = f.read()

    with open(filename, 'w') as f:
        for i in range(len(old_strings)):
            s = s.replace(old_strings[i], new_strings[i])
        f.write(s)


def _get_agent_name(agent):
    return '%s-%s-%s-exp(%s+-%s)' % (str(agent[0]).split('.')[1], str(agent[1]).split('.')[1], agent[2], agent[3][0], agent[3][1])


def _check_agent_names_unique(agent_specs):
    last_name = None
    for name in sorted(map(lambda a: _get_agent_name(a), agent_specs)):
        if name == last_name:
            return False
        last_name = name
    return True


class BuildPortfolioTest(unittest.TestCase):
    def test_kuhn_simple_build_portfolio(self):
        self.train_and_show_results({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'base_strategy_path': KUHN_EQUILIBRIUM_STRATEGY_PATH,
            'portfolio_name': 'kuhn_simple_portfolio',
            'parallel': True,
            'overwrite_portfolio_path': True,
            'opponent_tilt_types': [
                # (Action.FOLD, TiltType.ADD, 0.5, (100, 5, 10, 2, 2)),
                # (Action.CALL, TiltType.ADD, 0.5, (100, 5, 10, 2, 2)),

                (Action.FOLD, TiltType.ADD, 0.5, (100, 5)),
                (Action.CALL, TiltType.ADD, 0.5, (100, 5)),
                (Action.RAISE, TiltType.ADD, 0.75, (100, 5)),

                (Action.FOLD, TiltType.MULTIPLY, 0.5, (100, 5)),
                (Action.CALL, TiltType.MULTIPLY, 0.5, (100, 5)),
                (Action.RAISE, TiltType.MULTIPLY, 0.75, (100, 5)),

                (Action.FOLD, TiltType.MULTIPLY, 0.8, (100, 5)),
                (Action.CALL, TiltType.MULTIPLY, 0.8, (100, 5)),
                (Action.RAISE, TiltType.MULTIPLY, 0.8, (100, 5)),
            ],
        })

    def test_leduc_test_build_portfolio(self):
        self.train_and_show_results({
            'game_file_path': 'games/leduc.limit.2p.game',
            'base_strategy_path': LEDUC_EQUILIBRIUM_STRATEGY_PATH,
            'portfolio_name': 'leduc_test_portfolio',
            'opponent_tilt_types': [
                (Action.FOLD, TiltType.ADD, 0.5, (500, 500, 10, 10, 2)),
                (Action.CALL, TiltType.ADD, 0.5, (500, 500, 10, 10, 2)),
            ],
        })

    def test_leduc_simple_build_portfolio(self):
        self.train_and_show_results({
            'game_file_path': 'games/leduc.limit.2p.game',
            'base_strategy_path': LEDUC_EQUILIBRIUM_STRATEGY_PATH,
            'portfolio_name': 'leduc_simple_portfolio',
            'opponent_tilt_types': [
                (Action.FOLD, TiltType.ADD, 0.5, (100, 10, 1000, 50)),
                (Action.CALL, TiltType.ADD, 0.5, (100, 10, 1000, 50)),
                (Action.RAISE, TiltType.ADD, 0.75, (100, 10, 1000, 50)),

                (Action.FOLD, TiltType.MULTIPLY, 0.5, (100, 10, 1000, 50)),
                (Action.CALL, TiltType.MULTIPLY, 0.5, (100, 10, 1000, 50)),
                (Action.RAISE, TiltType.MULTIPLY, 0.75, (100, 10, 1000, 50)),

                (Action.FOLD, TiltType.MULTIPLY, 0.8, (100, 10, 1000, 50)),
                (Action.CALL, TiltType.MULTIPLY, 0.8, (100, 10, 1000, 50)),
                (Action.RAISE, TiltType.MULTIPLY, 0.8, (100, 10, 1000, 50)),
            ],
        })

    def test_leduc_final_build_portfolio(self):
        tilt_probabilities = [-1, -0.8, -0.6, -0.4, 0.4, 0.6, 0.8, 1]
        opponents = []
        for action in Action:
            for tilt_type in TiltType:
                for tilt_probability in tilt_probabilities:
                    opponents += [(action, tilt_type, tilt_probability, (100, 10, 1000, 50))]
        self.train_and_show_results({
            'game_file_path': 'games/leduc.limit.2p.game',
            'base_strategy_path': LEDUC_EQUILIBRIUM_STRATEGY_PATH,
            'portfolio_name': 'leduc_final_portfolio',
            'parallel': True,
            'opponent_tilt_types': opponents,
            'overwrite_portfolio_path': True,
        })

    def train_and_show_results(self, test_spec):
        game_file_path = test_spec['game_file_path']
        portfolio_name = test_spec['portfolio_name']
        agent_specs = test_spec['opponent_tilt_types']

        if not _check_agent_names_unique(agent_specs):
            raise AttributeError('Agents must be unique so that they have unique names')


        strategies_directory_base = '%s/%s' % (TEST_OUTPUT_DIRECTORY, portfolio_name)
        strategies_directory = strategies_directory_base
        if 'overwrite_portfolio_path' not in test_spec or not test_spec['overwrite_portfolio_path']:
            counter = 1
            while os.path.exists(strategies_directory):
                strategies_directory = '%s(%s)' % (strategies_directory_base, counter)
                counter += 1
        if not os.path.exists(strategies_directory):
            os.makedirs(strategies_directory)

        game = acpc.read_game_file(game_file_path)
        exp = Exploitability(game)

        # Delete results since they will be generated again
        for file in os.listdir(strategies_directory):
            absolute_path = '/'.join([strategies_directory, file])
            if os.path.isfile(absolute_path):
                os.remove(absolute_path)

        base_strategy, _ = read_strategy_from_file(
            game_file_path,
            test_spec['base_strategy_path'])

        num_opponents = len(agent_specs)
        opponents = []
        for agent in agent_specs:
            opponent_strategy = create_agent_strategy_from_trained_strategy(
                game_file_path,
                base_strategy,
                agent[0],
                agent[1],
                agent[2])
            opponents += [opponent_strategy]

        parallel = test_spec['parallel'] if 'parallel' in test_spec else False

        response_paths = ['%s/responses/%s-response.strategy' % (strategies_directory, _get_agent_name(agent)) for agent in agent_specs]

        opponent_responses = [None] * num_opponents
        responses_to_train_indices = []
        responses_to_train_opponents = []
        responses_to_train_params = []
        for i in range(num_opponents):
            if os.path.exists(response_paths[i]):
                response_strategy, _ = read_strategy_from_file(game_file_path, response_paths[i])
                opponent_responses[i] = response_strategy
            else:
                responses_to_train_indices += [i]
                responses_to_train_opponents += [opponents[i]]
                responses_to_train_params += [agent_specs[i][3]]

        def on_response_trained(response_index, response_strategy):
            output_file_path = response_paths[responses_to_train_indices[response_index]]
            output_file_dir = os.path.dirname(output_file_path)
            if not os.path.exists(output_file_dir):
                os.makedirs(output_file_dir)

            opponent_strategy = opponents[response_index]
            opponent_exploitability = exp.evaluate(opponent_strategy)
            response_exploitability = exp.evaluate(response_strategy)
            response_utility_vs_opponent = exp.evaluate(opponent_strategy, response_strategy)

            write_strategy_to_file(
                response_strategy,
                output_file_path,
                [
                    'Opponent exploitability: %s' % opponent_exploitability,
                    'Response exploitability: %s' % response_exploitability,
                    'Response value vs opponent: %s' % response_utility_vs_opponent,
                ])

        print('%s responses need to be trained' % len(responses_to_train_opponents))

        responses_to_train_strategies = train_portfolio_responses(
            game_file_path,
            responses_to_train_opponents,
            responses_to_train_params,
            log=True,
            parallel=parallel,
            callback=on_response_trained)

        for i, j in enumerate(responses_to_train_indices):
            opponent_responses[j] = responses_to_train_strategies[i]

        portfolio_strategies, response_indices = optimize_portfolio(
            game_file_path,
            opponents,
            opponent_responses,
            log=True,
            output_directory=strategies_directory)

        portfolio_size = len(portfolio_strategies)

        agent_names = [_get_agent_name(agent) for agent in np.take(agent_specs, response_indices, axis=0)]

        print()
        for a in agent_specs:
            print(_get_agent_name(a))

        anaconda_env_name = None
        if 'anaconda3/envs' in sys.executable:
            anaconda_env_name = sys.executable.split('/anaconda3/envs/')[1].split('/')[0]

        response_strategy_file_names = []
        for i, strategy in enumerate(portfolio_strategies):
            agent_name = agent_names[i]

            opponent_strategy = opponents[response_indices[i]]
            opponent_exploitability = exp.evaluate(opponent_strategy)
            response_exploitability = exp.evaluate(strategy)
            response_utility_vs_opponent = exp.evaluate(opponent_strategy, strategy)

            # Save portfolio response strategy
            response_strategy_output_file_path = '%s/%s-response.strategy' % (strategies_directory, agent_name)
            response_strategy_file_names += [response_strategy_output_file_path.split('/')[-1]]
            write_strategy_to_file(
                strategy,
                response_strategy_output_file_path,
                [
                    'Opponent exploitability: %s' % opponent_exploitability,
                    'Response exploitability: %s' % response_exploitability,
                    'Response value vs opponent: %s' % response_utility_vs_opponent,
                ])

            # Save opponent strategy
            opponent_strategy_file_name = '%s-opponent.strategy' % agent_name
            opponent_strategy_output_file_path = '%s/%s' % (strategies_directory, opponent_strategy_file_name)
            write_strategy_to_file(opponent_strategy, opponent_strategy_output_file_path)

            # Generate opponent ACPC script
            opponent_script_path = '%s/%s.sh' % (strategies_directory, agent_name)
            shutil.copy(BASE_OPPONENT_SCRIPT_PATH, opponent_script_path)
            _replace_in_file(
                opponent_script_path,
                OPPONENT_SCRIPT_REPLACE_STRINGS,
                [
                    WARNING_COMMENT,
                    game_file_path,
                    opponent_strategy_output_file_path.split('/')[-1]])
            if anaconda_env_name:
                _replace_in_file(
                    opponent_script_path,
                    [REPLACE_STRING_ENVIRONMENT_ACTIVATION],
                    ['source activate %s' % anaconda_env_name])

        for utility_estimation_method in UTILITY_ESTIMATION_METHODS:
            agent_name_method_name = '' if utility_estimation_method == UTILITY_ESTIMATION_METHODS[0] else '-%s' % utility_estimation_method
            agent_script_path = '%s/%s%s.sh' % (strategies_directory, portfolio_name, agent_name_method_name)
            shutil.copy(BASE_AGENT_SCRIPT_PATH, agent_script_path)

            strategies_replacement = ''
            for i in range(portfolio_size):
                strategies_replacement += '        "${SCRIPT_DIR}/%s"' % response_strategy_file_names[i]
                if i < (portfolio_size - 1):
                    strategies_replacement += ' \\\n'
            _replace_in_file(
                agent_script_path,
                AGENT_SCRIPT_REPLACE_STRINGS,
                [
                    WARNING_COMMENT,
                    game_file_path,
                    '"%s"' % utility_estimation_method,
                    strategies_replacement])
            if anaconda_env_name:
                _replace_in_file(
                    agent_script_path,
                    [REPLACE_STRING_ENVIRONMENT_ACTIVATION],
                    ['source activate %s' % anaconda_env_name])
