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
from tools.io_util import read_strategy_from_file
from implicit_modelling.build_portfolio import build_portfolio
from tools.io_util import write_strategy_to_file
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


UTILITY_ESTIMATION_METHODS = ['none', 'imaginary_observations']


KUHN_EQUILIBRIUM_STRATEGY_PATH = 'strategies/kuhn.limit.2p-equilibrium.strategy'
LEDUC_EQUILIBRIUM_STRATEGY_PATH = 'strategies/leduc.limit.2p-equilibrium.strategy'


def replace_in_file(filename, old_strings, new_strings):
    with open(filename) as f:
        s = f.read()

    with open(filename, 'w') as f:
        for i in range(len(old_strings)):
            s = s.replace(old_strings[i], new_strings[i])
        f.write(s)


def get_agent_name(agent):
        return '%s-%s-%s' % (str(agent[0]).split('.')[1], str(agent[1]).split('.')[1], agent[2])


class BuildPortfolioTest(unittest.TestCase):
    def test_kuhn_simple_build_portfolio(self):
        self.train_and_show_results({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'base_strategy_path': KUHN_EQUILIBRIUM_STRATEGY_PATH,
            'portfolio_name': 'kuhn_simple_portfolio',
            'parallel': True,
            'opponent_tilt_types': [
                # (Action.FOLD, TiltType.ADD, 0.5, (100, 300, 10, 2, 2)),
                # (Action.CALL, TiltType.ADD, 0.5, (100, 300, 10, 2, 2)),

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
        tilt_probabilities = [0.4, 0.6, 0.8, 1]
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
        })

    def train_and_show_results(self, test_spec):
        game_file_path = test_spec['game_file_path']
        portfolio_name = test_spec['portfolio_name']

        strategies_directory_base = '%s/%s' % (TEST_OUTPUT_DIRECTORY, portfolio_name)
        strategies_directory = strategies_directory_base
        counter = 1
        while os.path.exists(strategies_directory):
            strategies_directory = '%s(%s)' % (strategies_directory_base, counter)
            counter += 1
        os.makedirs(strategies_directory)

        game = acpc.read_game_file(game_file_path)
        exp = Exploitability(game)

        base_strategy, _ = read_strategy_from_file(
            game_file_path,
            test_spec['base_strategy_path'])

        agent_specs = test_spec['opponent_tilt_types']

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
        portfolio_strategies, response_indices = build_portfolio(
            game_file_path,
            opponents,
            [agent[3] for agent in agent_specs],
            log=True,
            output_directory=strategies_directory,
            parallel=parallel)

        portfolio_size = len(portfolio_strategies)

        agent_names = [get_agent_name(agent) for agent in np.take(agent_specs, response_indices, axis=0)]

        anaconda_env_name = None
        if 'anaconda3/envs' in sys.executable:
            anaconda_env_name = sys.executable.split('/anaconda3/envs/')[1].split('/')[0]

        response_strategy_paths = []
        for i, strategy in enumerate(portfolio_strategies):
            agent_name = agent_names[i]

            opponent_strategy = opponents[response_indices[i]]
            opponent_exploitability = exp.evaluate(opponent_strategy)
            response_exploitability = exp.evaluate(strategy)
            response_utility_vs_opponent = exp.evaluate(opponent_strategy, strategy)

            # Save portfolio response strategy
            response_strategy_output_file_path = '%s/%s-response.strategy' % (strategies_directory, agent_name)
            counter = 0
            while os.path.exists(response_strategy_output_file_path):
                counter += 1
                response_strategy_output_file_path = '%s/%s-%s-response.strategy' % (strategies_directory, agent_name, counter)
            response_strategy_paths += [response_strategy_output_file_path]
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
            counter = 0
            while os.path.exists(opponent_strategy_output_file_path):
                counter += 1
                opponent_strategy_output_file_path = '%s/%s-%s-opponent.strategy' % (strategies_directory, agent_name, counter)
            write_strategy_to_file(opponent_strategy, opponent_strategy_output_file_path)

            # Generate opponent ACPC script
            opponent_script_path = '%s/%s.sh' % (strategies_directory, agent_name)
            shutil.copy(BASE_OPPONENT_SCRIPT_PATH, opponent_script_path)
            replace_in_file(
                opponent_script_path,
                OPPONENT_SCRIPT_REPLACE_STRINGS,
                [
                    WARNING_COMMENT,
                    game_file_path,
                    opponent_strategy_output_file_path])
            if anaconda_env_name:
                replace_in_file(
                    opponent_script_path,
                    [REPLACE_STRING_ENVIRONMENT_ACTIVATION],
                    ['source activate %s' % anaconda_env_name])

        for utility_estimation_method in UTILITY_ESTIMATION_METHODS:
            agent_name_method_name = '' if utility_estimation_method == UTILITY_ESTIMATION_METHODS[0] else '-%s' % utility_estimation_method
            agent_script_path = '%s/%s%s.sh' % (strategies_directory, portfolio_name, agent_name_method_name)
            shutil.copy(BASE_AGENT_SCRIPT_PATH, agent_script_path)

            strategies_replacement = ''
            for i in range(portfolio_size):
                strategies_replacement += '        "${WORKSPACE_DIR}/%s"' % response_strategy_paths[i]
                if i < (portfolio_size - 1):
                    strategies_replacement += ' \\\n'
            replace_in_file(
                agent_script_path,
                AGENT_SCRIPT_REPLACE_STRINGS,
                [
                    WARNING_COMMENT,
                    game_file_path,
                    '"%s"' % utility_estimation_method,
                    strategies_replacement])
            if anaconda_env_name:
                replace_in_file(
                    agent_script_path,
                    [REPLACE_STRING_ENVIRONMENT_ACTIVATION],
                    ['source activate %s' % anaconda_env_name])
