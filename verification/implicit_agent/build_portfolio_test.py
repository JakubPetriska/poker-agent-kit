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


TEST_DIRECTORY = 'verification/implicit_agent'
TEST_OUTPUT_DIRECTORY = '%s/portfolios' % TEST_DIRECTORY

BASE_AGENT_SCRIPT_PATH = '%s/base_agent_script.sh' % TEST_DIRECTORY
BASE_OPPONENT_SCRIPT_PATH = '%s/base_opponent_script.sh' % TEST_DIRECTORY

REPLACE_STRING_COMMENT = '###COMMENT###'
REPLACE_STRING_GAME_FILE_PATH = '###GAME_FILE_PATH###'
REPLACE_STRING_ENVIRONMENT_ACTIVATION = '###ENVIRONMENT_ACTIVATION###'
OPPONENT_SCRIPT_REPLACE_STRING_COMMENT_FILE_PATH = '###STRATEGY_FILE_PATH###'
AGENT_SCRIPT_PORTFOLIO_STRATEGIES_PATHS = "###PORTFOLIO_STRATEGIES_PATHS###"

OPPONENT_SCRIPT_REPLACE_STRINGS = [
    REPLACE_STRING_COMMENT,
    REPLACE_STRING_GAME_FILE_PATH,
    OPPONENT_SCRIPT_REPLACE_STRING_COMMENT_FILE_PATH]

AGENT_SCRIPT_REPLACE_STRINGS = [
    REPLACE_STRING_COMMENT,
    REPLACE_STRING_GAME_FILE_PATH,
    AGENT_SCRIPT_PORTFOLIO_STRATEGIES_PATHS]

WARNING_COMMENT = 'This file is generated. Do not edit!'


KUHN_EQUILIBRIUM_STRATEGY_PATH = 'strategies/kuhn.limit.2p-equilibrium.strategy'
LEDUC_EQUILIBRIUM_STRATEGY_PATH = 'strategies/leduc.limit.2p-equilibrium.strategy'


def replace_in_file(filename, old_strings, new_strings):
    with open(filename) as f:
        s = f.read()

    with open(filename, 'w') as f:
        for i in range(len(old_strings)):
            s = s.replace(old_strings[i], new_strings[i])
        f.write(s)


class BuildPortfolioTest(unittest.TestCase):
    def test_kuhn_build_portfolio(self):
        self.train_and_show_results({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'base_strategy_path': KUHN_EQUILIBRIUM_STRATEGY_PATH,
            'portfolio_name': 'kuhn_simple_portfolio',
            'opponent_tilt_types': [
                ('FOLD-ADD-0.5-p=0.2', Action.FOLD, TiltType.ADD, 0.5, (100, 300, 10, 2, 2)),
                ('CALL-ADD-0.5-p=0.2', Action.CALL, TiltType.ADD, 0.5, (100, 300, 10, 2, 2)),

                # ('FOLD-ADD-0.5-p=0.2', Action.FOLD, TiltType.ADD, 0.5, (100, 5)),
                # ('CALL-ADD-0.5-p=0.2', Action.CALL, TiltType.ADD, 0.5, (100, 5)),
                # ('RAISE-ADD-0.75-p=0.2', Action.RAISE, TiltType.ADD, 0.75, (100, 5)),

                # ('FOLD-ADD-0.5-p=0.2', Action.FOLD, TiltType.MULTIPLY, 0.5, (100, 5)),
                # ('CALL-ADD-0.5-p=0.2', Action.CALL, TiltType.MULTIPLY, 0.5, (100, 5)),
                # ('RAISE-ADD-0.75-p=0.2', Action.RAISE, TiltType.MULTIPLY, 0.75, (100, 5)),

                # ('FOLD-MULTIPLY-0.8-p=0.2', Action.FOLD, TiltType.MULTIPLY, 0.8, (100, 5)),
                # ('CALL-MULTIPLY-0.8-p=0.2', Action.CALL, TiltType.MULTIPLY, 0.8, (100, 5)),
                # ('RAISE-MULTIPLY-0.8-p=0.2', Action.RAISE, TiltType.MULTIPLY, 0.8, (100, 5)),
            ],
        })

    def train_and_show_results(self, test_spec):
        game_file_path = test_spec['game_file_path']
        portfolio_name = test_spec['portfolio_name']
        strategies_directory = '%s/%s' % (TEST_OUTPUT_DIRECTORY, portfolio_name)

        if os.path.exists(strategies_directory):
            shutil.rmtree(strategies_directory)
        os.makedirs(strategies_directory)

        base_strategy, _ = read_strategy_from_file(
            game_file_path,
            test_spec['base_strategy_path'])

        agent_specs = test_spec['opponent_tilt_types']

        opponents = []
        for agent in agent_specs:
            opponent_strategy = create_agent_strategy_from_trained_strategy(
                game_file_path,
                base_strategy,
                agent[1],
                agent[2],
                agent[3])
            opponents += [opponent_strategy]
        portfolio_strategies, response_indices = build_portfolio(
            game_file_path,
            opponents,
            [agent[4] for agent in agent_specs],
            log=True,
            output_directory=strategies_directory)

        portfolio_size = len(portfolio_strategies)

        agent_names = [agent_spec[0] for agent_spec in np.take(agent_specs, response_indices, axis=0)]

        anaconda_env_name = None
        if 'anaconda3/envs' in sys.executable:
            anaconda_env_name = sys.executable.split('/anaconda3/envs/')[1].split('/')[0]

        response_strategy_paths = []
        for i, strategy in enumerate(portfolio_strategies):
            agent_name = agent_names[i]

            # Save portfolio response strategy
            response_strategy_output_file_path = '%s/%s-response.strategy' % (strategies_directory, agent_name)
            counter = 0
            while os.path.exists(response_strategy_output_file_path):
                counter += 1
                response_strategy_output_file_path = '%s/%s-%s-response.strategy' % (strategies_directory, agent_name, counter)
            response_strategy_paths += [response_strategy_output_file_path]
            write_strategy_to_file(strategy, response_strategy_output_file_path)

            # Save opponent strategy
            opponent_strategy_file_name = '%s-opponent.strategy' % agent_name
            opponent_strategy_output_file_path = '%s/%s' % (strategies_directory, opponent_strategy_file_name)
            counter = 0
            while os.path.exists(opponent_strategy_output_file_path):
                counter += 1
                opponent_strategy_output_file_path = '%s/%s-%s-opponent.strategy' % (strategies_directory, agent_name, counter)
            write_strategy_to_file(opponents[response_indices[i]], opponent_strategy_output_file_path)

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

        agent_script_path = '%s/%s.sh' % (strategies_directory, portfolio_name)
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
                strategies_replacement])
        if anaconda_env_name:
            replace_in_file(
                agent_script_path,
                [REPLACE_STRING_ENVIRONMENT_ACTIVATION],
                ['source activate %s' % anaconda_env_name])
