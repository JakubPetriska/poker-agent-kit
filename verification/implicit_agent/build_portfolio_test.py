import unittest
import os
import shutil
import math
import numpy as np
import matplotlib.pyplot as plt

import acpc_python_client as acpc

from tools.constants import Action
from weak_agents.action_tilted_agent import create_agent_strategy_from_trained_strategy, TiltType
from tools.io_util import read_strategy_from_file
from implicit_modelling.build_portfolio import build_portfolio
from tools.io_util import write_strategy_to_file


TEST_OUTPUT_DIRECTORY = 'verification/implicit_agent/portfolios'

KUHN_EQUILIBRIUM_STRATEGY_PATH = 'strategies/kuhn.limit.2p-equilibrium.strategy'
LEDUC_EQUILIBRIUM_STRATEGY_PATH = 'strategies/leduc.limit.2p-equilibrium.strategy'


class BuildPortfolioTest(unittest.TestCase):
    def test_kuhn_build_portfolio(self):
        self.train_and_show_results({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'base_strategy_path': KUHN_EQUILIBRIUM_STRATEGY_PATH,
            'strategy_output_folder': '%s/kuhn_simple_portfolio' % TEST_OUTPUT_DIRECTORY,
            'rnr_iterations': 1500,
            'opponent_tilt_types': [
                ('FOLD-ADD-0.5-p=0.2', Action.FOLD, TiltType.ADD, 0.5, 0.2),
                ('CALL-ADD-0.5-p=0.2', Action.CALL, TiltType.ADD, 0.5, 0.2),
                ('RAISE-ADD-0.75-p=0.2', Action.RAISE, TiltType.ADD, 0.75, 0.2),

                ('FOLD-ADD-0.5-p=0.2', Action.FOLD, TiltType.MULTIPLY, 0.5, 0.2),
                ('CALL-ADD-0.5-p=0.2', Action.CALL, TiltType.MULTIPLY, 0.5, 0.2),
                ('RAISE-ADD-0.75-p=0.2', Action.RAISE, TiltType.MULTIPLY, 0.75, 0.2),

                ('FOLD-MULTIPLY-0.8-p=0.2', Action.FOLD, TiltType.MULTIPLY, 0.8, 0.2),
                ('CALL-MULTIPLY-0.8-p=0.2', Action.CALL, TiltType.MULTIPLY, 0.8, 0.05),
                ('RAISE-MULTIPLY-0.8-p=0.2', Action.RAISE, TiltType.MULTIPLY, 0.8, 0.05),
            ],
        })

    def train_and_show_results(self, test_spec):
        game_file_path = test_spec['game_file_path']
        strategies_directory = test_spec['strategy_output_folder']

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
            [(agent[3], test_spec['rnr_iterations']) for agent in agent_specs],
            log=True,
            output_directory=strategies_directory)

        agent_names = [agent_spec[0] for agent_spec in np.take(agent_specs, response_indices, axis=0)]

        for i, strategy in enumerate(portfolio_strategies):
            agent_name = agent_names[i]

            # Save portfolio response strategy
            response_strategy_output_file_path = '%s/%s-response.strategy' % (strategies_directory, agent_name)
            counter = 0
            while os.path.exists(response_strategy_output_file_path):
                counter += 1
                response_strategy_output_file_path = '%s/%s-%s-response.strategy' % (strategies_directory, agent_name, counter)
            write_strategy_to_file(strategy, response_strategy_output_file_path)

            # Save opponent strategy
            opponent_strategy_output_file_path = '%s/%s-opponent.strategy' % (strategies_directory, agent_name)
            counter = 0
            while os.path.exists(opponent_strategy_output_file_path):
                counter += 1
                opponent_strategy_output_file_path = '%s/%s-%s-opponent.strategy' % (strategies_directory, agent_name, counter)
            write_strategy_to_file(opponents[response_indices[i]], opponent_strategy_output_file_path)
