import unittest
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import acpc_python_client as acpc

from tools.constants import Action
from weak_agents.action_tilted_agent import create_agent_strategy_from_trained_strategy, TiltType
from tools.io_util import read_strategy_from_file
from implicit_modelling.build_portfolio import build_portfolio


FIGURES_FOLDER = 'verification/build_portfolio'

KUHN_EQUILIBRIUM_STRATEGY_PATH = 'strategies/kuhn.limit.2p-equilibrium.strategy'
LEDUC_EQUILIBRIUM_STRATEGY_PATH = 'strategies/leduc.limit.2p-equilibrium.strategy'


class BuildPortfolioTest(unittest.TestCase):
    def test_kuhn_build_portfolio(self):
        self.train_and_show_results({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'base_strategy_path': KUHN_EQUILIBRIUM_STRATEGY_PATH,
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
        build_portfolio(
            game_file_path,
            opponents,
            [(agent[3], test_spec['rnr_iterations']) for agent in agent_specs],
            log=True)
