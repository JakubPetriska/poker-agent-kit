import unittest
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import acpc_python_client as acpc

from response.restricted_nash_response import RestrictedNashResponse
from cfr.main import Cfr
from tools.constants import Action
from weak_agents.action_tilted_agent import create_agent_strategy_from_trained_strategy, TiltType
from tools.io_util import read_strategy_from_file
from response.rnr_parameter_optimizer import RnrParameterOptimizer
from tools.game_utils import is_correct_strategy


KUHN_EQUILIBRIUM_STRATEGY_PATH = 'strategies/kuhn.limit.2p-equilibrium.strategy'


class RnrParameterOptimizerTest(unittest.TestCase):
    def test_kuhn_rnr_parameter_optimizer_1(self):
        self.train_and_show_results({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'base_strategy_path': KUHN_EQUILIBRIUM_STRATEGY_PATH,
            'opponent': ('FOLD-ADD-0.5-p=.8', Action.FOLD, TiltType.ADD, 0.5),
            'exploitability': 123,
            'max_delta': 1
        })

    def train_and_show_results(self, test_spec):
        game_file_path = test_spec['game_file_path']
        game = acpc.read_game_file(game_file_path)

        base_strategy, _ = read_strategy_from_file(
            game_file_path,
            test_spec['base_strategy_path'])

        opponent = test_spec['opponent']
        opponent_strategy = create_agent_strategy_from_trained_strategy(
            game_file_path,
            base_strategy,
            opponent[1],
            opponent[2],
            opponent[3])

        strategy, exploitability, p = RnrParameterOptimizer(game).train(
            opponent_strategy,
            test_spec['exploitability'],
            test_spec['max_delta'])

        self.assertTrue(strategy != None)
        self.assertTrue(is_correct_strategy(strategy))
        print('Final exploitability is %s with p of %s' % (exploitability, p))
