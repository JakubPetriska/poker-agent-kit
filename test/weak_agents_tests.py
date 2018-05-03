import unittest

import acpc_python_client as acpc

from tools.constants import Action
from weak_agents.action_tilted_agent import create_agent_strategy, create_agent_strategy_from_trained_strategy, TiltType
from tools.io_util import read_strategy_from_file
from evaluation.exploitability import Exploitability
from tools.game_utils import is_strategies_equal, is_correct_strategy


KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'
LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'


class WeakAgentsTests(unittest.TestCase):
    def test_kuhn_action_tilted_agent_not_crashing(self):
        strategy = create_agent_strategy(
            KUHN_POKER_GAME_FILE_PATH,
            Action.RAISE,
            TiltType.ADD,
            0.2,
            cfr_iterations=20,
            cfr_weight_delay=2,
            show_progress=False)
        self.assertTrue(is_correct_strategy(strategy))

    def test_leduc_add_action_tilted_agent_not_crashing(self):
        strategy = create_agent_strategy(
            LEDUC_POKER_GAME_FILE_PATH,
            Action.FOLD,
            TiltType.ADD,
            0.1,
            cfr_iterations=5,
            cfr_weight_delay=2,
            show_progress=False)
        self.assertTrue(is_correct_strategy(strategy))

    def test_leduc_multiply_action_tilted_agent_not_crashing(self):
        strategy = create_agent_strategy(
            LEDUC_POKER_GAME_FILE_PATH,
            Action.FOLD,
            TiltType.MULTIPLY,
            0.1,
            cfr_iterations=5,
            cfr_weight_delay=2,
            show_progress=False)
        self.assertTrue(is_correct_strategy(strategy))

    def test_kuhn_action_tilted_agent(self):
        kuhn_equilibrium, _ = read_strategy_from_file(
            KUHN_POKER_GAME_FILE_PATH,
            'strategies/kuhn.limit.2p-equilibrium.strategy')

        game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)
        exploitability = Exploitability(game)

        raise_add_tilted = create_agent_strategy_from_trained_strategy(
            KUHN_POKER_GAME_FILE_PATH,
            kuhn_equilibrium,
            Action.RAISE,
            TiltType.ADD,
            0.2)
        self.assertTrue(is_correct_strategy(raise_add_tilted))
        self.assertTrue(not is_strategies_equal(kuhn_equilibrium, raise_add_tilted))

        equilibrium_exploitability = exploitability.evaluate(kuhn_equilibrium)
        raise_add_tilted_exploitability = exploitability.evaluate(raise_add_tilted)
        self.assertTrue(raise_add_tilted_exploitability > equilibrium_exploitability)
