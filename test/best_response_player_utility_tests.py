import unittest

import acpc_python_client as acpc

from response.best_response import BestResponse
from evaluation.player_utility import PlayerUtility
from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import StrategyTreeNodeProvider
from tools.game_tree.nodes import ActionNode
from tools.walk_trees import walk_trees

KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'
KUHN_BIGDECK_2ROUND_POKER_GAME_FILE_PATH = 'games/kuhn.bigdeck.2round.limit.2p.game'
LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'


class BestResponsePlayerUtilityTests(unittest.TestCase):
    def test_kuhn_always_call_value(self):
        game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)
        strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

        # Create strategy such that it will always check or call
        def on_node(node):
            if isinstance(node, ActionNode):
                node.strategy[1] = 1
        walk_trees(on_node, strategy)

        best_response = BestResponse(game).solve(strategy)
        player_utilities, player_positions = PlayerUtility(game).evaluate(strategy, best_response)
        self.assertEqual(player_utilities.tolist(), [[-1 / 3, 1 / 3], [1 / 3, -1 / 3]])
        self.assertEqual(player_positions.tolist(), [[0, 1], [1, 0]])

    def test_kuhn_always_fold_value(self):
        game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)
        strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

        # Create strategy such that it will always check or fold
        def on_node(node):
            if isinstance(node, ActionNode):
                if 0 in node.children:
                    node.strategy[0] = 1
                else:
                    node.strategy[1] = 1
        walk_trees(on_node, strategy)

        best_response = BestResponse(game).solve(strategy)
        player_utilities, player_positions = PlayerUtility(game).evaluate(strategy, best_response)
        self.assertEqual(player_utilities.tolist(), [[-1, 1], [1, -1]])
        self.assertEqual(player_positions.tolist(), [[0, 1], [1, 0]])

    def test_kuhn_bigdeck_2round_always_fold_value(self):
        game = acpc.read_game_file(KUHN_BIGDECK_2ROUND_POKER_GAME_FILE_PATH)
        strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

        # Create strategy such that it will always check or fold
        def on_node(node):
            if isinstance(node, ActionNode):
                if 0 in node.children:
                    node.strategy[0] = 1
                else:
                    node.strategy[1] = 1
        walk_trees(on_node, strategy)

        best_response = BestResponse(game).solve(strategy)
        player_utilities, player_positions = PlayerUtility(game).evaluate(strategy, best_response)
        self.assertEqual(player_utilities.tolist(), [[-1, 1], [1, -1]])
        self.assertEqual(player_positions.tolist(), [[0, 1], [1, 0]])

    def test_leduc_always_fold_value(self):
        game = acpc.read_game_file(LEDUC_POKER_GAME_FILE_PATH)
        strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

        # Create strategy such that it will always check or fold
        def on_node(node):
            if isinstance(node, ActionNode):
                if 0 in node.children:
                    node.strategy[0] = 1
                else:
                    node.strategy[1] = 1
        walk_trees(on_node, strategy)

        best_response = BestResponse(game).solve(strategy)
        player_utilities, player_positions = PlayerUtility(game).evaluate(strategy, best_response)
        self.assertEqual(player_utilities.tolist(), [[-1, 1], [1, -1]])
        self.assertEqual(player_positions.tolist(), [[0, 1], [1, 0]])

    def test_leduc_always_call_not_crashing(self):
        game = acpc.read_game_file(LEDUC_POKER_GAME_FILE_PATH)
        strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

        # Create strategy such that it will always check or call
        def on_node(node):
            if isinstance(node, ActionNode):
                node.strategy[1] = 1
        walk_trees(on_node, strategy)

        best_response = BestResponse(game).solve(strategy)
        player_utilities, player_positions = PlayerUtility(game).evaluate(strategy, best_response)
        for i in range(player_utilities.shape[0]):
            for j in range(player_utilities.shape[1]):
                if i == j:
                    self.assertLess(player_utilities[i, j], 0)
                else:
                    self.assertGreater(player_utilities[i, j], 0)
        self.assertEqual(player_positions.tolist(), [[0, 1], [1, 0]])

    def test_leduc_uniform_not_crashing(self):
        game = acpc.read_game_file(LEDUC_POKER_GAME_FILE_PATH)
        strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

        # Create strategy such that it will have uniform probabilities over all actions
        def on_node(node):
            if isinstance(node, ActionNode):
                action_count = len(node.children)
                action_probability = 1 / action_count
                for a in node.children:
                    node.strategy[a] = action_probability
        walk_trees(on_node, strategy)

        best_response = BestResponse(game).solve(strategy)
        player_utilities, player_positions = PlayerUtility(game).evaluate(strategy, best_response)
        for i in range(player_utilities.shape[0]):
            for j in range(player_utilities.shape[1]):
                if i == j:
                    self.assertLess(player_utilities[i, j], 0)
                else:
                    self.assertGreater(player_utilities[i, j], 0)
        self.assertEqual(player_positions.tolist(), [[0, 1], [1, 0]])
