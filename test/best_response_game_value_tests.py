import unittest

import acpc_python_client as acpc

from response.best_response import BestResponse
from evaluation.game_value import GameValue
from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import StrategyTreeNodeProvider
from tools.game_tree.nodes import ActionNode
from tools.walk_tree import walk_tree

KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'
LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'


class BestResponseGameValueTests(unittest.TestCase):
    def test_kuhn_always_call_value(self):
        game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)
        strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

        # Create strategy such that it will always check or call
        def on_node(node):
            if isinstance(node, ActionNode):
                node.strategy[1] = 1
        walk_tree(strategy, on_node)

        best_response = BestResponse(game).solve(strategy)
        game_values = GameValue(game).evaluate(strategy, best_response)
        self.assertEqual(game_values, [-1 / 3, 1 / 3])

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
        walk_tree(strategy, on_node)

        best_response = BestResponse(game).solve(strategy)
        game_values = GameValue(game).evaluate(strategy, best_response)
        self.assertEqual(game_values, [-1, 1])

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
        walk_tree(strategy, on_node)

        best_response = BestResponse(game).solve(strategy)
        game_values = GameValue(game).evaluate(strategy, best_response)
        self.assertEqual(game_values, [-1, 1])

    def test_leduc_always_call_not_crashing(self):
        game = acpc.read_game_file(LEDUC_POKER_GAME_FILE_PATH)
        strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

        # Create strategy such that it will always check or call
        def on_node(node):
            if isinstance(node, ActionNode):
                node.strategy[1] = 1
        walk_tree(strategy, on_node)

        best_response = BestResponse(game).solve(strategy)
        game_values = GameValue(game).evaluate(strategy, best_response)
        self.assertLess(game_values[0], 0)

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
        walk_tree(strategy, on_node)

        best_response = BestResponse(game).solve(strategy)
        game_values = GameValue(game).evaluate(strategy, best_response)
        self.assertLess(game_values[0], 0)