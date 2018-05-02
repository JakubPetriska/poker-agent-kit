import unittest

import acpc_python_client as acpc

from response.restricted_nash_response import RestrictedNashResponse
from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import StrategyTreeNodeProvider
from tools.game_tree.nodes import ActionNode
from tools.walk_trees import walk_trees

KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'
KUHN_BIG_DECK_POKER_GAME_FILE_PATH = 'games/kuhn.bigdeck.limit.2p.game'
KUHN_BIG_DECK_2ROUND_POKER_GAME_FILE_PATH = 'games/kuhn.bigdeck.2round.limit.2p.game'
LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'


class RnrTests(unittest.TestCase):
    def test_kuhn_rnr_works(self):
        game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)

        opponent_strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()
        def on_node(node):
            if isinstance(node, ActionNode):
                action_count = len(node.children)
                action_probability = 1 / action_count
                for a in node.children:
                    node.strategy[a] = action_probability
        walk_trees(on_node, opponent_strategy)

        rnr = RestrictedNashResponse(
            game, opponent_strategy, 0.5, show_progress=False)
        rnr.train(5)

    def test_kuhn_bigdeck_rnr_works(self):
        game = acpc.read_game_file(KUHN_BIG_DECK_POKER_GAME_FILE_PATH)

        opponent_strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()
        def on_node(node):
            if isinstance(node, ActionNode):
                action_count = len(node.children)
                action_probability = 1 / action_count
                for a in node.children:
                    node.strategy[a] = action_probability
        walk_trees(on_node, opponent_strategy)

        rnr = RestrictedNashResponse(
            game, opponent_strategy, 0.5, show_progress=False)
        rnr.train(5)

    def test_kuhn_bigdeck_2round_rnr_works(self):
        game = acpc.read_game_file(KUHN_BIG_DECK_2ROUND_POKER_GAME_FILE_PATH)

        opponent_strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()
        def on_node(node):
            if isinstance(node, ActionNode):
                action_count = len(node.children)
                action_probability = 1 / action_count
                for a in node.children:
                    node.strategy[a] = action_probability
        walk_trees(on_node, opponent_strategy)

        rnr = RestrictedNashResponse(
            game, opponent_strategy, 0.5, show_progress=False)
        rnr.train(5)

    def test_leduc_rnr_works(self):
        game = acpc.read_game_file(LEDUC_POKER_GAME_FILE_PATH)

        opponent_strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()
        def on_node(node):
            if isinstance(node, ActionNode):
                action_count = len(node.children)
                action_probability = 1 / action_count
                for a in node.children:
                    node.strategy[a] = action_probability
        walk_trees(on_node, opponent_strategy)

        rnr = RestrictedNashResponse(
            game, opponent_strategy, 0.5, show_progress=False)
        rnr.train(5)
