import unittest

import acpc_python_client as acpc

from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import StrategyTreeNodeProvider
from tools.game_tree.nodes import ActionNode
from tools.walk_trees import walk_trees
from implicit_modelling.build_portfolio import train_portfolio_responses, optimize_portfolio


KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'


class ImplicitAgentTests(unittest.TestCase):
    def create_strategy(self, game, node_strategy_creator_callback):
        strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()
        walk_trees(node_strategy_creator_callback, strategy)
        return strategy

    def test_build_portfolio_not_crashing(self):
        game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)

        def on_node_always_call(node):
            if isinstance(node, ActionNode):
                node.strategy[1] = 1

        def on_node_always_fold(node):
            if isinstance(node, ActionNode):
                if 0 in node.children:
                    node.strategy[0] = 1
                else:
                    node.strategy[1] = 1

        def on_node_uniform(node):
            if isinstance(node, ActionNode):
                action_count = len(node.children)
                action_probability = 1 / action_count
                for a in node.children:
                    node.strategy[a] = action_probability

        opponents = [
            self.create_strategy(game, on_node_always_call),
            self.create_strategy(game, on_node_always_fold),
            self.create_strategy(game, on_node_uniform)]

        opponent_responses = train_portfolio_responses(
            KUHN_POKER_GAME_FILE_PATH,
            opponents,
            [(100, 800, 10, 2, 2)] * len(opponents))
        portfolio_strategies, opponent_indices = optimize_portfolio(
            KUHN_POKER_GAME_FILE_PATH,
            opponents,
            opponent_responses)
        self.assertGreaterEqual(len(portfolio_strategies), 1)
        self.assertEqual(len(portfolio_strategies), len(opponent_indices))
