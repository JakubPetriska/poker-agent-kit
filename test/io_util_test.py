import unittest

import acpc_python_client as acpc

from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import StrategyTreeNodeProvider
from tools.game_tree.nodes import ActionNode
from tools.walk_trees import walk_trees
from tools.io_util import write_strategy_to_file, read_strategy_from_file
from tools.game_utils import is_strategies_equal


KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'


class IoUtilTest(unittest.TestCase):
    def test_strategy_writing_and_reading(self):
        game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)
        strategy_tree = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

        def on_node(node):
            if isinstance(node, ActionNode):
                for a in range(3):
                    if a in node.children:
                        node.strategy[a] = 0.5
                    else:
                        node.strategy[a] = 7
        walk_trees(on_node, strategy_tree)

        write_strategy_to_file(strategy_tree, 'test/io_test_dummy.strategy')
        read_strategy_tree = read_strategy_from_file(KUHN_POKER_GAME_FILE_PATH, 'test/io_test_dummy.strategy')
        self.assertTrue(is_strategies_equal(strategy_tree, read_strategy_tree))
