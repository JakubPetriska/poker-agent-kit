import unittest
import os

from tools.sampling import read_log_file
from tools.walk_tree import walk_tree_with_data
from tools.game_tree.nodes import ActionNode

LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'
SAMPLE_LOG_FILE = 'test/sample_log.log'


class SamplingTests(unittest.TestCase):
    def test_log_parsing_to_sample_trees(self):
        players = read_log_file(
            LEDUC_POKER_GAME_FILE_PATH,
            SAMPLE_LOG_FILE,
            ['player_1', 'player_2'])

        callback_was_called_at_least_once = False

        def node_callback(node, data):
            nonlocal callback_was_called_at_least_once
            if isinstance(node, ActionNode):
                callback_was_called_at_least_once = True

                if data:
                    self.assertEqual(node.action_decision_counts, [0, 1, 0])
                else:
                    self.assertEqual(node.action_decision_counts, [0, 0, 0])

                return [data if action == 1 else data for action in node.children]
            else:
                return [data for _ in node.children]


        for name in players:
            player_tree = players[name]
            walk_tree_with_data(player_tree, True, node_callback)
        self.assertTrue(callback_was_called_at_least_once)
