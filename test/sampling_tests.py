import unittest
import os
import numpy as np

from tools.sampling import read_log_file
from tools.walk_tree import walk_tree_with_data
from tools.game_tree.nodes import ActionNode, BoardCardsNode, HoleCardsNode

LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'


class SamplingTests(unittest.TestCase):
    def test_log_parsing_to_sample_trees(self):
        players = read_log_file(
            LEDUC_POKER_GAME_FILE_PATH,
            'test/sample_log.log',
            ['player_1', 'player_2'])

        callback_was_called_at_least_once = False

        def node_callback(node, data):
            nonlocal callback_was_called_at_least_once
            if isinstance(node, ActionNode):
                callback_was_called_at_least_once = True

                if data:
                    self.assertTrue(np.all(node.action_decision_counts == [0, 1, 0]))
                else:
                    self.assertTrue(np.all(node.action_decision_counts == [0, 0, 0]))

                return [data if action == 1 else False for action in node.children]
            elif isinstance(node, HoleCardsNode):
                return [cards == (43,) or cards == (47,) for cards in node.children]
            elif isinstance(node, BoardCardsNode):
                return [data if cards == (50,) else False for cards in node.children]
            else:
                return [data for _ in node.children]


        for name in players:
            player_tree = players[name]
            walk_tree_with_data(player_tree, True, node_callback)
        self.assertTrue(callback_was_called_at_least_once)

    def test_log_parsing_to_sample_trees_performance(self):
        players = read_log_file(
            LEDUC_POKER_GAME_FILE_PATH,
            'test/sample_log-large.log',
            ['CFR_trained', 'Random_1'])
        visits_sum = 0
        for name in players:
            player_tree = players[name]
            for _, root_action_node in player_tree.children.items():
                 visits_sum += np.sum(root_action_node.action_decision_counts)
        self.assertEqual(visits_sum, 50000)
