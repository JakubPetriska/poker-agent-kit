import unittest
import random

import acpc_python_client as acpc

from tools.game_tree.builder import GameTreeBuilder
from tools.sampling import SamplesTreeNodeProvider
from tools.game_tree.nodes import ActionNode
from tools.walk_tree import walk_tree
from response.data_biased_response import DataBiasedResponse


KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'
LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'


class DataBiasedResponseTests(unittest.TestCase):
    def test_kuhn_data_biased_response_works(self):
        game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)
        samples_game_tree = GameTreeBuilder(
            game, SamplesTreeNodeProvider()).build_tree()

        # Create random strategy
        def on_node(node):
            if isinstance(node, ActionNode):
                for a in node.children:
                    node.action_decision_counts[a] = random.randrange(15)
        walk_tree(samples_game_tree, on_node)

        dbr = DataBiasedResponse(game, samples_game_tree, show_progress=False)
        dbr.train(10)

    def test_leduc_data_biased_response_works(self):
        game = acpc.read_game_file(LEDUC_POKER_GAME_FILE_PATH)
        samples_game_tree = GameTreeBuilder(
            game, SamplesTreeNodeProvider()).build_tree()

        # Create random strategy
        def on_node(node):
            if isinstance(node, ActionNode):
                for a in node.children:
                    node.action_decision_counts[a] = random.randrange(15)
        walk_tree(samples_game_tree, on_node)

        dbr = DataBiasedResponse(game, samples_game_tree, show_progress=False)
        dbr.train(5)
