import unittest
import random

import acpc_python_client as acpc

from tools.game_tree.builder import GameTreeBuilder
from tools.sampling import SamplesTreeNodeProvider
from tools.game_tree.nodes import ActionNode
from tools.walk_trees import walk_trees
from response.data_biased_response import DataBiasedResponse


KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'
LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'


class DataBiasedResponseTests(unittest.TestCase):
    pass
    # TODO make this work
    # def test_kuhn_data_biased_response_works(self):
    #     game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)
    #     samples_game_tree = GameTreeBuilder(
    #         game, SamplesTreeNodeProvider()).build_tree()

    #     # Create random strategy
    #     def on_node(node):
    #         if isinstance(node, ActionNode):
    #             for a in node.children:
    #                 node.action_decision_counts[a] = random.randrange(15)
    #     walk_trees(on_node, samples_game_tree)

    #     dbr = DataBiasedResponse(game, samples_game_tree, show_progress=False)
    #     dbr.train(10, 5)

    # def test_leduc_data_biased_response_works(self):
    #     game = acpc.read_game_file(LEDUC_POKER_GAME_FILE_PATH)
    #     samples_game_tree = GameTreeBuilder(
    #         game, SamplesTreeNodeProvider()).build_tree()

    #     # Create random strategy
    #     def on_node(node):
    #         if isinstance(node, ActionNode):
    #             for a in node.children:
    #                 node.action_decision_counts[a] = random.randrange(15)
    #     walk_trees(on_node, samples_game_tree)

    #     dbr = DataBiasedResponse(game, samples_game_tree, show_progress=False)
    #     dbr.train(10, 5)
