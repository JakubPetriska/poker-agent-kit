import random
import numpy as np

from cfr.main import Cfr, NUM_PLAYERS
from tools.game_tree.nodes import ActionNode
from tools.walk_trees import walk_trees


class DataBiasedResponse(Cfr):
    def __init__(
            self,
            game,
            opponent_sample_tree,
            p_max=0.8,
            show_progress=True):
        super().__init__(game, show_progress)
        self.p_max = p_max

        opponent_action_decision_counts = {}
        def callback(node):
            if isinstance(node, ActionNode):
                nonlocal opponent_action_decision_counts
                opponent_action_decision_counts[str(node)] = node.action_decision_counts
        walk_trees(callback, opponent_sample_tree)
        self.opponent_action_decision_counts = opponent_action_decision_counts

    def _get_algorithm_name(self):
        return 'DBR'

    def _get_opponent_strategy(self, player, nodes):
        opponent_index = (player + 1) % 2
        action_decision_counts = self.opponent_action_decision_counts[str(nodes[opponent_index])]
        samples_count = np.sum(action_decision_counts)
        p_conf = self.p_max * min(1, samples_count / 10)
        if random.random() <= p_conf:
            return action_decision_counts / samples_count
        else:
            return super(DataBiasedResponse, self)._get_opponent_strategy(nodes, player)
