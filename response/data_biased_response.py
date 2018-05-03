import random
import numpy as np

from cfr.main import Cfr, NUM_PLAYERS


class DataBiasedResponse(Cfr):
    def __init__(
            self,
            game,
            opponent_sample_tree,
            p_max=0.8,
            show_progress=True):
        super().__init__(game, show_progress)
        self.opponent_sample_tree = opponent_sample_tree
        self.p_max = p_max

    def _get_algorithm_name(self):
        return 'DBR'

    def _start_iteration(self, player):
        self._cfr(
            player,
            ([self.game_tree] * NUM_PLAYERS) + [self.opponent_sample_tree],
            None,
            [],
            [False] * NUM_PLAYERS,
            1)

    def _get_opponent_strategy(self, nodes):
        samples_node = nodes[-1]
        samples_count = np.sum(samples_node.action_decision_counts)
        p_conf = self.p_max * min(1, samples_count / 10)
        if random.random() <= p_conf:
            return samples_node.action_decision_counts / samples_count
        else:
            return super(DataBiasedResponse, self)._get_opponent_strategy(nodes)
