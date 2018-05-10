import random
import numpy as np

from cfr.main import Cfr, NUM_PLAYERS
from tools.game_tree.nodes import ActionNode
from tools.walk_trees import walk_trees


class RestrictedNashResponse(Cfr):
    def __init__(
            self,
            game,
            opponent_strategy_tree,
            p,
            show_progress=True):
        super().__init__(game, show_progress)
        self.p = p

        opponent_strategy = {}
        def callback(node):
            if isinstance(node, ActionNode):
                nonlocal opponent_strategy
                opponent_strategy[str(node)] = node.strategy
        walk_trees(callback, opponent_strategy_tree)
        self.opponent_strategy = opponent_strategy


    def _get_algorithm_name(self):
        return 'RNR'

    def _start_iteration(self, player):
        self.play_fix = random.random() <= self.p
        super(RestrictedNashResponse, self)._start_iteration(player)

    def _get_opponent_strategy(self, player, nodes):
        if self.play_fix:
            return self.opponent_strategy[str(nodes[(player + 1) % 2])]
        else:
            return super(RestrictedNashResponse, self)._get_opponent_strategy(player, nodes)
