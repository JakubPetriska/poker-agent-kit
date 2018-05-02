import random
import numpy as np

from cfr.main import Cfr, NUM_PLAYERS


class RestrictedNashResponse(Cfr):
    def __init__(
            self,
            game,
            opponent_strategy_tree,
            p,
            show_progress=True):
        super().__init__(game, show_progress)
        self.opponent_strategy_tree = opponent_strategy_tree
        self.p = p

    def _start_iteration(self, player):
        self.play_fix = random.random() <= self.p
        self._cfr(
            player,
            ([self.game_tree] * NUM_PLAYERS) + [self.opponent_strategy_tree],
            None,
            [],
            [False] * NUM_PLAYERS,
            1)

    def _get_current_strategy(self, nodes):
        opponent_strategy_node = nodes[-1]
        if self.play_fix:
            return opponent_strategy_node.strategy
        else:
            return super(RestrictedNashResponse, self)._get_current_strategy(nodes)
