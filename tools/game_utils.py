import numpy as np
import scipy.misc
import scipy.special

from tools.walk_tree import walk_tree
from tools.game_tree.nodes import ActionNode


def get_num_hole_card_combinations(game):
    num_players = game.get_num_players()
    num_hole_cards = game.get_num_hole_cards()
    num_cards = game.get_num_suits() * game.get_num_ranks()
    num_total_hole_cards = num_players * num_hole_cards
    return scipy.misc.comb(num_cards, num_total_hole_cards, exact=True) \
        * scipy.special.perm(num_total_hole_cards, num_total_hole_cards, exact=True)


def is_correct_strategy(strategy_tree):
    correct = True
    def on_node(node):
        if isinstance(node, ActionNode):
            nonlocal correct
            strategy_sum = np.sum(node.strategy)
            if strategy_sum != 1:
                correct = False
    walk_tree(strategy_tree, on_node)
    return correct
