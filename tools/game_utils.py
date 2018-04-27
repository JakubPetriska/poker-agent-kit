from math import isclose
import numpy as np
import scipy.misc
import scipy.special

from tools.walk_tree import walk_tree, walk_tree_with_data
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
            if not isclose(strategy_sum, 1):
                correct = False
            for i in range(3):
                if i not in node.children and node.strategy[i] != 0:
                    correct = False
    walk_tree(strategy_tree, on_node)
    return correct

def copy_strategy(dst, src):
    def on_node(dst_node, src_node):
        if isinstance(dst_node, ActionNode):
            np.copyto(dst_node.strategy, src_node.strategy)
        return [src_node.children[a] for a in src_node.children]
    walk_tree_with_data(dst, src, on_node)
