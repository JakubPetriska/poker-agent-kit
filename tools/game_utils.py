from math import isclose
import numpy as np
import scipy.misc
import scipy.special

from tools.walk_trees import walk_trees
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
    walk_trees(on_node, strategy_tree)
    return correct


def copy_strategy(dst, src):
    def on_node(dst_node, src_node):
        if isinstance(dst_node, ActionNode):
            np.copyto(dst_node.strategy, src_node.strategy)
        return [src_node.children[a] for a in src_node.children]
    walk_trees(on_node, dst, src)


def is_strategies_equal(first, second):
    equal = True
    def on_node(first_node, second_node):
        if isinstance(first_node, ActionNode):
            for a in range(3):
                if not isclose(first_node.strategy[a], second_node.strategy[a]):
                    nonlocal equal
                    equal = False
    walk_trees(on_node, first, second)
    return equal

def get_big_blind_size(game):
    big_blind = None
    for i in range(game.get_num_players()):
        player_blind = game.get_blind(i)
        if big_blind == None or player_blind > big_blind:
            big_blind = player_blind
    return big_blind
