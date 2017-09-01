import operator
import random
from functools import reduce

from cfr.constants import NUM_ACTIONS
from cfr.game_tree import HoleCardNode, TerminalNode

try:
    from tqdm import tqdm
except ImportError:
    print('!!! Install tqdm library for better progress information !!!\n')

NUM_PLAYERS = 2


class Cfr:
    def __init__(self, player_count, game_tree_root):
        self.player_count = player_count
        self.game_tree_root = game_tree_root

    def train(self, iterations, show_progress=True):
        cards = [1, 2, 3]

        if not show_progress:
            iterations_iterable = range(iterations)
        else:
            try:
                iterations_iterable = tqdm(range(iterations))
            except NameError:
                iterations_iterable = range(iterations)

        for i in iterations_iterable:
            random.shuffle(cards)
            self._cfr(self.game_tree_root, [1] * self.player_count, [])

    def _cfr(self, node, occurrence_probabilities,
             player_hole_cards, deck_cards):
        if type(node) == TerminalNode:
            return self._cfr_terminal(
                node, occurrence_probabilities,
                player_hole_cards, deck_cards)
        elif type(node) == HoleCardNode:
            return self._cfr_hole_card(
                node, occurrence_probabilities,
                player_hole_cards, deck_cards)
        return self._cfr_action(
            node, occurrence_probabilities,
            player_hole_cards, deck_cards)

    def _cfr_terminal(self, node, occurrence_probabilities, player_hole_cards, deck_cards):
        # TODO return utility for each player
        return [i for i in range(self.player_count)]

    def _cfr_hole_card(self, node, occurrence_probabilities, player_hole_cards, deck_cards):
        cards = node.possible_cards
        card_index = random.randrange(len(cards))
        hole_card = cards[card_index]
        player_hole_cards.append(hole_card)
        self._cfr(node.children[hole_card], occurrence_probabilities,
                  player_hole_cards,
                  cards[0:card_index] + cards[card_index + 1:])

    @staticmethod
    def _update_node_strategy(node, realization_weight):
        normalizing_sum = 0
        for a in range(NUM_ACTIONS):
            node.strategy[a] = node.regret_sum[a] if node.regret_sum[a] > 0 else 0
            normalizing_sum += node.strategy[a]

        for a in range(NUM_ACTIONS):
            if normalizing_sum > 0:
                node.strategy[a] /= normalizing_sum
            else:
                node.strategy[a] = 1.0 / NUM_ACTIONS
            node.strategy_sum[a] += realization_weight * node.strategy[a]

    def _cfr_action(self, node, occurrence_probabilities, player_hole_cards, deck_cards):
        node_player = node.player
        Cfr._update_node_strategy(node, occurrence_probabilities[node_player])
        strategy = node.strategy
        util = []
        node_util = [0] * self.player_count
        for a in range(NUM_ACTIONS):
            action_util = self._cfr(node.children[a], occurrence_probabilities,
                                    player_hole_cards, deck_cards)
            util.append(action_util)
            for player in range(self.player_count):
                node_util[player] += strategy[a] * action_util[player]

        for a in range(NUM_ACTIONS):
            regret = util[a][node_player] - node_util[node_player]

            opponent_occurrence_probabilities = occurrence_probabilities[0:player] \
                                                + occurrence_probabilities[player + 1:]
            occurrence_probability = reduce(operator.mul, opponent_occurrence_probabilities, 1)
            node.regret_sum[a] += regret * occurrence_probability

        return node_util
