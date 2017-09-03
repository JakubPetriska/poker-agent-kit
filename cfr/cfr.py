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
        if not show_progress:
            iterations_iterable = range(iterations)
        else:
            try:
                iterations_iterable = tqdm(range(iterations))
                iterations_iterable.set_description('CFR training')
            except NameError:
                iterations_iterable = range(iterations)

        for i in iterations_iterable:
            for player in range(self.player_count):
                cards = [1, 2, 3]
                random.shuffle(cards)
                self._cfr(
                    player, self.game_tree_root,
                    [1] * self.player_count, [],
                    cards, [False] * self.player_count)

    def _cfr(self, current_player, node, occurrence_probabilities,
             player_hole_cards, deck_cards, players_folded):
        if type(node) == TerminalNode:
            return self._cfr_terminal(
                current_player, node, occurrence_probabilities,
                player_hole_cards, deck_cards, players_folded)
        elif type(node) == HoleCardNode:
            return self._cfr_hole_card(
                current_player, node, occurrence_probabilities,
                player_hole_cards, deck_cards, players_folded)
        return self._cfr_action(
            current_player, node, occurrence_probabilities,
            player_hole_cards, deck_cards, players_folded)

    def _cfr_terminal(self, current_player, node, occurrence_probabilities,
                      player_hole_cards, deck_cards, players_folded):
        hole_cards_count = len(player_hole_cards)
        hole_cards = [None] * self.player_count
        for p in range(self.player_count):
            if p == current_player:
                hole_cards[p] = player_hole_cards
            else:
                new_hole_cards = deck_cards[0:hole_cards_count]
                deck_cards = deck_cards[hole_cards_count:]
                hole_cards[p] = new_hole_cards

        player_values = [sum(hole_cards[p]) for p in range(self.player_count)]

        showdown_player_values = filter(
            lambda player_val: not players_folded[player_val[0]],
            enumerate(player_values))
        winning_player = max(showdown_player_values, key=operator.itemgetter(1))[0]
        winner_prize = sum(node.pot_commitment) - node.pot_commitment[winning_player]
        return [
            winner_prize if p == winning_player else -node.pot_commitment[p]
            for p in range(self.player_count)]

    def _cfr_hole_card(self, current_player, node, occurrence_probabilities,
                       player_hole_cards, deck_cards, players_folded):
        hole_card = deck_cards[0]
        player_hole_cards.append(hole_card)
        self._cfr(current_player, node.children[hole_card], occurrence_probabilities,
                  player_hole_cards, deck_cards[1:], players_folded)

    @staticmethod
    def _update_node_strategy(node, realization_weight):
        normalizing_sum = 0
        for a in range(NUM_ACTIONS):
            node.strategy[a] = node.regret_sum[a] if node.regret_sum[a] > 0 else 0
            normalizing_sum += node.strategy[a]

        num_possible_actions = len(node.children)
        for a in range(NUM_ACTIONS):
            if normalizing_sum > 0:
                node.strategy[a] /= normalizing_sum
            elif a in node.children:
                node.strategy[a] = 1.0 / num_possible_actions
            else:
                node.strategy[a] = 0
            node.strategy_sum[a] += realization_weight * node.strategy[a]

    def _cfr_action(self, current_player, node, occurrence_probabilities,
                    player_hole_cards, deck_cards, players_folded):
        node_player = node.player
        Cfr._update_node_strategy(node, occurrence_probabilities[node_player])
        strategy = node.strategy
        util = [None] * NUM_ACTIONS
        node_util = [0] * self.player_count
        for a in range(NUM_ACTIONS):
            if a not in node.children:
                continue

            new_occurrence_probabilities = list(occurrence_probabilities)
            new_occurrence_probabilities[node_player] *= strategy[a]

            if a == 0:
                next_players_folded = list(players_folded)
                next_players_folded[node_player] = True
            else:
                next_players_folded = players_folded

            action_util = self._cfr(
                current_player, node.children[a], new_occurrence_probabilities,
                player_hole_cards, deck_cards, next_players_folded)
            util[a] = action_util
            for player in range(self.player_count):
                node_util[player] += strategy[a] * action_util[player]

        if node_player == current_player:
            for a in range(NUM_ACTIONS):
                if not util[a]:
                    continue
                regret = util[a][current_player] - node_util[current_player]

                opponent_occurrence_probabilities = occurrence_probabilities[0:player] \
                                                    + occurrence_probabilities[player + 1:]
                occurrence_probability = reduce(operator.mul, opponent_occurrence_probabilities, 1)
                node.regret_sum[a] += regret * occurrence_probability

        return node_util
