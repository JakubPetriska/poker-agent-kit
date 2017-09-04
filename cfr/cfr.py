import operator
import random
from functools import reduce

from cfr.build_tree import build_game_tree
from cfr.constants import NUM_ACTIONS
from cfr.game_tree import HoleCardNode, TerminalNode, ActionNode

try:
    from tqdm import tqdm
except ImportError:
    print('!!! Install tqdm library for better progress information !!!\n')


class Cfr:
    def __init__(self, game):
        self.game = game
        self.game_tree = build_game_tree(game)
        self.player_count = game.get_num_players()

    @staticmethod
    def _calculate_node_average_strategy(node):
        num_possible_actions = len(node.children)
        normalizing_sum = sum(node.strategy_sum)
        if normalizing_sum > 0:
            node.average_strategy = [
                node.strategy_sum[a] / normalizing_sum if a in node.children else 0
                for a in range(NUM_ACTIONS)
                ]
        else:
            action_probability = 1.0 / num_possible_actions
            node.average_strategy = [
                action_probability if a in node.children else 0
                for a in range(NUM_ACTIONS)
                ]

    @staticmethod
    def _calculate_tree_average_strategy(node):
        if type(node) == ActionNode:
            Cfr._calculate_node_average_strategy(node)
        if node.children:
            for child in node.children.values():
                Cfr._calculate_tree_average_strategy(child)

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
            cards = [1, 2, 3]
            random.shuffle(cards)

            num_hole_cards = self.game.get_num_hole_cards()
            hole_cards = []
            for p in range(self.player_count):
                player_hole_cards = cards[:num_hole_cards]
                hole_cards.append(sorted(player_hole_cards))
                cards = cards[num_hole_cards:]

            self._cfr(
                [self.game_tree] * self.player_count,
                [1] * self.player_count,
                hole_cards,
                [False] * self.player_count)

        Cfr._calculate_tree_average_strategy(self.game_tree)

    def _cfr(self, nodes, occurrence_probabilities, hole_cards, players_folded):
        node_type = type(nodes[0])
        if node_type == TerminalNode:
            return self._cfr_terminal(
                nodes, occurrence_probabilities,
                hole_cards, players_folded)
        elif node_type == HoleCardNode:
            return self._cfr_hole_card(
                nodes, occurrence_probabilities,
                hole_cards, players_folded)
        return self._cfr_action(
            nodes, occurrence_probabilities,
            hole_cards, players_folded)

    def _cfr_terminal(self, nodes, occurrence_probabilities, hole_cards, players_folded):
        player_values = [sum(hole_cards[p]) for p in range(self.player_count)]

        showdown_player_values = filter(
            lambda player_val: not players_folded[player_val[0]],
            enumerate(player_values))
        winning_player = max(showdown_player_values, key=operator.itemgetter(1))[0]
        winner_prize = sum(nodes[0].pot_commitment) - nodes[0].pot_commitment[winning_player]
        return [
            winner_prize if p == winning_player else -nodes[0].pot_commitment[p]
            for p in range(self.player_count)]

    def _cfr_hole_card(self, nodes, occurrence_probabilities, hole_cards, players_folded):
        next_nodes = [node.children[hole_cards[p][node.card_index]]
                      for p, node in enumerate(nodes)]
        self._cfr(next_nodes, occurrence_probabilities, hole_cards, players_folded)

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

    def _cfr_action(self, nodes, occurrence_probabilities,
                    hole_cards, players_folded):
        node_player = nodes[0].player
        node = nodes[node_player]
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
                [node.children[a] for node in nodes],
                new_occurrence_probabilities,
                hole_cards, next_players_folded)
            util[a] = action_util
            for player in range(self.player_count):
                node_util[player] += strategy[a] * action_util[player]

        for a in range(NUM_ACTIONS):
            if not util[a]:
                continue
            regret = util[a][node_player] - node_util[node_player]

            opponent_occurrence_probabilities = occurrence_probabilities[0:player] \
                                                + occurrence_probabilities[player + 1:]
            occurrence_probability = reduce(operator.mul, opponent_occurrence_probabilities, 1)
            node.regret_sum[a] += regret * occurrence_probability

        return node_util
