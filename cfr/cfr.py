import operator
import random
from functools import reduce

import acpc_python_client as acpc

from cfr.build_tree import GameTreeBuilder
from cfr.constants import NUM_ACTIONS
from cfr.game_tree import HoleCardsNode, TerminalNode, ActionNode, BoardCardsNode

try:
    from tqdm import tqdm
except ImportError:
    print('!!! Install tqdm library for better progress information !!!\n')


class Cfr:
    def __init__(self, game):
        self.game = game

        game_tree_builder = GameTreeBuilder(game)

        try:
            with tqdm(total=1) as progress:
                progress.set_description('Building game tree')
                self.game_tree = game_tree_builder.build_tree()
                progress.update(1)
        except NameError:
            self.game_tree = game_tree_builder.build_tree()

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

        deck = acpc.game_utils.generate_deck(self.game)
        for i in iterations_iterable:
            current_deck = list(deck)
            random.shuffle(current_deck)

            self._cfr(
                [self.game_tree] * self.player_count,
                [1] * self.player_count,
                None, [], current_deck,
                [False] * self.player_count)

        Cfr._calculate_tree_average_strategy(self.game_tree)

    def _cfr(self, nodes, reach_probs, hole_cards, board_cards, deck, players_folded):
        node_type = type(nodes[0])
        if node_type == TerminalNode:
            return self._cfr_terminal(
                nodes, hole_cards, board_cards, deck,
                players_folded)
        elif node_type == HoleCardsNode:
            return self._cfr_hole_cards(
                nodes, reach_probs,
                hole_cards, board_cards, deck,
                players_folded)
        elif node_type == BoardCardsNode:
            return self._cfr_board_cards(
                nodes, reach_probs,
                hole_cards, board_cards, deck,
                players_folded)
        return self._cfr_action(
            nodes, reach_probs,
            hole_cards, board_cards, deck,
            players_folded)

    def _cfr_terminal(self, nodes, hole_cards, board_cards, deck, players_folded):
        # TODO consider board cards and combinations
        player_values = [sum(hole_cards[p]) for p in range(self.player_count)]

        showdown_player_values = filter(
            lambda player_val: not players_folded[player_val[0]],
            enumerate(player_values))
        winning_player = max(showdown_player_values, key=operator.itemgetter(1))[0]
        winner_prize = sum(nodes[0].pot_commitment) - nodes[0].pot_commitment[winning_player]
        return [
            winner_prize if p == winning_player else -nodes[0].pot_commitment[p]
            for p in range(self.player_count)]

    def _cfr_hole_cards(self, nodes, reach_probs, hole_cards, board_cards, deck, players_folded):
        num_hole_cards = nodes[0].card_count
        next_hole_cards = []
        next_deck = list(deck)
        for p in range(self.player_count):
            next_hole_cards.append(tuple(sorted(next_deck[:num_hole_cards])))
            next_deck = next_deck[num_hole_cards:]

        next_nodes = [node.children[next_hole_cards[p]]
                      for p, node in enumerate(nodes)]
        return self._cfr(next_nodes, reach_probs, next_hole_cards, board_cards, next_deck, players_folded)

    def _cfr_board_cards(self, nodes, reach_probs, hole_cards, board_cards, deck, players_folded):
        num_board_cards = nodes[0].card_count
        selected_board_cards = tuple(sorted(deck[:num_board_cards]))
        next_nodes = [node.children[selected_board_cards]
                      for p, node in enumerate(nodes)]
        return self._cfr(next_nodes, reach_probs,
                         hole_cards, board_cards + [selected_board_cards], deck[num_board_cards:],
                         players_folded)

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

    def _cfr_action(self, nodes, reach_probs,
                    hole_cards, board_cards, deck, players_folded):
        node_player = nodes[0].player
        node = nodes[node_player]
        Cfr._update_node_strategy(node, reach_probs[node_player])
        strategy = node.strategy
        util = [None] * NUM_ACTIONS
        node_util = [0] * self.player_count
        for a in range(NUM_ACTIONS):
            if a not in node.children:
                continue

            next_reach_probs = list(reach_probs)
            next_reach_probs[node_player] *= strategy[a]

            if a == 0:
                next_players_folded = list(players_folded)
                next_players_folded[node_player] = True
            else:
                next_players_folded = players_folded

            action_util = self._cfr(
                [node.children[a] for node in nodes],
                next_reach_probs,
                hole_cards, board_cards, deck, next_players_folded)
            util[a] = action_util
            for player in range(self.player_count):
                node_util[player] += strategy[a] * action_util[player]

        for a in range(NUM_ACTIONS):
            if not util[a]:
                continue
            regret = util[a][node_player] - node_util[node_player]

            opponent_reach_probs = reach_probs[0:player] \
                                   + reach_probs[player + 1:]
            reach_prob = reduce(operator.mul, opponent_reach_probs, 1)
            node.regret_sum[a] += regret * reach_prob

        return node_util
