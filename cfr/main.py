import operator
import random
from functools import reduce

import acpc_python_client as acpc

from cfr.build_tree import GameTreeBuilder
from cfr.constants import NUM_ACTIONS
from cfr.game_tree import HoleCardsNode, TerminalNode, ActionNode, BoardCardsNode
from tools.hand_evaluation import get_winners

try:
    from tqdm import tqdm
except ImportError:
    pass


class Cfr:
    """Creates new ACPC poker using CFR algorithm which runs for specified number of iterations.

    !!! Currently only limit betting games and games with up to 5 cards total are supported !!!
    """

    def __init__(self, game):
        """Build new CFR instance.

        Args:
            game (Game): ACPC game definition object.
        """
        self.game = game

        if game.get_betting_type() != acpc.BettingType.LIMIT:
            raise AttributeError('No-limit betting games not supported')

        total_cards_count = game.get_num_hole_cards() + game.get_total_num_board_cards(game.get_num_rounds() - 1)
        if total_cards_count > 5:
            raise AttributeError('Only games with up to 5 cards are supported')

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
        """Run CFR for given number of iterations.

        The trained tree can be found by retrieving the game_tree
        property from this object. The result strategy is stored
        in average_strategy of each ActionNode in game tree.

        This method can be called multiple times on one instance
        to train more. This can be used for evaluation during training
        and to make number of training iterations dynamic.

        Args:
            iterations (int): Number of iterations.
            show_progress (bool): Show training progress bar.
        """
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
        player_count = self.player_count
        pot_commitment = nodes[0].pot_commitment

        if sum(players_folded) == player_count - 1:
            prize = sum(pot_commitment)
            return [-pot_commitment[player] if players_folded[player] else prize - pot_commitment[player]
                    for player in range(player_count)]

        flattened_board_cards = reduce(
            lambda res, cards: res + list(cards), board_cards, [])
        player_cards = [(list(hole_cards[p]) + flattened_board_cards) if not players_folded[p] else None
                        for p in range(player_count)]
        winners = get_winners(player_cards)
        winner_count = len(winners)
        value_per_winner = sum(pot_commitment) / winner_count
        return [value_per_winner - pot_commitment[p] if p in winners else -pot_commitment[p]
                for p in range(player_count)]

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
        """Update node strategy by normalizing regret sums."""
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
        for a in node.children:
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

        for a in node.children:
            # Calculate regret and add it to regret sums
            regret = util[a][node_player] - node_util[node_player]

            opponent_reach_probs = reach_probs[0:node_player] + reach_probs[node_player + 1:]
            reach_prob = reduce(operator.mul, opponent_reach_probs, 1)
            node.regret_sum[a] += regret * reach_prob

        return node_util
