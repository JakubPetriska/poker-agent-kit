import operator
import random
from functools import reduce
import numpy as np
import itertools

import acpc_python_client as acpc

from tools.constants import NUM_ACTIONS
from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import NodeProvider
from tools.game_tree.nodes import HoleCardsNode, TerminalNode, StrategyActionNode, BoardCardsNode
from tools.hand_evaluation import get_winners
from tools.game_utils import get_num_hole_card_combinations
from tools.utils import is_unique, intersection

try:
    from tqdm import tqdm
except ImportError:
    pass


class CfrActionNode(StrategyActionNode):
    def __init__(self, parent, player):
        super().__init__(parent, player)
        self.training_strategy = np.zeros(NUM_ACTIONS)
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)


class CfrNodeProvider(NodeProvider):
    def create_action_node(self, parent, player):
        return CfrActionNode(parent, player)


class Cfr:
    """Creates new ACPC poker using CFR algorithm which runs for specified number of iterations.

    !!! Currently only limit betting games and games with up to 5 cards total are supported !!!
    """

    def __init__(self, game, show_progress=True):
        """Build new CFR instance.

        Args:
            game (Game): ACPC game definition object.
        """
        self.game = game
        self.show_progress = show_progress

        if game.get_betting_type() != acpc.BettingType.LIMIT:
            raise AttributeError('No-limit betting games not supported')

        total_cards_count = game.get_num_hole_cards() \
            + game.get_total_num_board_cards(game.get_num_rounds() - 1)
        if total_cards_count > 5:
            raise AttributeError('Only games with up to 5 cards are supported')

        game_tree_builder = GameTreeBuilder(game, CfrNodeProvider())

        if not self.show_progress:
            self.game_tree = game_tree_builder.build_tree()
        else:
            try:
                with tqdm(total=1) as progress:
                    progress.set_description('Building game tree')
                    self.game_tree = game_tree_builder.build_tree()
                    progress.update(1)
            except NameError:
                self.game_tree = game_tree_builder.build_tree()

        self.player_count = game.get_num_players()

    @staticmethod
    def _calculate_node_average_strategy(node, minimal_action_probability):
        normalizing_sum = sum(node.strategy_sum)
        if normalizing_sum > 0:
            node.strategy = np.array(node.strategy_sum) / normalizing_sum
            if minimal_action_probability:
                normalize = False
                for a in range(NUM_ACTIONS):
                    action_probability = node.strategy[a]
                    if action_probability > 0 and action_probability < minimal_action_probability:
                        node.strategy[a] = 0
                        normalize = True
                if normalize:
                    node.strategy = node.strategy / sum(node.strategy)
        else:
            action_probability = 1.0 / len(node.children)
            node.strategy = [
                action_probability if a in node.children else 0
                for a in range(NUM_ACTIONS)]

    @staticmethod
    def _calculate_tree_average_strategy(node, minimal_action_probability):
        if isinstance(node, CfrActionNode):
            Cfr._calculate_node_average_strategy(node, minimal_action_probability)
        if node.children:
            for child in node.children.values():
                Cfr._calculate_tree_average_strategy(child, minimal_action_probability)

    def train(
        self,
        iterations,
        checkpoint_iterations=None,
        checkpoint_callback=lambda *args: None,
        minimal_action_probability=None):
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
        if not self.show_progress:
            iterations_iterable = range(iterations)
        else:
            try:
                iterations_iterable = tqdm(range(iterations))
                iterations_iterable.set_description('CFR training')
            except NameError:
                iterations_iterable = range(iterations)

        if checkpoint_iterations is None or checkpoint_iterations <= 0 or checkpoint_iterations > iterations:
            checkpoint_iterations = iterations

        iterations_left_to_checkpoint = checkpoint_iterations
        checkpoint_index = 0
        for i in iterations_iterable:
            self._cfr(
                [self.game_tree] * self.player_count,
                np.ones(self.player_count),
                None,
                [],
                [False] * self.player_count)
            iterations_left_to_checkpoint -= 1

            if iterations_left_to_checkpoint == 0 or i == iterations - 1:
                Cfr._calculate_tree_average_strategy(self.game_tree, minimal_action_probability)
                checkpoint_callback(self.game_tree, checkpoint_index, i + 1)
                checkpoint_index += 1
                iterations_left_to_checkpoint = checkpoint_iterations

    def _cfr(self, nodes, reach_probs, hole_cards, board_cards, players_folded):
        node_type = type(nodes[0])
        if node_type == TerminalNode:
            return self._cfr_terminal(
                nodes,
                hole_cards,
                board_cards,
                players_folded)
        elif node_type == HoleCardsNode:
            return self._cfr_hole_cards(
                nodes,
                reach_probs,
                hole_cards,
                board_cards,
                players_folded)
        elif node_type == BoardCardsNode:
            return self._cfr_board_cards(
                nodes,
                reach_probs,
                hole_cards,
                board_cards,
                players_folded)
        return self._cfr_action(
            nodes,
            reach_probs,
            hole_cards,
            board_cards,
            players_folded)

    def _cfr_terminal(self, nodes, hole_cards, board_cards, players_folded):
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
        return np.array([value_per_winner - pot_commitment[p] if p in winners else -pot_commitment[p]
            for p in range(player_count)])

    def _cfr_hole_cards(self, nodes, reach_probs, hole_cards, board_cards, players_folded):
        hole_card_combination_probability = 1 / get_num_hole_card_combinations(self.game)
        next_reach_probs = reach_probs * hole_card_combination_probability
        value_sums = np.zeros(self.player_count)
        hole_cards = [node.children for node in nodes]
        hole_card_combinations = filter(lambda comb: is_unique(*comb), itertools.product(*hole_cards))
        for hole_cards_combination in hole_card_combinations:
            next_nodes = [node.children[hole_cards_combination[i]] for i, node in enumerate(nodes)]
            player_values = self._cfr(
                next_nodes,
                next_reach_probs,
                hole_cards_combination,
                board_cards,
                players_folded)
            value_sums += player_values * hole_card_combination_probability
        return value_sums

    def _cfr_board_cards(self, nodes, reach_probs, hole_cards, board_cards, players_folded):
        possible_board_cards = intersection(*map(lambda node: node.children, nodes))
        board_cards_combination_probability = 1 / len(possible_board_cards)
        next_reach_probs = reach_probs * board_cards_combination_probability

        value_sums = np.zeros(self.player_count)
        for next_board_cards in possible_board_cards:
            selected_board_cards = tuple(sorted(next_board_cards))
            next_nodes = [node.children[selected_board_cards] for i, node in enumerate(nodes)]
            player_values = self._cfr(
                next_nodes,
                next_reach_probs,
                hole_cards,
                board_cards + [selected_board_cards],
                players_folded)
            value_sums += player_values * board_cards_combination_probability
        return value_sums

    @staticmethod
    def _update_node_strategy(node, realization_weight):
        """Update node strategy by normalizing regret sums."""
        normalizing_sum = 0
        for a in node.children:
            node.training_strategy[a] = node.regret_sum[a] if node.regret_sum[a] > 0 else 0
            normalizing_sum += node.training_strategy[a]

        num_possible_actions = len(node.children)
        for a in node.children:
            if normalizing_sum > 0:
                node.training_strategy[a] /= normalizing_sum
            else:
                node.training_strategy[a] = 1.0 / num_possible_actions
            node.strategy_sum[a] += realization_weight * node.training_strategy[a]

    def _cfr_action(self, nodes, reach_probs,
                    hole_cards, board_cards, players_folded):
        node_player = nodes[0].player
        node = nodes[node_player]
        Cfr._update_node_strategy(node, reach_probs[node_player])
        strategy = node.training_strategy
        util = [None] * NUM_ACTIONS
        node_util = np.zeros(self.player_count)
        for a in node.children:
            next_reach_probs =  np.copy(reach_probs)
            next_reach_probs[node_player] *= strategy[a]

            if a == 0:
                next_players_folded = list(players_folded)
                next_players_folded[node_player] = True
            else:
                next_players_folded = players_folded

            action_util = self._cfr(
                [node.children[a] for node in nodes],
                next_reach_probs,
                hole_cards,
                board_cards,
                next_players_folded)
            util[a] = action_util
            for player in range(self.player_count):
                node_util[player] += strategy[a] * action_util[player]

        for a in node.children:
            # Calculate regret and add it to regret sums
            regret = util[a][node_player] - node_util[node_player]

            opponent_reach_probs = np.concatenate([
                reach_probs[0:node_player],
                reach_probs[node_player + 1:]])
            node.regret_sum[a] += regret * np.prod(opponent_reach_probs)

        return node_util
