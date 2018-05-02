import operator
import random
from functools import reduce
import numpy as np
import itertools
import math

import acpc_python_client as acpc

from tools.constants import NUM_ACTIONS
from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import NodeProvider
from tools.game_tree.nodes import HoleCardsNode, TerminalNode, StrategyActionNode, BoardCardsNode
from tools.hand_evaluation import get_utility
from tools.game_utils import get_num_hole_card_combinations
from tools.utils import is_unique, intersection

try:
    from tqdm import tqdm
except ImportError:
    pass


NUM_PLAYERS = 2


class CfrActionNode(StrategyActionNode):
    def __init__(self, parent, player):
        super().__init__(parent, player)
        self.current_strategy = np.zeros(NUM_ACTIONS)
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)


class CfrNodeProvider(NodeProvider):
    def create_action_node(self, parent, player):
        return CfrActionNode(parent, player)


class Cfr:
    """Creates new ACPC poker using CFR+ algorithm which runs for specified number of iterations.

    !!! Currently only limit betting games with up to 5 cards total and 2 players are supported !!!
    """

    def __init__(self, game, show_progress=True):
        """Build new CFR instance.

        Args:
            game (Game): ACPC game definition object.
        """
        self.game = game
        self.show_progress = show_progress

        if game.get_num_players() != 2:
            raise AttributeError(
                'Only games with 2 players are supported')

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

    @staticmethod
    def _calculate_node_average_strategy(node, minimal_action_probability):
        normalizing_sum = np.sum(node.strategy_sum)
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

    def _get_algorithm_name(self):
        return 'CFR'

    def train(
        self,
        iterations,
        weight_delay=700,
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
                iterations_iterable.set_description('%s training' % self._get_algorithm_name())
            except NameError:
                iterations_iterable = range(iterations)

        if iterations <= weight_delay:
            raise AttributeError('Number of iterations must be larger than weight delay')

        if checkpoint_iterations is None or checkpoint_iterations <= 0 or checkpoint_iterations > iterations:
            checkpoint_iterations = iterations

        iterations_left_to_checkpoint = weight_delay + checkpoint_iterations
        checkpoint_index = 0
        for i in iterations_iterable:
            self.weight = max(i - weight_delay, 0)
            for player in range(2):
                self._start_iteration(player)
            iterations_left_to_checkpoint -= 1

            if iterations_left_to_checkpoint == 0 or i == iterations - 1:
                Cfr._calculate_tree_average_strategy(self.game_tree, minimal_action_probability)
                checkpoint_callback(self.game_tree, checkpoint_index, i + 1)
                checkpoint_index += 1
                iterations_left_to_checkpoint = checkpoint_iterations

    def _start_iteration(self, player):
        self._cfr(
            player,
            [self.game_tree] * NUM_PLAYERS,
            None,
            [],
            [False] * NUM_PLAYERS,
            1)

    def _cfr(self, player, nodes, hole_cards, board_cards, players_folded, opponent_reach_prob):
        node_type = type(nodes[0])
        if node_type == TerminalNode:
            return self._cfr_terminal(
                player,
                nodes,
                hole_cards,
                board_cards,
                players_folded,
                opponent_reach_prob)
        elif node_type == HoleCardsNode:
            return self._cfr_hole_cards(
                player,
                nodes,
                hole_cards,
                board_cards,
                players_folded,
                opponent_reach_prob)
        elif node_type == BoardCardsNode:
            return self._cfr_board_cards(
                player,
                nodes,
                hole_cards,
                board_cards,
                players_folded,
                opponent_reach_prob)
        else:
            return self._cfr_action(
                player,
                nodes,
                hole_cards,
                board_cards,
                players_folded,
                opponent_reach_prob)

    def _cfr_terminal(self, player, nodes, hole_cards, board_cards, players_folded, opponent_reach_prob):
        return get_utility(
            hole_cards,
            board_cards,
            players_folded,
            nodes[0].pot_commitment)[player] * opponent_reach_prob

    def _cfr_hole_cards(self, player, nodes, hole_cards, board_cards, players_folded, opponent_reach_prob):
        hole_card_combination_probability = 1 / get_num_hole_card_combinations(self.game)
        hole_cards = [node.children for node in nodes]
        hole_card_combinations = filter(lambda comb: is_unique(*comb), itertools.product(*hole_cards))

        value_sum = 0
        for hole_cards_combination in hole_card_combinations:
            next_nodes = [node.children[hole_cards_combination[i]] for i, node in enumerate(nodes)]
            player_utility = self._cfr(
                player,
                next_nodes,
                hole_cards_combination,
                board_cards,
                players_folded,
                opponent_reach_prob)
            value_sum += player_utility * hole_card_combination_probability
        return value_sum

    def _cfr_board_cards(self, player, nodes, hole_cards, board_cards, players_folded, opponent_reach_prob):
        possible_board_cards = intersection(*map(lambda node: node.children, nodes))
        board_cards_combination_probability = 1 / len(possible_board_cards)

        value_sum = 0
        for next_board_cards in possible_board_cards:
            selected_board_cards = sorted(next_board_cards)
            selected_board_cards_key = tuple(selected_board_cards)
            next_nodes = [node.children[selected_board_cards_key] for i, node in enumerate(nodes)]
            player_utility = self._cfr(
                player,
                next_nodes,
                hole_cards,
                board_cards + list(selected_board_cards),
                players_folded,
                opponent_reach_prob)
            value_sum += player_utility * board_cards_combination_probability
        return value_sum

    @staticmethod
    def _regret_matching(nodes):
        node = nodes[nodes[0].player]
        normalizing_sum = np.sum(node.regret_sum)
        if normalizing_sum > 0:
            node.current_strategy = node.regret_sum / normalizing_sum
        else:
            action_probability = 1 / len(node.children)
            current_strategy = np.zeros(NUM_ACTIONS)
            for a in node.children:
                current_strategy[a] = action_probability
            node.current_strategy = current_strategy

    def _get_current_strategy(self, nodes):
        return nodes[nodes[0].player].current_strategy

    def _cfr_action(self, player, nodes, hole_cards, board_cards, players_folded, opponent_reach_prob):
        node_player = nodes[0].player
        node = nodes[node_player]

        node_util = 0
        if player == node_player:
            current_strategy = self._get_current_strategy(nodes)

            util = np.zeros(NUM_ACTIONS)
            for a in node.children:
                if a == 0:
                    next_players_folded = list(players_folded)
                    next_players_folded[node_player] = True
                else:
                    next_players_folded = players_folded

                action_util = self._cfr(
                    player,
                    [node.children[a] for node in nodes],
                    hole_cards,
                    board_cards,
                    next_players_folded,
                    opponent_reach_prob)

                util[a] = action_util
                node_util += current_strategy[a] * action_util

            for a in node.children:
                node.regret_sum[a] = max(node.regret_sum[a] + util[a] - node_util, 0)

        else:
            Cfr._regret_matching(nodes)
            current_strategy = self._get_current_strategy(nodes)
            for a in node.children:
                if a == 0:
                    next_players_folded = list(players_folded)
                    next_players_folded[node_player] = True
                else:
                    next_players_folded = players_folded

                node_util += self._cfr(
                    player,
                    [node.children[a] for node in nodes],
                    hole_cards,
                    board_cards,
                    next_players_folded,
                    opponent_reach_prob * current_strategy[a])

            node.strategy_sum += opponent_reach_prob * current_strategy * self.weight
        return node_util
