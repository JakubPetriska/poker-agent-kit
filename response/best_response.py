import itertools

from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import StrategyTreeNodeProvider
from acpc_python_client.game_utils import generate_deck
from tools.game_tree.nodes import HoleCardsNode, TerminalNode, StrategyActionNode, BoardCardsNode
import numpy as np
from tools.hand_evaluation import get_winners
from tools.utils import flatten, intersection


class BestResponse:
    def __init__(self, game):
        self.game = game
        if game.get_num_players() != 2:
            raise AttributeError(
                'Only games with two players are supported')
        self.big_blind = None
        for i in range(game.get_num_players()):
            player_blind = game.get_blind(i)
            if self.big_blind == None or player_blind > self.big_blind:
                self.big_blind = player_blind

    def solve(self, strategy):
        game_tree_builder = GameTreeBuilder(self.game, StrategyTreeNodeProvider())
        best_response = game_tree_builder.build_tree()

        for position in range(2):
            self._get_exploitability(position, best_response, strategy, [], [])

        return best_response

    def _get_terminal_node_player_utility(self, player_position, node, best_response_cards, board_cards):
        parent_action = list(filter(lambda item: item[1] == node, node.parent.children.items()))[0][0]
        if parent_action == 0:
            player_folded = node.parent.player
            pot_amount = np.sum(node.pot_commitment)
            return -node.pot_commitment[player_position] + \
                (pot_amount if player_position != player_folded else 0)


        deck = filter(
            lambda card: card not in best_response_cards and card not in board_cards,
            generate_deck(self.game))
        num_hole_cards = self.game.get_num_hole_cards()

        player_value_sum = 0
        num_player_values = 0
        for player_cards in itertools.combinations(deck, num_hole_cards):
            hands = None
            if player_position == 0:
                hands = [player_cards, best_response_cards]
            else:
                hands = [best_response_cards, player_cards]
            winners = get_winners(hands)
            winner_count = len(winners)
            pot_amount = np.sum(node.pot_commitment)
            per_winner_value = pot_amount / winner_count
            player_value_sum += -node.pot_commitment[player_position] + \
                (per_winner_value if player_position in winners else 0)
            num_player_values += 1
        return player_value_sum / num_player_values

    def _get_exploitability(
            self, player_position, best_response_node, player_possible_nodes, best_response_cards, board_cards):
        if isinstance(best_response_node, TerminalNode):
            return self._get_terminal_node_player_utility(player_position, best_response_node, best_response_cards, board_cards)
        elif isinstance(best_response_node, HoleCardsNode):
            player_values_sum = 0
            for cards in best_response_node.children:
                new_player_cards = flatten(best_response_cards, cards)
                next_player_possible_nodes = []
                for other_cards in best_response_node.children:
                    if len(intersection(cards, other_cards)) == 0 \
                        and len(intersection(cards, board_cards)) == 0:
                        next_player_possible_nodes.append(player_possible_nodes.children[other_cards])
                player_values_sum += self._get_exploitability(
                    player_position,
                    best_response_node.children[cards],
                    next_player_possible_nodes,
                    new_player_cards,
                    board_cards)
            return player_values_sum / len(best_response_node.children)
        elif isinstance(best_response_node, BoardCardsNode):
            player_values_sum = 0
            for cards in best_response_node.children:
                new_board_cards = flatten(board_cards, cards)
                new_player_possible_nodes = filter(lambda node: cards in node.children, player_possible_nodes)
                new_player_possible_nodes = map(lambda node: node.children[cards], new_player_possible_nodes)
                player_values_sum += self._get_exploitability(
                    player_position,
                    best_response_node.children[cards],
                    list(new_player_possible_nodes),
                    best_response_cards,
                    new_board_cards)
            return player_values_sum / len(best_response_node.children)
        elif best_response_node.player == player_position:
            strategy_sum = np.zeros(3)
            for player_node in player_possible_nodes:
                strategy_sum += player_node.strategy
            player_node_strategy = strategy_sum / len(player_possible_nodes)

            values_sum = 0
            for a in best_response_node.children:
                player_value = self._get_exploitability(
                    player_position,
                    best_response_node.children[a],
                    list(map(lambda node: node.children[a], player_possible_nodes)),
                    best_response_cards,
                    board_cards)
                values_sum += player_value * player_node_strategy[a]
            return values_sum
        else:
            best_value = None
            best_value_action = None
            for a in best_response_node.children:
                player_value = self._get_exploitability(
                    player_position,
                    best_response_node.children[a],
                    list(map(lambda node: node.children[a], player_possible_nodes)),
                    best_response_cards,
                    board_cards)
                if (best_value is None) or (player_value < best_value):
                    best_value = player_value
                    best_value_action = a
            best_response_node.strategy[best_value_action] = 1
            return best_value
