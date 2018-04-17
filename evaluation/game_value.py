import itertools
import numpy as np

from tools.game_tree.nodes import HoleCardsNode, TerminalNode, StrategyActionNode, BoardCardsNode
from tools.utils import flatten, is_unique, intersection
from tools.hand_evaluation import get_utility


class GameValue:
    def __init__(self, game):
        self.game = game

    def evaluate(self, *args):
        num_players = self.game.get_num_players()
        if len(args) != num_players:
            raise AttributeError(
                'Only %s strategies provided for game with %s players' % (len(args), num_players))
        game_values_sum = np.zeros(num_players)
        player_positions_count = 0
        for player_positions in itertools.permutations(range(num_players)):
            nodes = [args[pos] for pos in player_positions]
            game_values = self._get_game_values(nodes, [], [], [False] * num_players)
            for i in range(num_players):
                game_values_sum[i] += game_values[player_positions[i]]
            player_positions_count += 1
        return (game_values_sum / player_positions_count).tolist()

    def _get_game_values(self, nodes, hole_cards, board_cards, players_folded):
        node = nodes[0]
        if isinstance(node, TerminalNode):
            return get_utility(hole_cards, board_cards, players_folded, node.pot_commitment)
        elif isinstance(node, HoleCardsNode):
            values = np.zeros([0, self.game.get_num_players()])

            hole_cards = [node.children for node in nodes]
            for hole_cards_combination in itertools.product(*hole_cards):
                if is_unique(*hole_cards_combination):
                    new_nodes = [
                        node.children[hole_cards_combination[i]]
                        for i, node in enumerate(nodes)]
                    player_values = self._get_game_values(new_nodes, hole_cards_combination, board_cards, players_folded)
                    values = np.append(values, [player_values], 0)
            return np.mean(values, 0)
        elif isinstance(node, BoardCardsNode):
            values = np.zeros([len(node.children), self.game.get_num_players()])
            possible_board_cards = intersection(*map(lambda node: node.children, nodes))
            for i, next_board_cards in enumerate(possible_board_cards):
                new_nodes = [node.children[next_board_cards] for node in nodes]
                new_board_cards = flatten(board_cards, next_board_cards)
                values[i, :] = self._get_game_values(
                    new_nodes,
                    hole_cards,
                    new_board_cards,
                    players_folded)
            return np.mean(values, 0)
        else:
            current_player_node = nodes[node.player]
            utilities = np.zeros(self.game.get_num_players())
            for a in current_player_node.children:
                new_nodes = [node.children[a] for node in nodes]
                new_players_folded = players_folded
                if a == 0:
                    new_players_folded = list(players_folded)
                    new_players_folded[current_player_node.player] = True
                action_utilities = self._get_game_values(new_nodes, hole_cards, board_cards, new_players_folded)
                utilities += action_utilities * current_player_node.strategy[a]
            return utilities
