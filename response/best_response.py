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
            self._get_exploitability(position, best_response, np.array([[strategy, 1, ()]]), [], [])

        return best_response

    def _get_exploitability(
            self,
            player_position,
            best_response_node,
            player_states,
            best_response_cards,
            board_cards):

        if isinstance(best_response_node, TerminalNode):
            parent_action = list(filter(lambda item: item[1] == best_response_node, best_response_node.parent.children.items()))[0][0]
            if parent_action == 0:
                player_folded = best_response_node.parent.player
                pot_amount = np.sum(best_response_node.pot_commitment)
                return -best_response_node.pot_commitment[player_position] + \
                    (pot_amount if player_position != player_folded else 0)

            player_value_sum = 0
            for state in player_states:
                hands = [state[2], best_response_cards]
                winners = get_winners(hands)
                winner_count = len(winners)
                pot_amount = np.sum(best_response_node.pot_commitment)
                per_winner_value = pot_amount / winner_count
                player_value = -best_response_node.pot_commitment[player_position] + \
                    (per_winner_value if 0 in winners else 0)
                player_value_sum += player_value * state[1]
            return player_value_sum


        elif isinstance(best_response_node, HoleCardsNode):
            player_values_sum = 0
            for cards in best_response_node.children:
                new_bets_response_cards = flatten(best_response_cards, cards)
                new_player_states = np.empty([0, 3])
                for other_cards in best_response_node.children:
                    if len(intersection(cards, other_cards)) == 0 and len(intersection(cards, board_cards)) == 0:
                        for state in player_states:
                            new_player_states = np.append(
                                new_player_states,
                                [[state[0].children[other_cards], -1, other_cards]],
                                axis=0)
                new_player_states[:, 1] = 1 / len(new_player_states)

                player_values_sum += self._get_exploitability(
                    player_position,
                    best_response_node.children[cards],
                    new_player_states,
                    new_bets_response_cards,
                    board_cards)
            return player_values_sum / len(best_response_node.children)

        elif isinstance(best_response_node, BoardCardsNode):
            player_values_sum = 0
            for cards in best_response_node.children:
                new_board_cards = flatten(board_cards, cards)

                new_player_states = np.empty([0, 3])
                for state in player_states:
                    if cards in state[0].children:
                        new_player_states = np.append(
                            new_player_states,
                            [[state[0].children[cards], state[1], state[2]]],
                            axis=0)

                new_player_states[:, 1] = 1 / len(new_player_states)

                player_values_sum += self._get_exploitability(
                    player_position,
                    best_response_node.children[cards],
                    new_player_states,
                    best_response_cards,
                    new_board_cards)
            return player_values_sum / len(best_response_node.children)

        elif best_response_node.player == player_position:
            player_node_strategy = np.zeros(3)
            for state in player_states:
                player_node_strategy += np.array(state[0].strategy) * state[1]

            values_sum = 0
            for a in best_response_node.children:
                new_player_states = np.empty([0, 3])
                for state in player_states:
                    new_player_states = np.append(
                        new_player_states,
                        [[state[0].children[a], state[1] * player_node_strategy[a], state[2]]],
                        axis=0)

                player_value = self._get_exploitability(
                    player_position,
                    best_response_node.children[a],
                    new_player_states,
                    best_response_cards,
                    board_cards)
                values_sum += player_value * player_node_strategy[a]
            return values_sum

        else:
            best_value = None
            best_value_actions = None
            for a in best_response_node.children:
                new_player_states = np.empty([0, 3])
                for state in player_states:
                    new_player_states = np.append(
                        new_player_states,
                        [[state[0].children[a], state[1], state[2]]],
                        axis=0)

                player_value = self._get_exploitability(
                    player_position,
                    best_response_node.children[a],
                    new_player_states,
                    best_response_cards,
                    board_cards)

                if (best_value is None) or (player_value < best_value):
                    best_value = player_value
                    best_value_actions = [a]
                elif player_value == best_value:
                    best_value_actions.append(a)
            best_value_action_probability = 1 / len(best_value_actions)
            for a in best_value_actions:
                best_response_node.strategy[a] = best_value_action_probability
            return best_value
