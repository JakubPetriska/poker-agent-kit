import numpy as np

from tools.agent_utils import convert_action_to_int
from tools.game_tree.nodes import BoardCardsNode, ActionNode, TerminalNode
from tools.tree_utils import get_parent_action
from tools.hand_evaluation import get_utility
from utility_estimation.utils import get_all_board_cards, get_board_cards


class SimpleUtilityEstimator():
    def __init__(self, game, mucking_enabled):
        if game.get_num_players() != 2:
            raise AttributeError(
                'Only games with 2 players are supported')

        self.game = game

    def get_utility_estimations(self, state, player, sampling_strategy, evaluated_strategies=None):
        if evaluated_strategies is None:
            evaluated_strategies = [sampling_strategy]

        num_players = self.game.get_num_players()
        opponent_player = (player + 1) % 2

        num_evaluated_strategies = len(evaluated_strategies)

        player_hole_cards = tuple(sorted([state.get_hole_card(player, c) for c in range(self.game.get_num_hole_cards())]))

        any_player_folded = False
        for p in range(num_players):
            any_player_folded = any_player_folded or state.get_player_folded(p)

        all_board_cards = get_all_board_cards(self.game, state)

        evaluated_strategies_nodes = [node.children[player_hole_cards] for node in evaluated_strategies]
        sampling_strategy_node = sampling_strategy.children[player_hole_cards]

        evaluated_strategies_reach_probabilities = np.ones(num_evaluated_strategies)
        sampling_strategy_reach_probability = 1

        round_index = 0
        action_index = 0
        while True:
            node = evaluated_strategies_nodes[0]
            if isinstance(node, BoardCardsNode):
                new_board_cards = get_board_cards(self.game, state, round_index)
                evaluated_strategies_nodes = [node.children[new_board_cards] for node in evaluated_strategies_nodes]
                sampling_strategy_node = sampling_strategy_node.children[new_board_cards]
            elif isinstance(node, ActionNode):
                action = convert_action_to_int(state.get_action_type(round_index, action_index))
                if node.player == player:
                    sampling_strategy_reach_probability *= sampling_strategy_node.strategy[action]
                    for i in range(num_evaluated_strategies):
                        evaluated_strategies_reach_probabilities[i] *= evaluated_strategies_nodes[i].strategy[action]

                action_index += 1
                if action_index == state.get_num_actions(round_index):
                    round_index += 1
                    action_index = 0
                evaluated_strategies_nodes = [node.children[action] for node in evaluated_strategies_nodes]
                sampling_strategy_node = sampling_strategy_node.children[action]
            elif isinstance(node, TerminalNode):
                players_folded = [state.get_player_folded(p) for p in range(num_players)]
                if players_folded[player]:
                    utility = -node.pot_commitment[player]
                else:
                    opponent_hole_cards = [state.get_hole_card(opponent_player, c) for c in range(self.game.get_num_hole_cards())]
                    hole_cards = [player_hole_cards if p == player else opponent_hole_cards for p in range(num_players)]
                    utility = get_utility(
                        hole_cards,
                        all_board_cards,
                        players_folded,
                        node.pot_commitment)[player]
                return utility * (evaluated_strategies_reach_probabilities / sampling_strategy_reach_probability)
