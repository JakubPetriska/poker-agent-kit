import numpy as np

from tools.agent_utils import convert_action_to_int
from tools.game_tree.nodes import HoleCardsNode, BoardCardsNode, ActionNode
from tools.tree_utils import get_parent_action
from tools.hand_evaluation import get_utility


class ImaginaryObservationsUtilityEstimator():
    def __init__(self, game, portfolio_strategies):
        self.game = game
        self.portfolio_strategies = portfolio_strategies
        self.num_experts = len(portfolio_strategies)

    def _get_reach_probabilities(self, match_state, expert_probabilities):
        expert_reach_probabilities = np.ones(self.num_experts)
        player_reach_probability = 1

        state = match_state.get_state()
        player = match_state.get_viewing_player()

        nodes = self.portfolio_strategies
        round_index = 0
        action_index = 0
        while True:
            node = nodes[0]
            child_key = None
            if isinstance(node, HoleCardsNode):
                hole_cards = [state.get_hole_card(player, c) for c in range(self.game.get_num_hole_cards())]
                child_key = tuple(sorted(hole_cards))
            elif isinstance(node, BoardCardsNode):
                total_num_board_cards = self.game.get_total_num_board_cards(round_index)
                round_num_board_cards = self.game.get_num_board_cards(round_index)
                start_board_card_index = total_num_board_cards - round_num_board_cards
                board_cards = [state.get_board_card(c) for c in range(start_board_card_index, total_num_board_cards)]
                child_key = tuple(sorted(board_cards))
            else:
                action = convert_action_to_int(state.get_action_type(round_index, action_index))
                if node.player == player:
                    player_reach_prob_multiplier = 0
                    for i in range(self.num_experts):
                        expert_action_probability = nodes[i].strategy[action]
                        expert_reach_probabilities[i] *= expert_action_probability
                        player_reach_prob_multiplier += expert_action_probability * expert_probabilities[i]
                    player_reach_probability *= player_reach_prob_multiplier
                child_key = action

                action_index += 1
                if action_index == state.get_num_actions(round_index):
                    round_index += 1
                    action_index = 0
                if round_index == self.game.get_num_rounds():
                    break
            nodes = [node.children[child_key] for node in nodes]

        return player_reach_probability, expert_reach_probabilities

    def _get_match_state_player_utility(self, match_state):
        num_players = self.game.get_num_players()
        state = match_state.get_state()
        hole_cards = [
            [state.get_hole_card(p, c) for c in range(self.game.get_num_hole_cards())]
            for p in range(num_players)]

        board_cards = [state.get_board_card(c) for c in range(self.game.get_total_num_board_cards(self.game.get_num_rounds() - 1))]
        players_folded = [state.get_player_folded(p) for p in range(num_players)]
        pot_commitment = [state.get_spent(p) for p in range(num_players)]
        return get_utility(hole_cards, board_cards, players_folded, pot_commitment)[match_state.get_viewing_player()]

    def get_expert_utility_estimations(self, match_state, expert_probabilities):
        player_reach_probability, expert_reach_probabilities = self._get_reach_probabilities(match_state, expert_probabilities)
        utility = self._get_match_state_player_utility(match_state)
        return utility * (expert_reach_probabilities / player_reach_probability)
