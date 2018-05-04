import sys
import numpy as np

import acpc_python_client as acpc

from tools.constants import NUM_ACTIONS
from tools.io_util import read_strategy_from_file
from tools.agent_utils import get_info_set, select_action, convert_action_to_int
from implicit_modelling.exp3g import Exp3G
from tools.game_tree.nodes import HoleCardsNode, BoardCardsNode, ActionNode
from tools.tree_utils import get_parent_action
from tools.hand_evaluation import get_utility


class ImplicitModellingAgent(acpc.Agent):
    def __init__(
            self,
            game_file_path,
            portfolio_strategy_files_paths,
            exp3g_gamma=0.02,
            exp3g_eta=0.025):
        super().__init__()
        self.portfolio_size = len(portfolio_strategy_files_paths)
        self.bandit_algorithm = Exp3G(exp3g_gamma, exp3g_eta, self.portfolio_size)

        self.portfolio_trees = []
        self.portfolio_dicts = []
        for portfolio_strategy_file_path in portfolio_strategy_files_paths:
            strategy_tree, strategy_dict = read_strategy_from_file(game_file_path, portfolio_strategy_file_path)
            self.portfolio_trees += [strategy_tree]
            self.portfolio_dicts += [strategy_dict]

    def on_game_start(self, game):
        pass

    def _get_info_set_current_strategy(self, info_set):
        experts_weights = self.bandit_algorithm.get_current_expert_probabilities()
        current_strategy = np.zeros(NUM_ACTIONS)
        for i in range(self.portfolio_size):
            current_strategy += experts_weights[i] * np.array(self.portfolio_dicts[i][info_set])
        return current_strategy

    def on_next_turn(self, game, match_state, is_acting_player):
        if not is_acting_player:
            return

        info_set = get_info_set(game, match_state)
        current_strategy = self._get_info_set_current_strategy(info_set)
        selected_action = select_action(current_strategy)
        self.set_next_action(selected_action)

    def _get_reach_probabilities(self, game, match_state):
        expert_reach_probabilities = np.ones(self.portfolio_size)
        player_reach_probability = 1

        state = match_state.get_state()
        player = match_state.get_viewing_player()
        expert_probabilities = self.bandit_algorithm.get_current_expert_probabilities()

        nodes = self.portfolio_trees
        round_index = 0
        action_index = 0
        while True:
            node = nodes[0]
            child_key = None
            if isinstance(node, HoleCardsNode):
                hole_cards = [state.get_hole_card(player, c) for c in range(game.get_num_hole_cards())]
                child_key = tuple(sorted(hole_cards))
            elif isinstance(node, BoardCardsNode):
                total_num_board_cards = game.get_total_num_board_cards(round_index)
                round_num_board_cards = game.get_num_board_cards(round_index)
                start_board_card_index = total_num_board_cards - round_num_board_cards
                board_cards = [state.get_board_card(c) for c in range(start_board_card_index, total_num_board_cards)]
                child_key = tuple(sorted(board_cards))
            else:
                action = convert_action_to_int(state.get_action_type(round_index, action_index))
                if node.player == player:
                    player_reach_prob_multiplier = 0
                    for i in range(self.portfolio_size):
                        expert_action_probability = nodes[i].strategy[action]
                        expert_reach_probabilities[i] *= expert_action_probability
                        player_reach_prob_multiplier += expert_action_probability * expert_probabilities[i]
                    player_reach_probability *= player_reach_prob_multiplier
                child_key = action

                action_index += 1
                if action_index == state.get_num_actions(round_index):
                    round_index += 1
                    action_index = 0
                if round_index == game.get_num_rounds():
                    break
            nodes = [node.children[child_key] for node in nodes]

        return player_reach_probability, expert_reach_probabilities

    def _get_match_state_player_utility(self, game, match_state):
        num_players = game.get_num_players()
        state = match_state.get_state()
        hole_cards = [
            [state.get_hole_card(p, c) for c in range(game.get_num_hole_cards())]
            for p in range(num_players)]

        board_cards = [state.get_board_card(c) for c in range(game.get_total_num_board_cards(game.get_num_rounds() - 1))]
        players_folded = [state.get_player_folded(p) for p in range(num_players)]
        pot_commitment = [state.get_spent(p) for p in range(num_players)]
        return get_utility(hole_cards, board_cards, players_folded, pot_commitment)[match_state.get_viewing_player()]

    def on_game_finished(self, game, match_state):
        player_reach_probability, expert_reach_probabilities = self._get_reach_probabilities(game, match_state)
        utility = self._get_match_state_player_utility(game, match_state)
        self.bandit_algorithm.update_weights(utility * (expert_reach_probabilities / player_reach_probability))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage {game_file_path} {dealer_hostname} {dealer_port} *{portfolio_strategy_files_paths}")
        sys.exit(1)

    client = acpc.Client(sys.argv[1], sys.argv[2], sys.argv[3])
    client.play(ImplicitModellingAgent(sys.argv[1], sys.argv[4:]))
