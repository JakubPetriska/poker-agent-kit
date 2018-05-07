import numpy as np

from tools.agent_utils import convert_action_to_int
from tools.game_tree.nodes import BoardCardsNode, ActionNode, TerminalNode
from tools.tree_utils import get_parent_action
from tools.hand_evaluation import get_utility
from tools.utils import is_unique, flatten


class ImaginaryObservationsUtilityEstimator():
    def __init__(self, game, portfolio_strategies):
        if game.get_num_players() != 2:
            raise AttributeError(
                'Only games with 2 players are supported')

        self.game = game
        self.portfolio_strategies = portfolio_strategies
        self.num_experts = len(portfolio_strategies)

    def _get_all_board_cards(self, state):
        total_num_board_cards = self.game.get_total_num_board_cards(state.get_round())
        return [state.get_board_card(c) for c in range(0, total_num_board_cards)]

    def _get_board_cards(self, state, round_index):
        total_num_board_cards = self.game.get_total_num_board_cards(round_index)
        round_num_board_cards = self.game.get_num_board_cards(round_index)
        start_board_card_index = total_num_board_cards - round_num_board_cards
        board_cards = [state.get_board_card(c) for c in range(start_board_card_index, total_num_board_cards)]
        return tuple(sorted(board_cards))

    def get_expert_utility_estimations(self, match_state, expert_probabilities):
        num_players = self.game.get_num_players()
        player = match_state.get_viewing_player()
        opponent_player = (player + 1) % 2
        state = match_state.get_state()

        expert_utilities = np.zeros(self.num_experts)

        any_player_folded = False
        for p in range(num_players):
            any_player_folded = any_player_folded or state.get_player_folded(p)
        all_board_cards = self._get_all_board_cards(state)

        hole_cards_node = self.portfolio_strategies[0]
        opponent_hole_cards = None
        possible_player_hole_cards = None
        if any_player_folded:
            possible_player_hole_cards = list(filter(
                lambda hole_cards: is_unique(hole_cards, all_board_cards),
                hole_cards_node.children))
        else:
            opponent_hole_cards = [state.get_hole_card(opponent_player, c) for c in range(self.game.get_num_hole_cards())]
            possible_player_hole_cards = list(filter(
                lambda hole_cards: is_unique(hole_cards, opponent_hole_cards, all_board_cards),
                hole_cards_node.children))
        nodes = [[expert_node.children[hole_cards] for hole_cards in possible_player_hole_cards] for expert_node in self.portfolio_strategies]
        num_nodes = len(possible_player_hole_cards)
        expert_reach_probabilities = np.ones([self.num_experts, num_nodes])
        player_reach_probabilities = np.ones(num_nodes)

        def add_terminals_to_utilities(nodes, players_folded, expert_reach_probabilities, player_reach_probabilities):
            nonlocal expert_utilities
            nonlocal player
            player_reach_probability = np.sum(player_reach_probabilities)
            for i in range(num_nodes):
                utility = None
                if players_folded[player]:
                    utility = -nodes[0][0].pot_commitment[player]
                else:
                    hole_cards = [possible_player_hole_cards[i] if p == player else opponent_hole_cards for p in range(num_players)]
                    utility = get_utility(
                        hole_cards,
                        all_board_cards,
                        players_folded,
                        nodes[0][0].pot_commitment)[match_state.get_viewing_player()]
                expert_utilities += utility * (expert_reach_probabilities[:, i] / player_reach_probability)

        def update_reach_proabilities(action, nodes, expert_reach_probabilities, player_reach_probabilities):
            for i in range(num_nodes):
                player_reach_prob_multiplier = 0
                for j in range(self.num_experts):
                    expert_action_probability = nodes[j][i].strategy[action]
                    expert_reach_probabilities[j][i] *= expert_action_probability
                    player_reach_prob_multiplier += expert_action_probability * expert_probabilities[j]
                if player_reach_probabilities is not None:
                    player_reach_probabilities[i] *= player_reach_prob_multiplier

        round_index = 0
        action_index = 0
        while True:
            node = nodes[0][0]
            if isinstance(node, BoardCardsNode):
                new_board_cards = self._get_board_cards(state, round_index)
                nodes = [[expert_node.children[new_board_cards] for expert_node in expert_nodes] for expert_nodes in nodes]
            elif isinstance(node, ActionNode):
                action = convert_action_to_int(state.get_action_type(round_index, action_index))
                if node.player == player:
                    for other_action in node.children:
                        if other_action != action and isinstance(node.children[other_action], TerminalNode) \
                                and (other_action == 0 or not any_player_folded):
                            expert_reach_probabilities_copy = np.copy(expert_reach_probabilities)
                            update_reach_proabilities(0, nodes, expert_reach_probabilities_copy, None)
                            players_folded = [True if p == player and other_action == 0 else False for p in range(2)]
                            new_nodes = [[expert_node.children[other_action] for expert_node in expert_nodes] for expert_nodes in nodes]
                            add_terminals_to_utilities(new_nodes, players_folded, expert_reach_probabilities_copy, player_reach_probabilities)
                    else:
                        update_reach_proabilities(action, nodes, expert_reach_probabilities, player_reach_probabilities)

                action_index += 1
                if action_index == state.get_num_actions(round_index):
                    round_index += 1
                    action_index = 0
                nodes = [[expert_node.children[action] for expert_node in expert_nodes] for expert_nodes in nodes]
            elif isinstance(node, TerminalNode):
                players_folded = [state.get_player_folded(p) for p in range(num_players)]
                add_terminals_to_utilities(nodes, players_folded, expert_reach_probabilities, player_reach_probabilities)
                break

        return expert_utilities
