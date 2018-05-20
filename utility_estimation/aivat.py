import numpy as np

from tools.agent_utils import convert_action_to_int
from tools.game_tree.nodes import BoardCardsNode, ActionNode, TerminalNode
from tools.tree_utils import get_parent_action
from tools.hand_evaluation import get_utility
from tools.io_util import read_strategy_from_file
from tools.utils import is_unique, flatten
from utility_estimation.utils import get_all_board_cards, get_board_cards
from evaluation.player_utility import PlayerUtility


class AivatUtilityEstimator():
    def __init__(self, game, mucking_enabled, **args):
        if game.get_num_players() != 2:
            raise AttributeError(
                'Only games with 2 players are supported')

        self.game = game
        self.mucking_enabled = mucking_enabled

        if mucking_enabled:
            raise AttributeError('Mucking not yet supported')

        if 'equilibirum_strategy_path' not in args:
            raise AttributeError(
                '"equilibirum_strategy_path" argument not provided')

        equilibirum_strategy_path = args['equilibirum_strategy_path']
        equilibirum_strategy, _ = read_strategy_from_file(game, equilibirum_strategy_path)
        utilities_dict = {}

        def callback(nodes, utilities):
            nonlocal utilities_dict

            key = ';'.join(map(lambda m: str(m), nodes))
            if key not in utilities_dict:
                utilities_dict[key] = utilities

        PlayerUtility(game).get_player_utilities(
            [equilibirum_strategy] * 2,
            [],
            [],
            [False] * 2,
            callback)
        self.utilities_dict = utilities_dict

    def get_utility_estimations(self, state, player, sampling_strategy, evaluated_strategies=None):
        if evaluated_strategies is None:
            evaluated_strategies = [sampling_strategy]

        num_players = self.game.get_num_players()
        opponent_player = (player + 1) % 2

        num_evaluated_strategies = len(evaluated_strategies)

        utilities = np.zeros(num_evaluated_strategies)

        any_player_folded = False
        for p in range(num_players):
            any_player_folded = any_player_folded or state.get_player_folded(p)

        all_board_cards = get_all_board_cards(self.game, state)

        opponent_hole_cards = None
        possible_player_hole_cards = None
        if any_player_folded and self.mucking_enabled:
            possible_player_hole_cards = list(filter(
                lambda hole_cards: is_unique(hole_cards, all_board_cards),
                sampling_strategy.children))
        else:
            opponent_hole_cards = [state.get_hole_card(opponent_player, c) for c in range(self.game.get_num_hole_cards())]
            possible_player_hole_cards = list(filter(
                lambda hole_cards: is_unique(hole_cards, opponent_hole_cards, all_board_cards),
                sampling_strategy.children))
        nodes = [
            [
                expert_node.children[hole_cards]
                for hole_cards in possible_player_hole_cards]
            for expert_node in evaluated_strategies]
        sampling_strategy_nodes = [sampling_strategy.children[hole_cards] for hole_cards in possible_player_hole_cards]

        opponent_node = sampling_strategy.children[tuple(sorted(opponent_hole_cards))]

        num_nodes = len(possible_player_hole_cards)
        evaluated_strategies_reach_probabilities = np.ones([num_evaluated_strategies, num_nodes])
        sampling_strategy_reach_probabilities = np.ones(num_nodes)

        # Calculate correction term for hole cards
        histories_actions_utilities = {}
        for i in range(num_nodes):
            history_actions_utilities = {}
            histories_actions_utilities[i] = history_actions_utilities
            for a in filter(lambda a: a != possible_player_hole_cards[i], sampling_strategy.children):
                nodes_tmp = [None, None]
                nodes_tmp[player] = sampling_strategy_nodes[i]
                nodes_tmp[opponent_player] = sampling_strategy.children[a]
                key = ';'.join(map(lambda m: str(m), nodes_tmp))
                history_actions_utilities[a] = self.utilities_dict[key][player]

        history_sampling_strategy_reach_probabilities_sum = np.sum(sampling_strategy_reach_probabilities)
        current_history_expected_value = 0
        for i in range(num_nodes):
            for a in filter(lambda a: a != possible_player_hole_cards[i], sampling_strategy.children):
                current_history_expected_value += histories_actions_utilities[i][a] \
                    * sampling_strategy_reach_probabilities[i] / num_nodes

        for i in range(num_nodes):
            sampling_strategy_reach_probabilities[i] /= num_nodes
            for j in range(num_evaluated_strategies):
                evaluated_strategies_reach_probabilities[j, i] /= num_nodes

        next_history_sampling_strategy_reach_probabilities_sum = np.sum(sampling_strategy_reach_probabilities)
        next_history_expected_value = 0
        for i in range(num_nodes):
            next_history_expected_value += histories_actions_utilities[i][tuple(sorted(opponent_hole_cards))] \
                * sampling_strategy_reach_probabilities[i]
        utilities += \
            (current_history_expected_value / history_sampling_strategy_reach_probabilities_sum) \
            - (next_history_expected_value / next_history_sampling_strategy_reach_probabilities_sum)

        def add_terminals_to_utilities(pot_commitment, players_folded, sampling_strategy_reach_probabilities, evaluated_strategies_reach_probabilities):
            nonlocal utilities
            nonlocal player

            sampling_strategy_reach_probability_sum = np.sum(sampling_strategy_reach_probabilities)
            if sampling_strategy_reach_probability_sum == 0:
                return

            for i in range(num_nodes):
                utility = None
                if players_folded[player]:
                    utility = -pot_commitment[player]
                else:
                    hole_cards = [possible_player_hole_cards[i] if p == player else opponent_hole_cards for p in range(num_players)]
                    utility = get_utility(
                        hole_cards,
                        all_board_cards,
                        players_folded,
                        pot_commitment)[player]
                utilities += utility * (evaluated_strategies_reach_probabilities[:, i] / sampling_strategy_reach_probability_sum)

        def update_reach_proabilities(action, sampling_strategy_nodes, nodes, sampling_strategy_reach_probabilities, evaluated_strategies_reach_probabilities):
            for i in range(num_nodes):
                sampling_strategy_reach_probabilities[i] *= sampling_strategy_nodes[i].strategy[action]
                for j in range(num_evaluated_strategies):
                    evaluated_strategies_reach_probabilities[j, i] *= nodes[j][i].strategy[action]

        round_index = 0
        action_index = 0
        while True:
            node = nodes[0][0]
            if isinstance(node, BoardCardsNode):
                new_board_cards = get_board_cards(self.game, state, round_index)

                num_board_cards = len(opponent_node.children)

                # Calculate correction term for board cards
                histories_actions_utilities = {}
                for i in range(num_nodes):
                    history_actions_utilities = {}
                    histories_actions_utilities[i] = history_actions_utilities
                    for a in filter(lambda a: a in sampling_strategy_nodes[i].children, opponent_node.children):
                        nodes_tmp = [None, None]
                        nodes_tmp[player] = sampling_strategy_nodes[i].children[a]
                        nodes_tmp[opponent_player] = opponent_node.children[a]
                        key = ';'.join(map(lambda m: str(m), nodes_tmp))
                        history_actions_utilities[a] = self.utilities_dict[key][player]

                history_sampling_strategy_reach_probabilities_sum = np.sum(sampling_strategy_reach_probabilities)
                importance_sampling_ratio = np.sum(evaluated_strategies_reach_probabilities, axis=1) / history_sampling_strategy_reach_probabilities_sum

                current_history_expected_value = 0
                for i in range(num_nodes):
                    for a in filter(lambda a: a in sampling_strategy_nodes[i].children, opponent_node.children):
                        current_history_expected_value += histories_actions_utilities[i][a] \
                            * sampling_strategy_reach_probabilities[i] / num_board_cards

                for i in range(num_nodes):
                    sampling_strategy_reach_probabilities[i] /= num_board_cards
                    for j in range(num_evaluated_strategies):
                        evaluated_strategies_reach_probabilities[j, i] /= num_board_cards

                next_history_sampling_strategy_reach_probabilities_sum = np.sum(sampling_strategy_reach_probabilities)
                next_history_expected_value = 0
                for i in range(num_nodes):
                    next_history_expected_value += histories_actions_utilities[i][new_board_cards] \
                        * sampling_strategy_reach_probabilities[i]
                utilities += \
                    ((current_history_expected_value / history_sampling_strategy_reach_probabilities_sum) \
                    - (next_history_expected_value / next_history_sampling_strategy_reach_probabilities_sum)) * importance_sampling_ratio


                nodes = [[expert_node.children[new_board_cards] for expert_node in expert_nodes] for expert_nodes in nodes]
                sampling_strategy_nodes = [node.children[new_board_cards] for node in sampling_strategy_nodes]
                opponent_node = opponent_node.children[new_board_cards]
            elif isinstance(node, ActionNode):
                action = convert_action_to_int(state.get_action_type(round_index, action_index))
                if node.player == player:
                    # Calculate correction term for player actions
                    histories_actions_utilities = {}
                    for i in range(num_nodes):
                        history_actions_utilities = {}
                        histories_actions_utilities[i] = history_actions_utilities
                        for a in node.children:
                            nodes_tmp = [None, None]
                            nodes_tmp[player] = sampling_strategy_nodes[i].children[a]
                            nodes_tmp[opponent_player] = opponent_node.children[a]
                            key = ';'.join(map(lambda m: str(m), nodes_tmp))
                            history_actions_utilities[a] = self.utilities_dict[key][player]

                    history_sampling_strategy_reach_probabilities_sum = np.sum(sampling_strategy_reach_probabilities)
                    importance_sampling_ratio = np.sum(evaluated_strategies_reach_probabilities, axis=1) / history_sampling_strategy_reach_probabilities_sum

                    current_history_expected_value = 0
                    for i in range(num_nodes):
                        for a in sampling_strategy_nodes[i].children:
                            current_history_expected_value += histories_actions_utilities[i][a] \
                                * sampling_strategy_reach_probabilities[i] * sampling_strategy_nodes[i].strategy[a]

                    update_reach_proabilities(
                        action,
                        sampling_strategy_nodes,
                        nodes,
                        sampling_strategy_reach_probabilities,
                        evaluated_strategies_reach_probabilities)

                    next_history_sampling_strategy_reach_probabilities_sum = np.sum(sampling_strategy_reach_probabilities)
                    next_history_expected_value = 0
                    for i in range(num_nodes):
                        next_history_expected_value += histories_actions_utilities[i][action] \
                            * sampling_strategy_reach_probabilities[i]
                    utilities += \
                        ((current_history_expected_value / history_sampling_strategy_reach_probabilities_sum) \
                        - (next_history_expected_value / next_history_sampling_strategy_reach_probabilities_sum)) * importance_sampling_ratio

                action_index += 1
                if action_index == state.get_num_actions(round_index):
                    round_index += 1
                    action_index = 0
                nodes = [[expert_node.children[action] for expert_node in expert_nodes] for expert_nodes in nodes]
                sampling_strategy_nodes = [node.children[action] for node in sampling_strategy_nodes]
                opponent_node = opponent_node.children[action]
            elif isinstance(node, TerminalNode):
                players_folded = [state.get_player_folded(p) for p in range(num_players)]
                add_terminals_to_utilities(node.pot_commitment, players_folded, sampling_strategy_reach_probabilities, evaluated_strategies_reach_probabilities)
                break

        return utilities
