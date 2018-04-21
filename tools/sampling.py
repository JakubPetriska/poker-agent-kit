import acpc_python_client as acpc

from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.nodes import ActionNode
from tools.constants import NUM_ACTIONS
from tools.game_tree.node_provider import NodeProvider
from tools.game_tree.nodes import ActionNode, HoleCardsNode, BoardCardsNode


def read_log_file(
    game_file_path,
    log_file_path,
    player_names,
    player_trees=None):

    game = acpc.read_game_file(game_file_path)
    num_players = game.get_num_players()
    if len(player_names) != num_players:
        raise AttributeError('Wrong number of player names provided')
    if game.get_betting_type() != acpc.BettingType.LIMIT:
        raise AttributeError('Only limit betting games are supported')

    players = {}
    for i in range(num_players):
        player_name = player_names[i]
        player_tree = None
        if player_trees and player_name in player_trees:
            player_tree = player_trees[player_name]
        else:
            player_tree = GameTreeBuilder(game, SamplesTreeNodeProvider()).build_tree()
        players[player_name] = player_tree

    with open(log_file_path, 'r') as strategy_file:
        for line in strategy_file:
            if not line.strip() or line.strip().startswith('#'):
                continue
            player_names = [name.strip() for name in line.split(':')[-1].split('|')]
            state = acpc.parse_state(game_file_path, line)

            current_player_trees = [players[name] for name in player_names]
            _add_state_to_sample_trees(game, state, current_player_trees, 0, 0)

    return players


def _action_type_to_int(action_type):
    if action_type == acpc.ActionType.FOLD:
        return 0
    elif action_type == acpc.ActionType.CALL:
        return 1
    else:
        return 2


def _add_state_to_sample_trees(
    game,
    state,
    player_nodes,
    round_index,
    action_index):

    tmp_node = player_nodes[0]
    if isinstance(tmp_node, HoleCardsNode):
        num_hole_cards = game.get_num_hole_cards()

        new_player_nodes = []
        for p, node in enumerate(player_nodes):
            player_hole_cards = tuple(sorted([
                state.get_hole_card(p, c)
                for c in range(num_hole_cards)]))
            new_player_nodes.append(node.children[player_hole_cards])

        _add_state_to_sample_trees(
            game,
            state,
            new_player_nodes,
            round_index,
            action_index)

    elif isinstance(tmp_node, BoardCardsNode):
        num_board_cards = game.get_num_board_cards(round_index)
        total_num_board_cards = game.get_total_num_board_cards(round_index)
        board_cards = tuple(sorted([
            state.get_board_card(i)
            for i in range(total_num_board_cards - num_board_cards, total_num_board_cards)]))

        new_player_nodes = [node.children[board_cards] for node in player_nodes]
        _add_state_to_sample_trees(
            game,
            state,
            new_player_nodes,
            round_index,
            action_index)

    elif isinstance(tmp_node, ActionNode):
        player_node = player_nodes[tmp_node.player]
        player_action = _action_type_to_int(
            state.get_action_type(round_index, action_index))
        player_node.action_decision_counts[player_action] += 1

        new_player_nodes = [node.children[player_action] for node in player_nodes]
        new_round_index = round_index
        new_action_index = action_index + 1
        if new_action_index >= state.get_num_actions(round_index):
            new_round_index += 1
            new_action_index = 0
        _add_state_to_sample_trees(
            game,
            state,
            new_player_nodes,
            new_round_index,
            new_action_index)


class SamplesActionNode(ActionNode):
    def __init__(self, parent, player):
        super().__init__(parent, player)
        self.action_decision_counts = [0] * NUM_ACTIONS


class SamplesTreeNodeProvider(NodeProvider):
    def create_action_node(self, parent, player):
        return SamplesActionNode(parent, player)
