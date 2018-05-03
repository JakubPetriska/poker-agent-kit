import random

import acpc_python_client as acpc


ACTIONS = [
    acpc.ActionType.FOLD,
    acpc.ActionType.CALL,
    acpc.ActionType.RAISE
]


def convert_action_to_str(action):
    if action == acpc.ActionType.FOLD:
        return 'f'
    elif action == acpc.ActionType.CALL:
        return 'c'
    elif action == acpc.ActionType.RAISE:
        return 'r'
    else:
        raise RuntimeError('Invalid action: %s' % action)


def select_action(strategy):
    """Randomly select action from node strategy.

    Args:
        strategy (list(float)): Strategy of the node

    Returns:
        acpc.ActionType: Selected action.
    """
    choice = random.random()
    probability_sum = 0
    for i in range(3):
        action_probability = strategy[i]
        if action_probability == 0:
            continue
        probability_sum += action_probability
        if choice < probability_sum:
            return ACTIONS[i]
    # Return the last action since it could have not been selected due to floating point error
    return ACTIONS[2]


def get_info_set(game, match_state):
    """Return unique string representing each game state.

    Result is used as a node key in strategy.

    Args:
        game (Game): Game definition object
        match_state (MatchState): Current game state

    Returns:
        string: Representation of current game state.
    """
    state = match_state.get_state()
    info_set = ''

    num_hole_cards = game.get_num_hole_cards()
    viewing_player = match_state.get_viewing_player()
    info_set += '%s:' % ':'.join([str(state.get_hole_card(viewing_player, i)) for i in range(num_hole_cards)])

    total_board_cards_count = 0
    for round_index in range(state.get_round() + 1):
        new_total_board_cards_count = game.get_total_num_board_cards(round_index)
        if new_total_board_cards_count > total_board_cards_count:
            info_set += ':%s:' % ':'.join(
                [str(state.get_board_card(i))
                 for i in range(total_board_cards_count, new_total_board_cards_count)])
            total_board_cards_count = new_total_board_cards_count

        info_set += ''.join(
            [convert_action_to_str(state.get_action_type(round_index, action_index))
             for action_index in range(state.get_num_actions(round_index))])
    return info_set
