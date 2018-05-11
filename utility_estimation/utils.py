def get_all_board_cards(game, state):
    total_num_board_cards = game.get_total_num_board_cards(state.get_round())
    return [state.get_board_card(c) for c in range(0, total_num_board_cards)]

def get_board_cards(game, state, round_index):
    total_num_board_cards = game.get_total_num_board_cards(round_index)
    round_num_board_cards = game.get_num_board_cards(round_index)
    start_board_card_index = total_num_board_cards - round_num_board_cards
    board_cards = [state.get_board_card(c) for c in range(start_board_card_index, total_num_board_cards)]
    return tuple(sorted(board_cards))
