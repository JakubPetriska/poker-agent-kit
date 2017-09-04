import random
import sys

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


class StrategyAgent(acpc.Agent):
    def __init__(self, strategy_file_path):
        super().__init__()

        self.current_info_set = None

        strategy = {}
        with open(strategy_file_path, 'r') as strategy_file:
            for line in strategy_file:
                line_split = line.split(' ')
                strategy[line_split[0]] = [float(probStr) for probStr in line_split[1:4]]
        self.strategy = strategy

    def on_game_start(self, game):
        self.current_info_set = ''

    def on_next_turn(self, game, match_state, is_acting_player):
        state = match_state.get_state()

        if not self.current_info_set:
            num_hole_cards = game.get_num_hole_cards()
            for i in range(num_hole_cards):
                self.current_info_set += '%s:' % state.get_hole_card(i)

        round_index = state.get_round()
        num_actions = state.get_num_actions(round_index)
        if num_actions > 0:
            new_action_type = state.get_action_type(round_index, num_actions - 1)
            self.current_info_set += '%s' % convert_action_to_str(new_action_type)

        if not is_acting_player:
            return

        strategy = self.strategy[self.current_info_set]
        self.set_next_action(select_action(strategy))

    def on_game_finished(self, game, match_state):
        pass


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage {game_file_path} {strategy_file_path} {dealer_hostname} {dealer_port}")
        sys.exit(1)

    client = acpc.Client(sys.argv[1], sys.argv[3], sys.argv[4])
    client.play(StrategyAgent(sys.argv[2]))
