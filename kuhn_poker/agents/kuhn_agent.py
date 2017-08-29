import random
import sys

import acpc_python_client as acpc

ACTIONS = [
    acpc.ActionType.FOLD,
    acpc.ActionType.CALL,
    acpc.ActionType.RAISE
]


def convert_to_strategy_card(card):
    if card == 43:
        return 1
    elif card == 47:
        return 2
    elif card == 51:
        return 3
    else:
        raise RuntimeError(
            'Invalid card: (%s, rank: %s, suit: %s)' % (
                card,
                acpc.game_utils.card_rank(card),
                acpc.game_utils.card_suit(card)))


def convert_action_to_str(action):
    if action == acpc.ActionType.CALL:
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


class KuhnAgent(acpc.Agent):
    def __init__(self, strategy_file_path):
        super().__init__()

        self.strategy = {}
        with open(strategy_file_path, 'r') as strategy_file:
            for line in strategy_file:
                line_split = line.split(':')
                node_path = line_split[0].strip()
                action_probabilities = [float(probStr) for probStr in line_split[1].split()]
                self.strategy[node_path] = action_probabilities

    def on_game_start(self, game):
        pass

    def on_next_turn(self, game, match_state, is_acting_player):
        if not is_acting_player:
            return

        state = match_state.get_state()
        card = state.get_hole_card(0)
        strategy_card = convert_to_strategy_card(card)
        num_actions = state.get_num_actions(0)
        info_set = str(strategy_card)
        for i in range(num_actions):
            action = state.get_action_type(0, i)
            action_str = convert_action_to_str(action)
            info_set += action_str

        self.set_next_action(select_action(self.strategy[info_set]))

    def on_game_finished(self, game, match_state):
        pass


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage {game_file_path} {strategy_file_path} {dealer_hostname} {dealer_port}")
        sys.exit(1)

    client = acpc.Client(sys.argv[1], sys.argv[3], sys.argv[4])
    client.play(KuhnAgent(sys.argv[2]))
