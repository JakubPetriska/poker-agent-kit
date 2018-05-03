import random
import sys

import acpc_python_client as acpc

from tools.agent_utils import select_action, get_info_set
from tools.io_util import read_strategy_from_file


class StrategyAgent(acpc.Agent):
    """Agent able to play any game when provided with game definition and correct strategy."""

    def __init__(self, strategy_file_path):
        super().__init__()
        self.strategy = read_strategy_from_file(None, strategy_file_path)

    def on_game_start(self, game):
        pass

    def on_next_turn(self, game, match_state, is_acting_player):
        if not is_acting_player:
            return

        info_set = get_info_set(game, match_state)
        node_strategy = self.strategy[info_set]
        selected_action = select_action(node_strategy)
        self.set_next_action(selected_action)

    def on_game_finished(self, game, match_state):
        pass


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage {game_file_path} {strategy_file_path} {dealer_hostname} {dealer_port}")
        sys.exit(1)

    client = acpc.Client(sys.argv[1], sys.argv[3], sys.argv[4])
    client.play(StrategyAgent(sys.argv[2]))
