import random
import sys

import acpc_python_client as acpc


class KuhnRandomAgent(acpc.Agent):
    def __init__(self):
        super().__init__()

    def on_game_start(self, game):
        pass

    def on_next_turn(self, game, match_state, is_acting_player):
        if not is_acting_player:
            return

        print('%s: %s %s' % (
            match_state.get_viewing_player(),
            self.is_fold_valid(),
            self.is_raise_valid()
        ))

        # Select between passing (fold or initial call)
        # or betting (raising or calling a bet)
        selected_action = random.randrange(2)
        if selected_action == 0 and self.is_fold_valid():
            self.set_next_action(acpc.ActionType.FOLD)
        elif selected_action == 1 and self.is_raise_valid():
            self.set_next_action(acpc.ActionType.RAISE)
        else:
            self.set_next_action(acpc.ActionType.CALL)

    def on_game_finished(self, game, match_state):
        pass


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage {game_file_path} {dealer_hostname} {dealer_port}")
        sys.exit(1)

    client = acpc.Client(sys.argv[1], sys.argv[2], sys.argv[3])
    client.play(KuhnRandomAgent())
