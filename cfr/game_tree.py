from cfr.constants import NUM_ACTIONS


class Node:
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.children = {}

    def set_child(self, key, child):
        self.children[key] = child


class TerminalNode(Node):
    def __init__(self, parent):
        super().__init__(parent)


class HoleCardNode(Node):
    def __init__(self, parent, possible_cards):
        super().__init__(parent)
        self.possible_cards = possible_cards


class ActionNode(Node):
    def __init__(self, parent, player):
        super().__init__(parent)
        self.player = player
        self.regret_sum = [0] * NUM_ACTIONS
        self.strategy = [0] * NUM_ACTIONS
        self.strategy_sum = [0] * NUM_ACTIONS
