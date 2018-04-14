from tools.game_tree.nodes import TerminalNode, HoleCardsNode, BoardCardsNode, ActionNode, StrategyActionNode


class NodeProvider:
    def create_terminal_node(self, parent, pot_commitment):
        return TerminalNode(parent, pot_commitment)

    def create_hole_cards_node(self, parent, card_count):
        return HoleCardsNode(parent, card_count)

    def create_board_cards_node(self, parent, card_count):
        return BoardCardsNode(parent, card_count)

    def create_action_node(self, parent, player):
        return ActionNode(parent, player)

class StrategyTreeNodeProvider:
    def create_terminal_node(self, parent, pot_commitment):
        return TerminalNode(parent, pot_commitment)

    def create_hole_cards_node(self, parent, card_count):
        return HoleCardsNode(parent, card_count)

    def create_board_cards_node(self, parent, card_count):
        return BoardCardsNode(parent, card_count)

    def create_action_node(self, parent, player):
        return StrategyActionNode(parent, player)
