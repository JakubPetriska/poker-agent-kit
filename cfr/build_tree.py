from cfr.game_tree import HoleCardNode, ActionNode, TerminalNode


def build_game_tree(game):
    if not game.get_num_players() == 2:
        raise AttributeError('Only games with 2 players supported')

    # TODO obey the game, this is just hardcoded Kuhn tree for 2 players
    root = HoleCardNode(None)
    for card in range(1, 4):
        first_action_node = ActionNode(root, 0)
        root.set_child(card, first_action_node)

        second_action_node_call = ActionNode(first_action_node, 1)
        first_action_node.set_child(1, second_action_node_call)

        second_action_node_call.set_child(1, TerminalNode(second_action_node_call, [1, 1]))

        third_action_node_call_raise = ActionNode(second_action_node_call, 0)
        second_action_node_call.set_child(2, third_action_node_call_raise)

        third_action_node_call_raise.set_child(0, TerminalNode(second_action_node_call, [1, 2]))
        third_action_node_call_raise.set_child(1, TerminalNode(second_action_node_call, [2, 2]))

        second_action_node_raise = ActionNode(first_action_node, 1)
        first_action_node.set_child(2, second_action_node_raise)

        second_action_node_raise.set_child(0, TerminalNode(second_action_node_raise, [2, 1]))
        second_action_node_raise.set_child(1, TerminalNode(second_action_node_raise, [2, 2]))
    return root
