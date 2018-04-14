from tools.game_tree.nodes import HoleCardsNode, BoardCardsNode, ActionNode, TerminalNode


def walk_tree(tree, callback):
    callback(tree)
    for _, child_node in tree.children.items():
        walk_tree(child_node, callback)
