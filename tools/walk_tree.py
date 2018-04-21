from tools.game_tree.nodes import HoleCardsNode, BoardCardsNode, ActionNode, TerminalNode


def walk_tree(tree, callback):
    callback(tree)
    for _, child_node in tree.children.items():
        walk_tree(child_node, callback)

def walk_tree_with_data(tree, data, callback):
    new_data = callback(tree, data)
    if len(new_data) != len(tree.children.items()):
        raise RuntimeError(
            'Returned data array must have same number of items as the node has children')
    for i, item in enumerate(tree.children.items()):
        node = item[1]
        walk_tree_with_data(node, new_data[i], callback)
