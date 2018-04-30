from tools.game_tree.nodes import HoleCardsNode, BoardCardsNode, ActionNode, TerminalNode


def walk_trees(callback, *trees):
    callback(*trees)
    for child_key in trees[0].children.keys():
        walk_trees(callback, *[tree.children[child_key] for tree in trees])


def walk_trees_with_data(callback, data, *trees):
    new_data = callback(data, *trees)
    if len(new_data) != len(trees[0].children.items()):
        raise RuntimeError(
            'Returned data array must have same number of items as the node has children')
    for i, child_key in enumerate(trees[0].children.keys()):
        walk_trees_with_data(callback, new_data[i], *[tree.children[child_key] for tree in trees])
