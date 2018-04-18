from tools.game_tree.nodes import HoleCardsNode, ActionNode, BoardCardsNode


def _action_to_str(action):
    if action == 0:
        return 'f'
    elif action == 1:
        return 'c'
    else:
        return 'r'


def get_strategy(tree, callback, prefix=''):
    if isinstance(tree, HoleCardsNode) or isinstance(tree, BoardCardsNode):
        for key, child_node in tree.children.items():
            new_prefix = prefix
            if new_prefix and not new_prefix.endswith(':'):
                new_prefix += ':'
            new_prefix += ':'.join([str(card) for card in key]) + ':'
            get_strategy(child_node, callback, new_prefix)
    elif isinstance(tree, ActionNode):
        callback((prefix, tree.strategy))
        for action, child_node in tree.children.items():
            get_strategy(child_node, callback, prefix + _action_to_str(action))


def get_strategy_lines(tree):
    strategy_lines = []

    def process_node_strategy(strategy):
        node_strategy_str = ' '.join([str(prob) for prob in strategy[1]])
        strategy_lines.append('%s %s\n' % (strategy[0], node_strategy_str))

    get_strategy(tree, process_node_strategy)
    return strategy_lines


def print_strategy(tree):
    def print_strategy_line(strategy):
        node_strategy_str = ' '.join([str(prob) for prob in strategy[1]])
        print('%s %s' % (strategy[0], node_strategy_str))
    get_strategy(tree, print_strategy_line)
