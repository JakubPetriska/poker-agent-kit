import os
import numpy as np

import acpc_python_client as acpc

from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import StrategyTreeNodeProvider
from tools.walk_trees import walk_trees
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


def write_strategy_to_file(tree, output_path, prefix_lines=None):
    output_directory = os.path.dirname(output_path)
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    with open(output_path, 'w') as file:
        if prefix_lines:
            for line in prefix_lines:
                if line.endswith('\n'):
                    line_with_newline = line
                else:
                    line_with_newline = '%s\n' % line
                file.write(line_with_newline)
        for line in sorted(get_strategy_lines(tree)):
            file.write(line)


def read_strategy_from_file(game_file_path, strategy_file_path):
    strategy = {}
    with open(strategy_file_path, 'r') as strategy_file:
        for line in strategy_file:
            if not line.strip() or line.strip().startswith('#'):
                continue
            line_split = line.split(' ')
            strategy[line_split[0]] = [float(probStr) for probStr in line_split[1:4]]

    if not game_file_path:
        return strategy

    game = acpc.read_game_file(game_file_path)
    strategy_tree = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

    def on_node(node):
        if isinstance(node, ActionNode):
            nonlocal strategy
            node_strategy = np.array(strategy[str(node)])
            np.copyto(node.strategy, node_strategy)

    walk_trees(on_node, strategy_tree)
    return strategy_tree, strategy
