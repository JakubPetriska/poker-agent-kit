import sys

import acpc_python_client as acpc
from tqdm import tqdm  # TODO make import optional

from cfr.cfr import Cfr
from cfr.game_tree import HoleCardsNode, ActionNode


def _action_to_str(action):
    if action == 0:
        return 'f'
    elif action == 1:
        return 'c'
    else:
        return 'r'


def _get_strategy_lines(lines, node, prefix=''):
    if type(node) == HoleCardsNode:
        for card, child_node in node.children.items():
            new_prefix = prefix
            if new_prefix and not new_prefix.endswith(':'):
                new_prefix += ':'
            new_prefix += '%s:' % card
            _get_strategy_lines(lines, child_node, new_prefix)
    elif type(node) == ActionNode:
        node_strategy_str = ' '.join([str(prob) for prob in node.average_strategy])
        lines.append('%s %s\n' % (prefix, node_strategy_str))

        for action, child_node in node.children.items():
            _get_strategy_lines(lines, child_node, prefix + _action_to_str(action))


def write_strategy(game_tree, output_path):
    with tqdm(total=1) as progress:
        progress.set_description('Obtaining strategy entries')
        strategy_file_lines = []
        _get_strategy_lines(strategy_file_lines, game_tree)
        progress.update(1)

    with tqdm(total=1) as progress:
        progress.set_description('Sorting strategy file')
        strategy_file_lines_sorted = sorted(strategy_file_lines)
        progress.update(1)

    with tqdm(total=1) as progress:
        progress.set_description('Writing strategy file')
        with open(output_path, 'w') as file:
            for line in strategy_file_lines_sorted:
                file.write(line)
        progress.update(1)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage {game_file_path} {iterations} {strategy_output_path}")
        sys.exit(1)

    iterations = int(sys.argv[2])
    output_path = sys.argv[3]
    game = acpc.read_game_file(sys.argv[1])

    cfr = Cfr(game)
    cfr.train(iterations)

    write_strategy(cfr.game_tree, output_path)
