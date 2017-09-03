import sys

import acpc_python_client as acpc
from tqdm import tqdm  # TODO make import optional

from cfr.build_tree import build_game_tree
from cfr.cfr import Cfr
from cfr.game_tree import HoleCardNode, ActionNode


def _action_to_str(action):
    if action == 0:
        return 'f'
    elif action == 1:
        return 'c'
    else:
        return 'r'


def _get_strategy_lines(lines, node, prefix=''):
    if type(node) == HoleCardNode:
        for card, child_node in node.children.items():
            _get_strategy_lines(lines, child_node, '%s%s:' % (prefix, str(card)))
    elif type(node) == ActionNode:
        node_strategy_str = ' '.join([str(prob) for prob in node.average_strategy])
        lines.append('%s %s\n' % (prefix, node_strategy_str))
        old_prefix = prefix
        if old_prefix.endswith(':'):
            old_prefix = old_prefix[:-1]
        for action, child_node in node.children.items():
            _get_strategy_lines(lines, child_node, '%s%s:' % (old_prefix, _action_to_str(action)))


def write_strategy(game_tree, output_path):
    with tqdm(total=100) as progress:
        progress.set_description('Obtaining strategy entries')
        strategy_file_lines = []
        _get_strategy_lines(strategy_file_lines, game_tree)
        progress.update(100)

    with tqdm(total=100) as progress:
        progress.set_description('Sorting strategy file')
        strategy_file_lines_sorted = sorted(strategy_file_lines)
        progress.update(100)

    with tqdm(total=100) as progress:
        progress.set_description('Writing strategy file')
        with open(output_path, 'w') as file:
            for line in strategy_file_lines_sorted:
                file.write(line)
        progress.update(100)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage {game_file_path} {iterations} {strategy_output_path}")
        sys.exit(1)

    game = acpc.read_game_file(sys.argv[1])
    game_tree = build_game_tree(game)

    iterations = int(sys.argv[2])

    output_path = sys.argv[3]

    cfr = Cfr(game.get_num_players(), game_tree)
    cfr.train(iterations)

    write_strategy(game_tree, output_path)
