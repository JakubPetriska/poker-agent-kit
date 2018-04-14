import sys
import os

import acpc_python_client as acpc

try:
    from tqdm import tqdm
except ImportError:
    print('!!! Install tqdm library for better progress information !!!\n')

from cfr.main import Cfr
from tools.game_tree.nodes import HoleCardsNode, ActionNode, BoardCardsNode
from tools.output_util import get_strategy

"""Trains strategy for poker agent using CFR algorithm and writes it to specified file.

Usage:
python train.py {game_file_path} {iterations} {strategy_output_path}

  game_file_path: Path to ACPC game definition file of a poker game for which we want create the strategy.
  iterations: Number of iterations for which the CFR algorithm will run.
  strategy_output_path: Path to file into which the result strategy will be written.
"""


def _write_to_output_file(output_path, lines):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as file:
        for line in lines:
            file.write(line)


def _write_strategy(game_tree, iterations, output_path):
    strategy_file_lines = []

    def process_strategy_line(strategy):
        node_strategy_str = ' '.join([str(prob) for prob in strategy[1]])
        strategy_file_lines.append('%s %s\n' % (strategy[0], node_strategy_str))

    try:
        with tqdm(total=1) as progress:
            progress.set_description('Obtaining strategy entries')
            get_strategy(game_tree, process_strategy_line)
            progress.update(1)
    except NameError:
        get_strategy(game_tree, process_strategy_line)

    try:
        with tqdm(total=1) as progress:
            progress.set_description('Sorting strategy file')
            strategy_file_lines_sorted = sorted(strategy_file_lines)
            progress.update(1)
    except NameError:
        strategy_file_lines_sorted = sorted(strategy_file_lines)

    strategy_file_lines_sorted = ['#  Training iterations: %s\n' % iterations] + strategy_file_lines_sorted

    try:
        with tqdm(total=1) as progress:
            progress.set_description('Writing strategy file')
            _write_to_output_file(output_path, strategy_file_lines_sorted)
            progress.update(1)
    except NameError:
        _write_to_output_file(output_path, strategy_file_lines_sorted)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage {game_file_path} {iterations} {strategy_output_path}")
        sys.exit(1)

    iterations = int(sys.argv[2])
    output_path = sys.argv[3]
    game = acpc.read_game_file(sys.argv[1])

    cfr = Cfr(game)
    cfr.train(iterations)

    _write_strategy(cfr.game_tree, iterations, output_path)
