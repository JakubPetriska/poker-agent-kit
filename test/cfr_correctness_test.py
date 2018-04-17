import os
import unittest
from unittest import TestSuite
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import acpc_python_client as acpc

from cfr.main import Cfr
from evaluation.exploitability import Exploitability

FIGURES_FOLDER = 'test/cfr-correctness-plots'

KUHN_TEST_SPEC = {
    'title': 'Kuhn Poker CFR trained strategy exploitability',
    'game_file_path': 'games/kuhn.limit.2p.game',
    'test_counts': 3,
    'training_iterations': 800000,
    'checkpoint_iterations': 5000
}

KUHN_BIGDECK_TEST_SPEC = {
    'title': 'Kuhn Bigdeck Poker CFR trained strategy exploitability',
    'game_file_path': 'games/kuhn.bigdeck.limit.2p.game',
    'test_counts': 1,
    'training_iterations': 450000,
    'checkpoint_iterations': 10000
}

KUHN_BIGDECK_2ROUND_TEST_SPEC = {
    'title': 'Kuhn Bigdeck 2round Poker CFR trained strategy exploitability',
    'game_file_path': 'games/kuhn.bigdeck.2round.limit.2p.game',
    'test_counts': 1,
    'training_iterations': 1000,
    'checkpoint_iterations': 1000
}

LEDUC_TEST_SPEC = {
    'title': 'Leduc Hold\'em Poker CFR trained strategy exploitability',
    'game_file_path': 'games/leduc.limit.2p.game',
    'test_counts': 1,
    'training_iterations': 1000,
    'checkpoint_iterations': 1000
}


class CfrCorrectnessTests(unittest.TestCase):
    def test_kuhn_cfr_correctness(self):
        self.train_and_show_results(KUHN_TEST_SPEC)

    def test_kuhn_bigdeck_cfr_correctness(self):
        self.train_and_show_results(KUHN_BIGDECK_TEST_SPEC)

    def test_kuhn_bigdeck_2round_cfr_correctness(self):
        self.train_and_show_results(KUHN_BIGDECK_2ROUND_TEST_SPEC)

    def test_leduc_cfr_correctness(self):
        self.train_and_show_results(LEDUC_TEST_SPEC)

    def train_and_show_results(self, test_spec):
        game = acpc.read_game_file(test_spec['game_file_path'])

        checkpoints_count = math.ceil(
            test_spec['training_iterations'] / test_spec['checkpoint_iterations'])
        iteration_counts = np.zeros(checkpoints_count)
        exploitability = np.zeros([test_spec['test_counts'], checkpoints_count])
        for i in range(test_spec['test_counts']):
            print('%s/%s' % (i + 1, test_spec['test_counts']))

            cfr = Cfr(game)

            exploitability = Exploitability(game)

            def checkpoint_callback(game_tree, checkpoint_index, iterations):
                if i == 0:
                    iteration_counts[checkpoint_index] = iterations
                exploitability[i][checkpoint_index] = exploitability.evaluate(game_tree)

            cfr.train(
                test_spec['training_iterations'],
                test_spec['checkpoint_iterations'],
                checkpoint_callback)

        plt.figure()
        for i in range(test_spec['test_counts']):
            plt.plot(iteration_counts, exploitability[i])

        plt.title(test_spec['title'])
        plt.xlabel('Training iterations')
        plt.ylabel('Strategy exploitability [mbb/g]')
        plt.grid()

        game_name = test_spec['game_file_path'].split('/')[1][:-5]
        figure_output_path = '%s/%s.png' % (FIGURES_FOLDER, game_name)

        figures_directory = os.path.dirname(figure_output_path)
        if not os.path.exists(figures_directory):
            os.makedirs(figures_directory)

        plt.savefig(figure_output_path)

        print('\033[91mThis test needs your assistance! ' +
            'Check the generated graph %s!\033[0m' % figure_output_path)


test_classes = [
    CfrCorrectnessTests
]


def load_tests(loader, tests, pattern):
    suite = TestSuite()
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite


if __name__ == "__main__":
    unittest.main()
