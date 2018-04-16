import os
import unittest
from unittest import TestSuite
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import acpc_python_client as acpc

from cfr.main import Cfr
from evaluation.exploitability import ExploitabilityCalculator

FIGURES_FOLDER = 'test/cfr-correctness-plots'

KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'
KUHN_TEST_ITERATIONS = 5
KUHN_TEST_TRAINING_ITERATIONS = 3000
KUHN_TEST_CHECKPOINT_ITERATIONS = 100

LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'
LEDUC_TEST_ITERATIONS = 1
LEDUC_TEST_TRAINING_ITERATIONS = 1000000
LEDUC_TEST_CHECKPOINT_ITERATIONS = 50000


class CfrCorrectnessTests(unittest.TestCase):
    def train_and_show_results(
            self,
            title,
            game_file_path,
            test_counts,
            training_iterations,
            checkpoint_training_iterations,
            yaxis_tick,
            show_progress=False):

        game = acpc.read_game_file(game_file_path)

        checkpoints_count = math.ceil(
            training_iterations / checkpoint_training_iterations)
        iteration_counts = np.zeros(checkpoints_count)
        exploitability = np.zeros([test_counts, checkpoints_count])
        for i in range(test_counts):
            if show_progress:
                print('%s/%s' % (i + 1, test_counts))

            cfr = Cfr(game, show_progress)

            exploitability_calculator = ExploitabilityCalculator(game)

            def checkpoint_callback(game_tree, checkpoint_index, iterations):
                if i == 0:
                    iteration_counts[checkpoint_index] = iterations
                exploitability[i][checkpoint_index] = exploitability_calculator.get_exploitability(
                    game_tree)

            cfr.train(training_iterations,
                    checkpoint_training_iterations, checkpoint_callback)

        _, ax = plt.subplots(1, 1)
        for i in range(test_counts):
            plt.plot(iteration_counts, exploitability[i] * -1000)

        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=yaxis_tick))

        plt.title(title)
        plt.xlabel('Training iterations')
        plt.ylabel('Strategy exploitability [mbb/g]')
        plt.grid()

        game_name = game_file_path.split('/')[1][:-5]
        figure_output_path = '%s/%s.png' % (FIGURES_FOLDER, game_name)

        figures_directory = os.path.dirname(figure_output_path)
        if not os.path.exists(figures_directory):
            os.makedirs(figures_directory)

        plt.savefig(figure_output_path)

        print('\033[91mThis test needs your assistance! ' +
            'Check the generated graph %s!\033[0m' % figure_output_path)

    # def test_kuhn_cfr_correctness(self):
    #     self.train_and_show_results(
    #         'Kuhn Poker CFR trained strategy exploitability',
    #         KUHN_POKER_GAME_FILE_PATH,
    #         KUHN_TEST_ITERATIONS,
    #         KUHN_TEST_TRAINING_ITERATIONS,
    #         KUHN_TEST_CHECKPOINT_ITERATIONS,
    #         10
    #     )

    def test_leduc_cfr_correctness(self):
        self.train_and_show_results(
            'Leduc Hold\'em Poker CFR trained strategy exploitability',
            LEDUC_POKER_GAME_FILE_PATH,
            LEDUC_TEST_ITERATIONS,
            LEDUC_TEST_TRAINING_ITERATIONS,
            LEDUC_TEST_CHECKPOINT_ITERATIONS,
            50,
            True
        )


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
