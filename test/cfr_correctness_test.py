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
from response.best_response import BestResponse
from evaluation.player_utility import PlayerUtility

from tools.output_util import write_strategy_to_file

FIGURES_FOLDER = 'test/cfr_correctness'


class CfrCorrectnessTests(unittest.TestCase):
    def test_kuhn_cfr_correctness(self):
        kuhn_test_spec = {
            'title': 'Kuhn Poker CFR trained strategy exploitability',
            'game_file_path': 'games/kuhn.limit.2p.game',
            'test_counts': 1,
            'training_iterations': 1000,
            'checkpoint_iterations': 10
        }
        self.train_and_show_results(kuhn_test_spec)

    def test_kuhn_bigdeck_cfr_correctness(self):
        kuhn_bigdeck_test_spec = {
            'title': 'Kuhn Bigdeck Poker CFR trained strategy exploitability',
            'game_file_path': 'games/kuhn.bigdeck.limit.2p.game',
            'test_counts': 1,
            'training_iterations': 1000,
            'checkpoint_iterations': 10
        }
        self.train_and_show_results(kuhn_bigdeck_test_spec)

    def test_kuhn_bigdeck_2round_cfr_correctness(self):
        kuhn_bigdeck_2round_test_spec = {
            'title': 'Kuhn Bigdeck 2round Poker CFR trained strategy exploitability',
            'game_file_path': 'games/kuhn.bigdeck.2round.limit.2p.game',
            'test_counts': 1,
            'training_iterations': 1000,
            'checkpoint_iterations': 10
        }
        self.train_and_show_results(kuhn_bigdeck_2round_test_spec)

    def test_leduc_cfr_correctness(self):
        leduc_test_spec = {
            'title': 'Leduc Hold\'em Poker CFR trained strategy exploitability',
            'game_file_path': 'games/leduc.limit.2p.game',
            'test_counts': 1,
            'training_iterations': 1000,
            'checkpoint_iterations': 10
        }
        self.train_and_show_results(leduc_test_spec)

    def train_and_show_results(self, test_spec):
        game = acpc.read_game_file(test_spec['game_file_path'])

        exploitability = Exploitability(game)

        checkpoints_count = math.ceil(
            test_spec['training_iterations'] / test_spec['checkpoint_iterations'])
        iteration_counts = np.zeros(checkpoints_count)
        exploitability_values = np.zeros([test_spec['test_counts'], checkpoints_count])

        for i in range(test_spec['test_counts']):
            print('%s/%s' % (i + 1, test_spec['test_counts']))

            def checkpoint_callback(game_tree, checkpoint_index, iterations):
                if i == 0:
                    iteration_counts[checkpoint_index] = iterations
                exploitability_values[i, checkpoint_index] = exploitability.evaluate(game_tree)

            cfr = Cfr(game)
            cfr.train(
                test_spec['training_iterations'],
                test_spec['checkpoint_iterations'],
                checkpoint_callback,
                minimal_action_probability=0.00006)

            best_response = BestResponse(game).solve(cfr.game_tree)
            player_utilities, _ = PlayerUtility(game).evaluate(cfr.game_tree, best_response)
            print(player_utilities.tolist())

            plt.figure(dpi=160)
            for j in range(i + 1):
                run_index = math.floor(j / 2)
                plt.plot(
                    iteration_counts,
                    exploitability_values[j],
                    label='Run %s' % (run_index + 1),
                    linewidth=0.8)

            plt.title(test_spec['title'])
            plt.xlabel('Training iterations')
            plt.ylabel('Strategy exploitability [mbb/g]')
            plt.grid()
            if test_spec['test_counts'] > 1:
                plt.legend()

            game_name = test_spec['game_file_path'].split('/')[1][:-5]
            figure_output_path = '%s/%s(it:%s-st:%s).png' % (FIGURES_FOLDER, game_name, test_spec['training_iterations'], test_spec['checkpoint_iterations'])

            figures_directory = os.path.dirname(figure_output_path)
            if not os.path.exists(figures_directory):
                os.makedirs(figures_directory)

            plt.savefig(figure_output_path)

            write_strategy_to_file(
                cfr.game_tree,
                'test/cfr_correctness/%s(it:%s).strategy' % (game_name, test_spec['training_iterations']),
                ['# Game utility against best response: %s' % str(player_utilities.tolist())])

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
    unittest.main(verbosity=2)
