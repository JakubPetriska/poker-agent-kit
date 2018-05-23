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
from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import StrategyTreeNodeProvider
from tools.game_utils import copy_strategy, is_correct_strategy

from tools.io_util import write_strategy_to_file

FIGURES_FOLDER = 'verification/cfr_correctness'
COLLECT_MIN_EXPLOITABILITY = True
CHECK_STRATEGY_CORRECTNESS = True


class CfrCorrectnessTests(unittest.TestCase):
    def test_kuhn_cfr_correctness(self):
        kuhn_test_spec = {
            'title': 'Kuhn Poker CFR trained strategy exploitability',
            'game_file_path': 'games/kuhn.limit.2p.game',
            'training_iterations': 960,
            'weight_delay': 700,
            'checkpoint_iterations': 10
        }
        self.train_and_show_results(kuhn_test_spec)

    def test_kuhn_bigdeck_cfr_correctness(self):
        kuhn_bigdeck_test_spec = {
            'title': 'Kuhn Bigdeck Poker CFR trained strategy exploitability',
            'game_file_path': 'games/kuhn.bigdeck.limit.2p.game',
            'training_iterations': 1450,
            'weight_delay': 700,
            'checkpoint_iterations': 10
        }
        self.train_and_show_results(kuhn_bigdeck_test_spec)

    def test_kuhn_bigdeck_2round_cfr_correctness(self):
        kuhn_bigdeck_2round_test_spec = {
            'title': 'Kuhn Bigdeck 2round Poker CFR trained strategy exploitability',
            'game_file_path': 'games/kuhn.bigdeck.2round.limit.2p.game',
            'training_iterations': 1500,
            'weight_delay': 700,
            'checkpoint_iterations': 10
        }
        self.train_and_show_results(kuhn_bigdeck_2round_test_spec)

    def test_leduc_cfr_correctness(self):
        leduc_test_spec = {
            'title': 'Leduc Hold\'em Poker CFR trained strategy exploitability',
            'game_file_path': 'games/leduc.limit.2p.game',
            'training_iterations': 1500,
            'weight_delay': 700,
            'checkpoint_iterations': 10
        }
        self.train_and_show_results(leduc_test_spec)

    def train_and_show_results(self, test_spec):
        game = acpc.read_game_file(test_spec['game_file_path'])

        exploitability = Exploitability(game)

        iteration_counts = np.zeros(0)
        exploitability_values = np.zeros([1, 0])
        best_exploitability = float("inf")
        best_exploitability_strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()

        def checkpoint_callback(game_tree, checkpoint_index, iterations):
            nonlocal iteration_counts
            nonlocal exploitability_values
            nonlocal best_exploitability
            nonlocal best_exploitability_strategy

            iteration_counts = np.append(iteration_counts, iterations)

            if CHECK_STRATEGY_CORRECTNESS:
                self.assertTrue(is_correct_strategy(game_tree))

            exploitability_value = exploitability.evaluate(game_tree)
            exploitability_values = np.append(exploitability_values, exploitability_value)
            if COLLECT_MIN_EXPLOITABILITY and exploitability_value < best_exploitability:
                best_exploitability = exploitability_value
                copy_strategy(best_exploitability_strategy, game_tree)

        cfr = Cfr(game)
        cfr.train(
            test_spec['training_iterations'],
            weight_delay=test_spec['weight_delay'],
            checkpoint_iterations=test_spec['checkpoint_iterations'],
            checkpoint_callback=checkpoint_callback,
            minimal_action_probability=0.00006)

        best_response = BestResponse(game).solve(cfr.game_tree)
        player_utilities, _ = PlayerUtility(game).evaluate(cfr.game_tree, best_response)
        print(player_utilities.tolist())
        print('Exploitability: %s' % exploitability.evaluate(cfr.game_tree))

        if COLLECT_MIN_EXPLOITABILITY:
            min_exploitability = exploitability.evaluate(best_exploitability_strategy)
            min_exploitability_best_response = BestResponse(game).solve(best_exploitability_strategy)
            min_exploitability_player_utilities, _ = PlayerUtility(game).evaluate(best_exploitability_strategy, min_exploitability_best_response)
            self.assertEqual(min_exploitability, exploitability_values.min())
            print('Minimum exploitability: %s' % min_exploitability)
            print('Minimum exploitability player utilities: %s' % min_exploitability_player_utilities.tolist())
        else:
            print('Minimum exploitability: %s' % exploitability_values.min())

        plt.figure(dpi=160)
        plt.plot(iteration_counts, exploitability_values, linewidth=0.8)

        plt.title(test_spec['title'])
        plt.xlabel('Training iterations')
        plt.ylabel('Strategy exploitability [mbb/g]')
        plt.grid()

        game_name = test_spec['game_file_path'].split('/')[1][:-5]
        figure_output_path = '%s/%s(it:%s-st:%s).png' % (FIGURES_FOLDER, game_name, test_spec['training_iterations'], test_spec['checkpoint_iterations'])

        figures_directory = os.path.dirname(figure_output_path)
        if not os.path.exists(figures_directory):
            os.makedirs(figures_directory)

        plt.savefig(figure_output_path)

        write_strategy_to_file(
            cfr.game_tree,
            '%s/%s(it:%s).strategy' % (FIGURES_FOLDER, game_name, test_spec['training_iterations']),
            ['# Game utility against best response: %s' % player_utilities.tolist()])


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
