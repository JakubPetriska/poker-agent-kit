import os
import unittest
from unittest import TestSuite
import random
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import acpc_python_client as acpc

from response.data_biased_response import DataBiasedResponse
from evaluation.exploitability import Exploitability
from tools.game_tree.builder import GameTreeBuilder
from tools.sampling import SamplesTreeNodeProvider
from tools.game_tree.node_provider import StrategyTreeNodeProvider
from tools.game_tree.nodes import ActionNode
from tools.walk_trees import walk_trees
from tools.game_utils import is_correct_strategy
from tools.io_util import write_strategy_to_file


FIGURES_FOLDER = 'test/dbr_correctness'
P_MAX_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]


class DbrCorrectnessTests(unittest.TestCase):
    def test_kuhn_cfr_correctness(self):
        kuhn_test_spec = {
            'title': 'Kuhn Poker DBR strategy performance',
            'game_file_path': 'games/kuhn.limit.2p.game',
            'test_counts': 1,
            'training_iterations': 100,
        }
        self.train_and_show_results(kuhn_test_spec)

    # def test_kuhn_bigdeck_cfr_correctness(self):
    #     kuhn_bigdeck_test_spec = {
    #         'title': 'Kuhn Bigdeck Poker CFR trained strategy exploitability',
    #         'game_file_path': 'games/kuhn.bigdeck.limit.2p.game',
    #         'test_counts': 1,
    #         'training_iterations': 1000,
    #         'checkpoint_iterations': 10
    #     }
    #     self.train_and_show_results(kuhn_bigdeck_test_spec)

    # def test_kuhn_bigdeck_2round_cfr_correctness(self):
    #     kuhn_bigdeck_2round_test_spec = {
    #         'title': 'Kuhn Bigdeck 2round Poker CFR trained strategy exploitability',
    #         'game_file_path': 'games/kuhn.bigdeck.2round.limit.2p.game',
    #         'test_counts': 1,
    #         'training_iterations': 1000,
    #         'checkpoint_iterations': 10
    #     }
    #     self.train_and_show_results(kuhn_bigdeck_2round_test_spec)

    # def test_leduc_cfr_correctness(self):
    #     leduc_test_spec = {
    #         'title': 'Leduc Hold\'em Poker CFR trained strategy exploitability',
    #         'game_file_path': 'games/leduc.limit.2p.game',
    #         'test_counts': 1,
    #         'training_iterations': 1000,
    #         'checkpoint_iterations': 10
    #     }
    #     self.train_and_show_results(leduc_test_spec)

    def train_and_show_results(self, test_spec):
        game = acpc.read_game_file(test_spec['game_file_path'])

        weak_opponent_samples_tree = GameTreeBuilder(game, SamplesTreeNodeProvider()).build_tree()
        weak_opponent_strategy_tree = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()
        def on_node(samples_node, strategy_node):
            if isinstance(samples_node, ActionNode):
                child_count = len(samples_node.children)
                samples_count = random.randrange(15)
                for i, a in enumerate(samples_node.children):
                    if i < (child_count - 1) and samples_count > 0:
                        action_samples_count = random.randrange(samples_count + 1)
                        samples_count -= action_samples_count
                        samples_node.action_decision_counts[a] = action_samples_count
                    else:
                        samples_node.action_decision_counts[a] = samples_count
                samples_sum = np.sum(samples_node.action_decision_counts)
                if samples_sum > 0:
                    strategy_node.strategy = samples_node.action_decision_counts / samples_sum
                else:
                    for a in strategy_node.children:
                        strategy_node.strategy[a] = 1 / len(strategy_node.children)
        walk_trees(on_node, weak_opponent_samples_tree, weak_opponent_strategy_tree)

        self.assertTrue(is_correct_strategy(weak_opponent_strategy_tree))

        exploitability = Exploitability(game)
        num_test_counts = test_spec['test_counts']
        data = np.zeros([num_test_counts, 2, len(P_MAX_VALUES)])
        for i in range(num_test_counts):
            print('%s/%s' % (i + 1, num_test_counts))

            for j, p_max in enumerate(P_MAX_VALUES):
                print('Pmax: %s - %s/%s' % (p_max, j + 1, len(P_MAX_VALUES)))

                dbr = DataBiasedResponse(game, weak_opponent_samples_tree, p_max=p_max)
                dbr.train(test_spec['training_iterations'])

                data[i, 0, j] = exploitability.evaluate(dbr.game_tree)
                data[i, 1, j] = exploitability.evaluate(weak_opponent_strategy_tree, dbr.game_tree)

                plt.figure(dpi=160)
                for k in range(i + 1):
                    run_index = math.floor(k / 2)
                    xdata = data[k, 0, :] if k < i or j == (len(P_MAX_VALUES) - 1) else data[k, 0, 0:j+1]
                    ydata = data[k, 1, :] if k < i or j == (len(P_MAX_VALUES) - 1) else data[k, 1, 0:j+1]
                    plt.plot(
                        xdata,
                        ydata,
                        label='Run %s' % (run_index + 1),
                        marker='o',
                        linewidth=0.8)

                if 'title' in test_spec:
                    plt.title(test_spec['title'])
                plt.xlabel('DBR trained strategy exploitability [mbb/g]')
                plt.ylabel('Random opponent exploitation by DBR strategy [mbb/g]')
                plt.grid()
                if num_test_counts > 1:
                    plt.legend()

                game_name = test_spec['game_file_path'].split('/')[1][:-5]
                figure_output_path = '%s/%s(it:%s).png' % (FIGURES_FOLDER, game_name, test_spec['training_iterations'])

                figures_directory = os.path.dirname(figure_output_path)
                if not os.path.exists(figures_directory):
                    os.makedirs(figures_directory)

                plt.savefig(figure_output_path)

        print('\033[91mThis test needs your assistance! ' +
            'Check the generated graph %s!\033[0m' % figure_output_path)


test_classes = [
    DbrCorrectnessTests
]


def load_tests(loader, tests, pattern):
    suite = TestSuite()
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
