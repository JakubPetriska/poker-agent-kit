import unittest
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import acpc_python_client as acpc

from response.restricted_nash_response import RestrictedNashResponse
from cfr.main import Cfr
from tools.constants import Action
from weak_agents.action_tilted_agent import create_agent_strategy_from_trained_strategy, TiltType
from tools.io_util import read_strategy_from_file
from evaluation.exploitability import Exploitability
from tools.io_util import get_new_path, create_path_dirs
from tools.game_utils import is_correct_strategy
from response.best_response import BestResponse
from tools.io_util import write_strategy_to_file


FIGURES_FOLDER = 'verification/rnr_correctness'

KUHN_EQUILIBRIUM_STRATEGY_PATH = 'strategies/kuhn.limit.2p-equilibrium.strategy'
LEDUC_EQUILIBRIUM_STRATEGY_PATH = 'strategies/leduc.limit.2p-equilibrium.strategy'

PLOT_OPPONENT_EXPLOITABILITY = True
LINE_WIDTH = 0.7

if PLOT_OPPONENT_EXPLOITABILITY:
    PLOT_COUNT_PER_AGENT = 3
else:
    PLOT_COUNT_PER_AGENT = 2


def get_agent_name(agent):
        return '%s-%s-%s' % (str(agent[0]).split('.')[1], str(agent[1]).split('.')[1], agent[2])


class RnrCorrectnessTest(unittest.TestCase):
    def test_kuhn_rnr(self):
        self.train_and_show_results({
            'title': 'Restricted Nash Response agent exploitability',
            'game_file_path': 'games/kuhn.limit.2p.game',
            'base_strategy_path': KUHN_EQUILIBRIUM_STRATEGY_PATH,
            'opponent_tilt_types': [
                (Action.FOLD, TiltType.ADD, 0.5, 0.8),
                (Action.FOLD, TiltType.ADD, 0.5, 0.5),
                (Action.FOLD, TiltType.ADD, 0.5, 0.2),
                (Action.FOLD, TiltType.ADD, 0.5, 1),

                (Action.CALL, TiltType.ADD, 0.5, 0.8),
                (Action.CALL, TiltType.ADD, 0.5, 0.5),
                (Action.CALL, TiltType.ADD, 0.5, 0.2),

                (Action.RAISE, TiltType.ADD, 0.75, 0.8),
                (Action.RAISE, TiltType.ADD, 0.75, 0.5),
                (Action.RAISE, TiltType.ADD, 0.75, 0.2),
            ],
            'training_iterations': 1000,
            'checkpoint_iterations': 10,
            'overwrite_figure': True,
            # 'print_response_strategies': True,
            # 'print_opponent_strategies': True,
            # 'print_best_responses': True,
        })

    def train_and_show_results(self, test_spec):
        game_file_path = test_spec['game_file_path']
        game = acpc.read_game_file(game_file_path)

        base_strategy, _ = read_strategy_from_file(
            game_file_path,
            test_spec['base_strategy_path'])

        agents = test_spec['opponent_tilt_types']
        num_agents = len(agents)

        game_name = game_file_path.split('/')[1][:-5]
        overwrite_figure = test_spec['overwrite_figure'] if 'overwrite_figure' in test_spec else False
        figure_path = get_new_path(
            '%s/%s(it:%s-st:%s)' % (FIGURES_FOLDER, game_name, test_spec['training_iterations'], test_spec['checkpoint_iterations']),
            '.png',
            overwrite_figure)
        create_path_dirs(figure_path)

        exp = Exploitability(game)

        checkpoints_count = math.ceil(
            (test_spec['training_iterations'] - 700) / test_spec['checkpoint_iterations'])
        iteration_counts = np.zeros(checkpoints_count)
        exploitability_values = np.zeros([num_agents, checkpoints_count])
        vs_opponent_utility_values = np.zeros([num_agents, checkpoints_count])
        opponent_exploitability_values = np.zeros(num_agents)
        for i, agent in enumerate(agents):
            print('%s/%s' % (i + 1, num_agents))

            opponent_strategy = create_agent_strategy_from_trained_strategy(
                    game_file_path,
                    base_strategy,
                    agent[0],
                    agent[1],
                    agent[2])

            self.assertTrue(is_correct_strategy(opponent_strategy))

            if 'print_opponent_strategies' in test_spec and test_spec['print_opponent_strategies']:
                write_strategy_to_file(
                    opponent_strategy,
                    '%s/%s.strategy' % (os.path.dirname(figure_path), get_agent_name(agent)))

            if 'print_best_responses' in test_spec and test_spec['print_best_responses']:
                opponent_best_response = BestResponse(game).solve(opponent_strategy)
                write_strategy_to_file(
                    opponent_best_response,
                    '%s/%s-best_response.strategy' % (os.path.dirname(figure_path), get_agent_name(agent)))


            if PLOT_OPPONENT_EXPLOITABILITY:
                opponent_exploitability = exp.evaluate(opponent_strategy)
                opponent_exploitability_values[i] = opponent_exploitability
                print('%s exploitability: %s' % (get_agent_name(agent), opponent_exploitability))

            def checkpoint_callback(game_tree, checkpoint_index, iterations):
                if i == 0:
                    iteration_counts[checkpoint_index] = iterations
                self.assertTrue(is_correct_strategy(game_tree))
                exploitability_values[i, checkpoint_index] = exp.evaluate(game_tree)
                vs_opponent_utility_values[i, checkpoint_index] = exp.evaluate(opponent_strategy, game_tree)

            rnr = RestrictedNashResponse(game, opponent_strategy, agent[3])
            rnr.train(
                test_spec['training_iterations'],
                checkpoint_iterations=test_spec['checkpoint_iterations'],
                checkpoint_callback=checkpoint_callback)

            if 'print_response_strategies' in test_spec and test_spec['print_response_strategies']:
                write_strategy_to_file(
                    rnr.game_tree,
                    '%s-%s-p=%s.strategy' % (figure_path[:-len('.png')], get_agent_name(agent), agent[3]))

            print('Vs opponent value: %s' % exp.evaluate(opponent_strategy, rnr.game_tree))
            print('Exploitability: %s' % exp.evaluate(rnr.game_tree))

            plt.figure(dpi=300)
            ax = plt.subplot(111)
            for j in range(i + 1):
                p = plt.plot(
                    iteration_counts,
                    exploitability_values[j],
                    label='%s-p=%s exploitability' % (get_agent_name(agents[j]), agents[j][3]),
                    linewidth=LINE_WIDTH)
                plt.plot(
                    iteration_counts,
                    vs_opponent_utility_values[j],
                    '--',
                    label='Utility against opponent strategy',
                    color=p[0].get_color(),
                    linewidth=LINE_WIDTH)
                if PLOT_OPPONENT_EXPLOITABILITY:
                    plt.plot(
                        iteration_counts,
                        np.ones(checkpoints_count) * opponent_exploitability_values[j],
                        ':',
                        label='Opponent exploitability',
                        color=p[0].get_color(),
                        linewidth=LINE_WIDTH)

            plt.title(test_spec['title'])
            plt.xlabel('Training iterations')
            plt.ylabel('Strategy exploitability [mbb/g]')
            plt.grid()
            handles, labels = ax.get_legend_handles_labels()
            new_handles = []
            new_labels = []
            for i in range(PLOT_COUNT_PER_AGENT):
                for j in range(i, len(handles), PLOT_COUNT_PER_AGENT):
                    new_handles += [handles[j]]
                    new_labels += [labels[j]]
            lgd = plt.legend(new_handles, new_labels, loc='upper center', bbox_to_anchor=(0.5,-0.1), ncol=PLOT_COUNT_PER_AGENT)

            plt.savefig(figure_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

        print('Figure written to %s' % figure_path)
