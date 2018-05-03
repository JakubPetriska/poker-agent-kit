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


FIGURES_FOLDER = 'verification/rnr_correctness'

KUHN_EQUILIBRIUM_STRATEGY_PATH = 'strategies/kuhn.limit.2p-equilibrium.strategy'
LEDUC_EQUILIBRIUM_STRATEGY_PATH = 'strategies/leduc.limit.2p-equilibrium.strategy'

PLOT_OPPONENT_EXPLOITABILITY = True
LINE_WIDTH = 0.7

if PLOT_OPPONENT_EXPLOITABILITY:
    PLOT_COUNT_PER_AGENT = 3
else:
    PLOT_COUNT_PER_AGENT = 2

class RnrCorrectnessTest(unittest.TestCase):
    def test_kuhn_rnr(self):
        self.train_and_show_results({
            'title': 'Restricted Nash Response agent exploitability',
            'game_file_path': 'games/kuhn.limit.2p.game',
            'base_strategy_path': KUHN_EQUILIBRIUM_STRATEGY_PATH,
            'opponent_tilt_types': [
                ('FOLD-ADD-0.5-p=.8', Action.FOLD, TiltType.ADD, 0.5, 0.8),
                ('FOLD-ADD-0.5-p=.5', Action.FOLD, TiltType.ADD, 0.5, 0.5),
                ('FOLD-ADD-0.5-p=.2', Action.FOLD, TiltType.ADD, 0.5, 0.2),
                ('CALL-ADD-0.5-p=.8', Action.CALL, TiltType.ADD, 0.5, 0.8),
                ('CALL-ADD-0.5-p=.5', Action.CALL, TiltType.ADD, 0.5, 0.5),
                ('CALL-ADD-0.5-p=.2', Action.CALL, TiltType.ADD, 0.5, 0.2),
                ('RAISE-ADD-0.75-p=.8', Action.RAISE, TiltType.ADD, 0.75, 0.8),
                ('RAISE-ADD-0.75-p=.5', Action.RAISE, TiltType.ADD, 0.75, 0.5),
                ('RAISE-ADD-0.75-p=.2', Action.RAISE, TiltType.ADD, 0.75, 0.2),
                # ('RAISE-ADD-0.75-p=0', Action.RAISE, TiltType.ADD, 0.75, 0),
                # ('RAISE-ADD-0.75-p=0.01', Action.RAISE, TiltType.ADD, 0.75, 0.01),
                # ('RAISE-ADD-0.75-p=0.05', Action.RAISE, TiltType.ADD, 0.75, 0.05),
                # ('RAISE-ADD-0.75-p=0.08', Action.RAISE, TiltType.ADD, 0.75, 0.08),
                # ('RAISE-ADD-0.75-p=0.1', Action.RAISE, TiltType.ADD, 0.75, 0.1),
                # ('RAISE-ADD-0.75-p=1', Action.RAISE, TiltType.ADD, 0.75, 1),
            ],
            'training_iterations': 2000,
            'checkpoint_iterations': 10,
        })

    def train_and_show_results(self, test_spec):
        game_file_path = test_spec['game_file_path']
        game = acpc.read_game_file(game_file_path)

        base_strategy, _ = read_strategy_from_file(
            game_file_path,
            test_spec['base_strategy_path'])

        agents = test_spec['opponent_tilt_types']
        num_agents = len(agents)

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
                    agent[1],
                    agent[2],
                    agent[3])

            if PLOT_OPPONENT_EXPLOITABILITY:
                opponent_exploitability = exp.evaluate(opponent_strategy)
                opponent_exploitability_values[i] = opponent_exploitability
                print('%s exploitability: %s' % (agent[0], opponent_exploitability))

            def checkpoint_callback(game_tree, checkpoint_index, iterations):
                if i == 0:
                    iteration_counts[checkpoint_index] = iterations
                exploitability_values[i, checkpoint_index] = exp.evaluate(game_tree)
                vs_opponent_utility_values[i, checkpoint_index] = exp.evaluate(opponent_strategy, game_tree)

            rnr = RestrictedNashResponse(game, opponent_strategy, agent[4])
            rnr.train(
                test_spec['training_iterations'],
                checkpoint_iterations=test_spec['checkpoint_iterations'],
                checkpoint_callback=checkpoint_callback)

            print('Exploitability: %s' % exp.evaluate(rnr.game_tree))

            plt.figure(dpi=300)
            ax = plt.subplot(111)
            for j in range(i + 1):
                p = plt.plot(
                    iteration_counts,
                    exploitability_values[j],
                    label='%s exploitability' % agents[j][0],
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

            game_name = test_spec['game_file_path'].split('/')[1][:-5]
            figure_output_path = '%s/%s(it:%s-st:%s).png' \
                % (FIGURES_FOLDER, game_name, test_spec['training_iterations'], test_spec['checkpoint_iterations'])

            figures_directory = os.path.dirname(figure_output_path)
            if not os.path.exists(figures_directory):
                os.makedirs(figures_directory)

            plt.savefig(figure_output_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
