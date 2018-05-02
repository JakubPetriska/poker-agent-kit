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


FIGURES_FOLDER = 'verification/rnr'

KUHN_EQUILIBRIUM_STRATEGY_PATH = 'strategies/kuhn.limit.2p-equilibrium.strategy'
LEDUC_EQUILIBRIUM_STRATEGY_PATH = 'strategies/leduc.limit.2p-equilibrium.strategy'


class RnrVerificationTest(unittest.TestCase):
    def test_kuhn_rnr(self):
        self.train_and_show_results({
            'title': 'Restricted Nash Response agent exploitability',
            'game_file_path': 'games/kuhn.limit.2p.game',
            'base_strategy_path': KUHN_EQUILIBRIUM_STRATEGY_PATH,
            'opponent_tilt_types': [
                ('FOLD-ADD-0.5', Action.FOLD, TiltType.ADD, 0.5),
                ('CALL-ADD-0.5', Action.CALL, TiltType.ADD, 0.5),
                ('RAISE-ADD-0.75', Action.RAISE, TiltType.ADD, 0.75),
            ],
            'training_iterations': 1000,
            'checkpoint_iterations': 10,
        })

    def train_and_show_results(self, test_spec):
        game_file_path = test_spec['game_file_path']
        game = acpc.read_game_file(game_file_path)

        base_strategy = read_strategy_from_file(
            game_file_path,
            test_spec['base_strategy_path'])

        agents = test_spec['opponent_tilt_types']
        num_agents = len(agents)

        exploitability = Exploitability(game)

        checkpoints_count = math.ceil(
            test_spec['training_iterations'] / test_spec['checkpoint_iterations'])
        iteration_counts = np.zeros(checkpoints_count)
        exploitability_values = np.zeros([num_agents, checkpoints_count])
        for i, agent in enumerate(agents):
            print('%s/%s' % (i + 1, num_agents))

            def checkpoint_callback(game_tree, checkpoint_index, iterations):
                if i == 0:
                    iteration_counts[checkpoint_index] = iterations
                exploitability_values[i, checkpoint_index] = exploitability.evaluate(game_tree)

            opponent_strategy = create_agent_strategy_from_trained_strategy(
                    game_file_path,
                    base_strategy,
                    agent[1],
                    agent[2],
                    agent[3])

            rnr = RestrictedNashResponse(game, opponent_strategy, 0.5)
            rnr.train(
                test_spec['training_iterations'],
                test_spec['checkpoint_iterations'],
                checkpoint_callback)

            print('Exploitability: %s' % exploitability.evaluate(rnr.game_tree))

            plt.figure(dpi=160)
            for j in range(i + 1):
                plt.plot(
                    iteration_counts,
                    exploitability_values[j],
                    label=agents[j][0],
                    linewidth=0.8)

            plt.title(test_spec['title'])
            plt.xlabel('Training iterations')
            plt.ylabel('Strategy exploitability [mbb/g]')
            plt.grid()
            plt.legend()

            game_name = test_spec['game_file_path'].split('/')[1][:-5]
            figure_output_path = '%s/rnr-%s(it:%s-st:%s).png' % (FIGURES_FOLDER, game_name, test_spec['training_iterations'], test_spec['checkpoint_iterations'])

            figures_directory = os.path.dirname(figure_output_path)
            if not os.path.exists(figures_directory):
                os.makedirs(figures_directory)

            plt.savefig(figure_output_path)
