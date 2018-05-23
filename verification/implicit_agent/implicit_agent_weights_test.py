import unittest
import os
import sys
import shutil
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

import acpc_python_client as acpc

from tools.constants import Action
from weak_agents.action_tilted_agent import create_agent_strategy_from_trained_strategy, TiltType
from tools.io_util import read_strategy_from_file, get_new_path
from tools.io_util import write_strategy_to_file
from implicit_modelling.implicit_modelling_agent import ImplicitModellingAgent
from tools.game_utils import get_big_blind_size
from utility_estimation.simple import SimpleUtilityEstimator
from utility_estimation.imaginary_observations import ImaginaryObservationsUtilityEstimator
from utility_estimation.aivat import AivatUtilityEstimator


TEST_DIRECTORY = 'verification/implicit_agent'
PORTFOLIOS_DIRECTORY = '%s/portfolios' % TEST_DIRECTORY
GAME_LOGS_DIRECTORY = '%s/weights_test' % TEST_DIRECTORY

START_DEALER_AND_OPPONENT_SCRIPT_PATH = './scripts/start_dealer_and_other_players.sh'

NUM_EVAL_HANDS = 3000


class WeightsLoggingImplicitModellingAgent(ImplicitModellingAgent):
    def __init__(
            self,
            game_file_path,
            portfolio_strategy_files_paths,
            data,
            exp3g_gamma=0.02,
            exp3g_eta=0.025,
            utility_estimator_class=SimpleUtilityEstimator,
            utility_estimator_args=None):
        super().__init__(
            game_file_path,
            portfolio_strategy_files_paths,
            exp3g_gamma,
            exp3g_eta,
            utility_estimator_class,
            utility_estimator_args)
        self.data = data
        self.hand_index = 0

    def on_game_finished(self, game, match_state):
        super().on_game_finished(game, match_state)
        self.data[self.hand_index, :] = self.bandit_algorithm.weights
        self.hand_index += 1


class ImplicitAgentWeightsTest(unittest.TestCase):
    def test_kuhn_simple_portfolio(self):
        self.evaluate_agent({
            'portfolio_name': 'kuhn_simple_portfolio',
            'game_file_path': 'games/kuhn.limit.2p.game',
            'utility_estimator_class': AivatUtilityEstimator,
            'num_iterations': 1,
            'utility_estimator_args': {
                'equilibirum_strategy_path': 'strategies/kuhn.limit.2p-equilibrium.strategy'
            },
            'overwrite': True,
        })

    def test_leduc_easy_portfolio(self):
        self.evaluate_agent({
            'portfolio_name': 'leduc_small_portfolio',
            'game_file_path': 'games/leduc.limit.2p.game',
            'utility_estimator_class': AivatUtilityEstimator,
            'num_iterations': 50,
            'utility_estimator_args': {
                'equilibirum_strategy_path': 'strategies/leduc.limit.2p-equilibrium.strategy'
            }
        })

    def test_leduc_easy_portfolio_io(self):
        self.evaluate_agent({
            'portfolio_name': 'leduc_small_portfolio',
            'game_file_path': 'games/leduc.limit.2p.game',
            'utility_estimator_class': ImaginaryObservationsUtilityEstimator,
            'num_iterations': 50,
        })

    def test_leduc_hard_portfolio(self):
        self.evaluate_agent({
            'portfolio_name': 'leduc_small_hard_portfolio',
            'game_file_path': 'games/leduc.limit.2p.game',
            'utility_estimator_class': AivatUtilityEstimator,
            'num_iterations': 50,
            'utility_estimator_args': {
                'equilibirum_strategy_path': 'strategies/leduc.limit.2p-equilibrium.strategy'
            }
        })

    def evaluate_agent(self, test_spec):
        portfolio_name = test_spec['portfolio_name']
        portfolio_directory = '%s/%s' % (PORTFOLIOS_DIRECTORY, portfolio_name)

        game_file_path = test_spec['game_file_path']
        game = acpc.read_game_file(game_file_path)
        if game.get_num_players() != 2:
            raise AttributeError('Only games with 2 players are supported')

        response_strategy_paths = []
        opponent_names = []
        opponent_script_paths = []
        for file in os.listdir(portfolio_directory):
            if file.endswith('-response.strategy'):
                response_strategy_paths += [file]
            elif file.endswith('.sh') and not file.startswith(portfolio_name):
                opponent_names += [file[:-len('.sh')]]
                opponent_script_paths += ['%s/%s' % (portfolio_directory, file)]

        response_strategy_paths = list(sorted(response_strategy_paths))
        opponent_names = list(sorted(opponent_names))
        opponent_script_paths = list(sorted(opponent_script_paths))

        portfolio_size = len(response_strategy_paths)

        num_iterations = test_spec['num_iterations']

        overwrite_path = test_spec['overwrite'] if 'overwrite' in test_spec else False
        logs_dir = get_new_path('/'.join([GAME_LOGS_DIRECTORY, '%s-%s' % (portfolio_name, num_iterations)]), overwrite_base_path=overwrite_path)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        big_blind_size = get_big_blind_size(game)

        env = os.environ.copy()
        env['PATH'] = os.path.dirname(sys.executable) + ':' + env['PATH']

        print()
        print('Portfolio strategies order:')
        with open('%s/portfolio_strategies_order.log' % logs_dir, 'w') as file:
            for strategy_path in response_strategy_paths:
                print(os.path.basename(strategy_path))
                file.write(os.path.basename(strategy_path) + '\n')

        # data = np.random.rand(num_iterations, portfolio_size, NUM_EVAL_HANDS, portfolio_size)
        # if True:

        data = np.zeros([num_iterations, portfolio_size, NUM_EVAL_HANDS, portfolio_size])

        print()
        for i in range(portfolio_size):
            for j in range(num_iterations):
                opponent_name = opponent_names[i]
                logs_path = '%s/%s' % (logs_dir, opponent_name)

                proc = subprocess.Popen(
                    [
                        START_DEALER_AND_OPPONENT_SCRIPT_PATH,
                        game_file_path,
                        logs_path,
                        opponent_name,
                        opponent_script_paths[i],
                        portfolio_name],
                    env=env,
                    stdout=subprocess.PIPE)
                port_number = proc.stdout.readline().decode('utf-8').strip()

                client = acpc.Client(game_file_path, '127.0.1.1', port_number)
                full_response_strategy_paths = ['%s/%s' % (portfolio_directory, s) for s in response_strategy_paths]
                utility_estimator_args = test_spec['utility_estimator_args'] if 'utility_estimator_args' in test_spec else None
                client.play(WeightsLoggingImplicitModellingAgent(
                    game_file_path,
                    full_response_strategy_paths,
                    data[j, i, :, :],
                    utility_estimator_class=test_spec['utility_estimator_class'],
                    utility_estimator_args=utility_estimator_args))

                scores_line = proc.stdout.readline().decode('utf-8').strip()
                agent_score = float(scores_line.split(':')[1].split('|')[1])
                agent_score_mbb_per_game = (agent_score / NUM_EVAL_HANDS) * big_blind_size
                print('%s/%s %s vs %s: %s' % (j + 1, num_iterations, portfolio_name, opponent_name, agent_score_mbb_per_game))

            weights_sums = np.sum(data, axis=3, keepdims=True)
            weights_sums[weights_sums == 0] = 1
            weights_sums = np.repeat(weights_sums, portfolio_size, axis=3)

            mean_weights = np.mean(data / weights_sums, axis=0)

            self.assertEqual(mean_weights.shape, (4, 3000, 4))

            parsed_opponent_names = []
            for i in range(portfolio_size):
                original_name = opponent_names[i]
                minus_count = original_name.count('-')
                if minus_count == 4:
                    parsed_name = ' '.join(original_name.split('-')[:3])
                else:
                    parts =original_name.split('-')
                    parsed_name = ' '.join(parts[:2]) + ' -' + parts[3]
                parsed_opponent_names += [parsed_name]

            fig = plt.figure()
            for i in range(portfolio_size):
                # ax = plt.subplot(2, 2, i + 1)
                ax = plt.subplot(200 + 20 + i + 1)

                ax.set_title(parsed_opponent_names[i])
                im = plt.imshow(np.transpose(mean_weights[i]), aspect='auto', cmap=plt.cm.Wistia)
                plt.xlabel('Hand number')
                plt.yticks(np.arange(portfolio_size), [r'$W_%s$' % (i + 1) for i in range(portfolio_size)])
                # plt.yticks(rotation=35)
                # plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                # plt.yticks(np.arange(num_agents), agent_names)

                # plt.tick_params(
                #     axis='x',
                #     which='both',
                #     bottom=False,
                #     top=False,
                #     labelbottom=False)

            plt.tight_layout()

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)

            # plt.tight_layout()
            # plt.gcf().subplots_adjust(left=0.1)

            plot_path = '%s/weights.png' % logs_dir
            plt.savefig(plot_path, dpi=160)
