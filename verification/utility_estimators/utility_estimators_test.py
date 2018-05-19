import unittest
import os
import sys
import shutil
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate
import copy

import acpc_python_client as acpc

from tools.io_util import get_new_path, read_strategy_from_file
from tools.match_evaluation import get_player_utilities_from_log_file, get_logs_data, calculate_confidence_interval
from utility_estimation.simple import SimpleUtilityEstimator
from utility_estimation.imaginary_observations import ImaginaryObservationsUtilityEstimator
from utility_estimation.aivat import AivatUtilityEstimator


FILES_PATH = 'verification/utility_estimators'

ACPC_INFRASTRUCTURE_DIR = os.getcwd() + '/../acpc-python-client/acpc_infrastructure'
MATCH_SCRIPT = './play_match.pl'

class UtilityEstimatorsTest(unittest.TestCase):
    def test_kuhn_small_data(self):
        self.run_evaluation({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'agents': [
                ('Equilibrium_1', 'strategies/kuhn.limit.2p-equilibrium-agent.sh', 'strategies/kuhn.limit.2p-equilibrium.strategy'),
                ('Equilibrium_2', 'strategies/kuhn.limit.2p-equilibrium-agent.sh'),
            ],
            'num_matches': 5,
            'num_match_hands': 2000,
            'utility_estimators': [
                ('chips', SimpleUtilityEstimator),
                ('imaginary_observations', ImaginaryObservationsUtilityEstimator),
                ('AIVAT', AivatUtilityEstimator, {
                    'equilibirum_strategy_path': 'strategies/kuhn.limit.2p-equilibrium.strategy'
                }),
            ],
            # 'force_recreate_data': True,
        })

    def test_leduc_small_data(self):
        self.run_evaluation({
            'game_file_path': 'games/leduc.limit.2p.game',
            'agents': [
                ('Equilibrium_1', 'strategies/leduc.limit.2p-equilibrium-agent.sh', 'strategies/leduc.limit.2p-equilibrium.strategy'),
                ('Equilibrium_2', 'strategies/leduc.limit.2p-equilibrium-agent.sh'),
            ],
            'num_matches': 5,
            'num_match_hands': 2000,
            'utility_estimators': [
                ('chips', SimpleUtilityEstimator),
                ('imaginary_observations', ImaginaryObservationsUtilityEstimator),
                ('AIVAT', AivatUtilityEstimator, {
                    'equilibirum_strategy_path': 'strategies/leduc.limit.2p-equilibrium.strategy'
                }),
            ],
            # 'force_recreate_data': True,
        })

    def run_evaluation(self, test_spec):
        print()

        workspace_dir = os.getcwd()

        game_file_path = workspace_dir + '/' + test_spec['game_file_path']
        game = acpc.read_game_file(game_file_path)

        if game.get_num_players() != 2:
            raise AttributeError('Only games with 2 players are supported')

        agents = test_spec['agents']
        num_matches = test_spec['num_matches']
        num_match_hands = test_spec['num_match_hands']

        if game.get_num_players() != len(agents):
            raise AttributeError('Wrong number of players')

        game_name = game_file_path.split('/')[-1][:-len('.game')]

        test_directory = '%s/%s/test-%s-[%s]-%sx%s' % (workspace_dir, FILES_PATH, game_name, ';'.join(map(lambda a: a[0], agents)), num_matches, num_match_hands)
        test_data_directory = '%s/data' % test_directory

        force_recreate_data = test_spec['force_recreate_data'] if 'force_recreate_data' in test_spec else False
        data_created = True
        if not force_recreate_data:
            if os.path.exists(test_directory):
                for i in range(num_matches):
                    if not os.path.exists('%s/match_%s' % (test_data_directory, i)):
                        data_created = False
                        break
            else:
                data_created = False

        if not data_created or force_recreate_data:
            if os.path.exists(test_data_directory):
                shutil.rmtree(test_data_directory)
            for i in range(num_matches):
                match_data_dir = '%s/match_%s' % (test_data_directory, i)
                if not os.path.exists(match_data_dir):
                    os.makedirs(match_data_dir)

                seed = int(datetime.now().timestamp())

                env = os.environ.copy()
                env['PATH'] = os.path.dirname(sys.executable) + ':' + env['PATH']

                proc = subprocess.Popen(
                    [
                        MATCH_SCRIPT,
                        '%s/normal' % match_data_dir,
                        game_file_path,
                        str(num_match_hands),
                        str(seed),
                        agents[0][0],
                        workspace_dir + '/' + agents[0][1],
                        agents[1][0],
                        workspace_dir + '/' + agents[1][1],
                    ],
                    cwd=ACPC_INFRASTRUCTURE_DIR,
                    env=env,
                    stdout=subprocess.PIPE)
                proc.stdout.readline().decode('utf-8').strip()

                proc = subprocess.Popen(
                    [
                        MATCH_SCRIPT,
                        '%s/reversed' % match_data_dir,
                        game_file_path,
                        str(num_match_hands),
                        str(seed),
                        agents[1][0],
                        workspace_dir + '/' + agents[1][1],
                        agents[0][0],
                        workspace_dir + '/' + agents[0][1],
                    ],
                    cwd=ACPC_INFRASTRUCTURE_DIR,
                    env=env,
                    stdout=subprocess.PIPE)
                proc.stdout.readline().decode('utf-8').strip()

            print('Data created')

        log_file_paths = []
        for i in range(num_matches):
            log_file_paths += [
                '%s/match_%s/normal.log' % (test_data_directory, i),
                '%s/match_%s/reversed.log' % (test_data_directory, i),
            ]

        agent_strategies = {}
        for agent in agents:
            if len(agent) >= 3:
                strategy, _ = read_strategy_from_file(game_file_path, agent[2])
                agent_strategies[agent[0]] = strategy

        utility_estimators = test_spec['utility_estimators']
        output_table = [[None for j in range(3)] for i in range(len(utility_estimators))]
        for i, utility_estimator_spec in enumerate(utility_estimators):
            utility_estimator_name = utility_estimator_spec[0]
            utility_estimator_class = utility_estimator_spec[1]
            utility_estimator_instance = None
            if utility_estimator_class is not None:
                if len(utility_estimator_spec) == 2:
                    utility_estimator_instance = utility_estimator_class(game, False)
                elif len(utility_estimator_spec) > 2:
                    utility_estimator_args = utility_estimator_spec[2]
                    utility_estimator_instance = utility_estimator_class(
                        game,
                        False,
                        **utility_estimator_args)
            log_readings = [
                get_player_utilities_from_log_file(
                    log_file_path,
                    game_file_path=game_file_path,
                    utility_estimator=utility_estimator_instance,
                    player_strategies=agent_strategies)
                for log_file_path in log_file_paths]

            data, player_names = get_logs_data(*log_readings)
            player_zero_index = player_names.index(agents[0][0])

            output_table[i][0] = utility_estimator_name
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)

            output_table[i][1] = means[player_zero_index]
            output_table[i][2] = stds[player_zero_index]

        print()
        print(tabulate(output_table, headers=['mean', 'SD'], tablefmt='grid'))
