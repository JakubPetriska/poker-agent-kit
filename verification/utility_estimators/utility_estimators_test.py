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

from tools.io_util import get_new_path
from tools.match_evaluation import get_player_utilities_from_log_file, get_logs_data, calculate_confidence_interval


FILES_PATH = 'verification/utility_estimators'

ACPC_INFRASTRUCTURE_DIR = os.getcwd() + '/../acpc-python-client/acpc_infrastructure'
MATCH_SCRIPT = './play_match.pl'

class UtilityEstimatorsTest(unittest.TestCase):
    def test_leduc_small_data(self):
        self.run_evaluation({
            'game_file_path': 'games/leduc.limit.2p.game',
            'agents': [
                ('Equilibrium_1', 'strategies/leduc.limit.2p-equilibrium-agent.sh'),
                ('Equilibrium_2', 'strategies/leduc.limit.2p-equilibrium-agent.sh'),
            ],
            'num_matches': 5,
            'num_match_hands': 2000,
        })

    def run_evaluation(self, test_spec):
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

        test_directory = '%s/%s/test-[%s]-%sx%s' % (workspace_dir, FILES_PATH, ';'.join(map(lambda a: a[0], agents)), num_matches, num_match_hands)
        test_data_directory = '%s/data' % test_directory

        data_created = True
        if os.path.exists(test_directory):
            for i in range(num_matches):
                if not os.path.exists('%s/match_%s' % (test_data_directory, i)):
                    data_created = False
                    break
        else:
            data_created = False

        if not data_created:
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


        log_file_paths = []
        for i in range(num_matches):
            log_file_paths += [
                '%s/match_%s/normal.log' % (test_data_directory, i),
                '%s/match_%s/reversed.log' % (test_data_directory, i),
            ]
        log_readings = [
            get_player_utilities_from_log_file(log_file_path)
            for log_file_path in log_file_paths]

        data, _ = get_logs_data(*log_readings)

        output_table = [[None for j in range(3)]]

        output_table[0][0] = 'chips'
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        output_table[0][1] = means[0]
        output_table[0][2] = stds[0]

        print()
        print(tabulate(output_table, headers=['mean', 'SD'], tablefmt='grid'))
