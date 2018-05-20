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
import multiprocessing

import acpc_python_client as acpc

from tools.constants import Action
from weak_agents.action_tilted_agent import create_agent_strategy_from_trained_strategy, TiltType
from tools.io_util import get_new_path, read_strategy_from_file, write_strategy_to_file
from tools.match_evaluation import get_player_utilities_from_log_file, get_logs_data, calculate_confidence_interval
from tools.strategy_agent import StrategyAgent
from utility_estimation.simple import SimpleUtilityEstimator
from utility_estimation.imaginary_observations import ImaginaryObservationsUtilityEstimator
from utility_estimation.aivat import AivatUtilityEstimator


FILES_PATH = 'verification/utility_estimators_off_policy'

KUHN_EQUILIBRIUM_STRATEGY_PATH = 'strategies/kuhn.limit.2p-equilibrium.strategy'
LEDUC_EQUILIBRIUM_STRATEGY_PATH = 'strategies/leduc.limit.2p-equilibrium.strategy'

ACPC_INFRASTRUCTURE_DIR = os.getcwd() + '/../acpc-python-client'
MATCH_SCRIPT = './scripts/start_dealer.pl'


def _get_agent_name(agent):
    return '%s-%s-%s' % (str(agent[0]).split('.')[1], str(agent[1]).split('.')[1], agent[2])


def _run_agent(args):
    client = acpc.Client(args[0], '127.0.1.1', args[1])
    client.play(StrategyAgent(args[2]))


class UtilityEstimatorsOffPolicyTest(unittest.TestCase):
    def test_kuhn_small_data(self):
        self.run_evaluation({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'test_name': 'kuhn_small_data',
            'base_agent': ('Equilibrium', KUHN_EQUILIBRIUM_STRATEGY_PATH),
            'base_validation_agents_strategy_path': KUHN_EQUILIBRIUM_STRATEGY_PATH,
            'validation_agents': [
                (Action.FOLD, TiltType.ADD, 0.5),
                (Action.CALL, TiltType.ADD, 0.5),
                (Action.RAISE, TiltType.ADD, 0.75),
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
            'test_name': 'leduc_small_data',
            'base_agent': ('Equilibrium', LEDUC_EQUILIBRIUM_STRATEGY_PATH),
            'base_validation_agents_strategy_path': LEDUC_EQUILIBRIUM_STRATEGY_PATH,
            'validation_agents': [
                (Action.FOLD, TiltType.ADD, 0.5),
                (Action.CALL, TiltType.ADD, 0.5),
                (Action.RAISE, TiltType.ADD, 0.75),
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

    def test_leduc_big_data(self):
        self.run_evaluation({
            'game_file_path': 'games/leduc.limit.2p.game',
            'test_name': 'leduc_big_data',
            'base_agent': ('Equilibrium', LEDUC_EQUILIBRIUM_STRATEGY_PATH),
            'base_validation_agents_strategy_path': LEDUC_EQUILIBRIUM_STRATEGY_PATH,
            'validation_agents': [
                (Action.FOLD, TiltType.ADD, 0.5),
                (Action.CALL, TiltType.ADD, 0.5),
                (Action.RAISE, TiltType.ADD, 0.75),
            ],
            'num_matches': 25,
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

        test_name = test_spec['test_name']
        base_agent = test_spec['base_agent']
        validation_agents = test_spec['validation_agents']
        num_matches = test_spec['num_matches']
        num_match_hands = test_spec['num_match_hands']

        game_name = game_file_path.split('/')[-1][:-len('.game')]

        validation_agent_names = [_get_agent_name(agent) for agent in validation_agents]

        test_directory = '%s/%s/%s' % (workspace_dir, FILES_PATH, test_name)
        agents_data_directories = []
        for validation_agent in validation_agents:
            agent_data_dir = '%s/%s-[%s;%s]-%sx%s' % (
                test_directory,
                game_name,
                base_agent[0],
                _get_agent_name(validation_agent),
                num_matches,
                num_match_hands)
            agents_data_directories += [agent_data_dir]

        force_recreate_data = test_spec['force_recreate_data'] if 'force_recreate_data' in test_spec else False

        base_validation_agent_strategy = None

        validation_agent_strategies = []

        for x in range(len(validation_agents)):
            agent_data_directory = agents_data_directories[x]
            validation_agent = validation_agents[x]
            data_created = True
            if not force_recreate_data:
                if os.path.exists(agent_data_directory):
                    for i in range(num_matches):
                        match_dir = '%s/match_%s' % (agent_data_directory, i)
                        if not os.path.exists(match_dir) or len(os.listdir(match_dir)) == 0:
                            data_created = False
                            break
                else:
                    data_created = False

            if base_validation_agent_strategy is None:
                base_validation_agent_strategy, _ = read_strategy_from_file(
                    game_file_path,
                    test_spec['base_validation_agents_strategy_path'])

            validation_agent_strategy = create_agent_strategy_from_trained_strategy(
                game_file_path,
                base_validation_agent_strategy,
                validation_agent[0],
                validation_agent[1],
                validation_agent[2])

            validation_agent_strategies += [validation_agent_strategy]

            if not data_created or force_recreate_data:
                if os.path.exists(agent_data_directory):
                    shutil.rmtree(agent_data_directory)

                validation_agent_strategy_path = '%s/%s.strategy' % (test_directory, _get_agent_name(validation_agent))

                write_strategy_to_file(validation_agent_strategy, validation_agent_strategy_path)

                for i in range(num_matches):
                    match_data_dir = '%s/match_%s' % (agent_data_directory, i)
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
                            base_agent[0],
                            _get_agent_name(validation_agent),
                        ],
                        cwd=ACPC_INFRASTRUCTURE_DIR,
                        env=env,
                        stdout=subprocess.PIPE)
                    ports_string = proc.stdout.readline().decode('utf-8').strip()
                    ports = ports_string.split(' ')

                    args = [
                        (game_file_path, ports[0], base_agent[1]),
                        (game_file_path, ports[1], validation_agent_strategy_path),
                    ]

                    with multiprocessing.Pool(2) as p:
                        p.map(_run_agent, args)


                    proc = subprocess.Popen(
                        [
                            MATCH_SCRIPT,
                            '%s/reversed' % match_data_dir,
                            game_file_path,
                            str(num_match_hands),
                            str(seed),
                            _get_agent_name(validation_agent),
                            base_agent[0],
                        ],
                        cwd=ACPC_INFRASTRUCTURE_DIR,
                        env=env,
                        stdout=subprocess.PIPE)
                    ports_string = proc.stdout.readline().decode('utf-8').strip()
                    ports = ports_string.split(' ')

                    args = [
                        (game_file_path, ports[0], validation_agent_strategy_path),
                        (game_file_path, ports[1], base_agent[1]),
                    ]

                    with multiprocessing.Pool(2) as p:
                        p.map(_run_agent, args)


                print('Data created')

        output = []

        def prin(string=''):
            nonlocal output
            output += [string]
            print(string)

        utility_estimators = test_spec['utility_estimators']

        agents_log_files_paths = []
        for x in range(len(validation_agents)):
            agents_data_directory = agents_data_directories[x]
            log_file_paths = []
            for i in range(num_matches):
                log_file_paths += [
                    '%s/match_%s/normal.log' % (agents_data_directory, i),
                    '%s/match_%s/reversed.log' % (agents_data_directory, i),
                ]
            agents_log_files_paths += [log_file_paths]

        agent_strategies = {}
        for i in range(len(validation_agents)):
            agent_strategies[validation_agent_names[i]] = validation_agent_strategies[i]

        prin('Cell contains utility of row player based on observation of column player')
        for utility_estimator_spec in utility_estimators:
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

            prin()
            prin('%s (mean | SD)' % utility_estimator_name)

            output_table = [[None for j in range(len(validation_agents) + 1)] for i in range(len(validation_agents))]
            for i in range(len(validation_agents)):
                output_table[i][0] = validation_agent_names[i]
            for x in range(len(validation_agents)):
                log_readings = [
                    get_player_utilities_from_log_file(
                        log_file_path,
                        game_file_path=game_file_path,
                        utility_estimator=utility_estimator_instance,
                        player_strategies=agent_strategies,
                        evaluated_strategies=validation_agent_strategies)
                    for log_file_path in agents_log_files_paths[x]]

                data, player_names = get_logs_data(*log_readings)
                means = np.mean(data, axis=0)
                stds = np.std(data, axis=0)

                player_index = player_names.index(validation_agent_names[x])
                for y in range(len(validation_agents)):
                    output_table[y][x + 1] = '%s | %s' % (means[player_index][y], stds[player_index][y])

            prin(tabulate(output_table, headers=validation_agent_names, tablefmt='grid'))

        prin()
        prin('Total num hands: %s' % data.shape[0])

        output_log_path = get_new_path(
            '%s/output-%sx%s' % (test_directory, num_matches, num_match_hands),
            '.log')
        with open(output_log_path, 'w') as file:
            for line in output:
                file.write(line + '\n')
