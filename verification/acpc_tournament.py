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
import random

import acpc_python_client as acpc

from tools.io_util import get_new_path
from tools.match_evaluation import get_player_utilities_from_log_file, get_logs_data, calculate_confidence_interval


FILES_PATH = 'verification/tournaments'
ACPC_INFRASTRUCTURE_DIR = os.getcwd() + '/../acpc-python-client/acpc_infrastructure'
MATCH_SCRIPT = './play_match.pl'

NUM_TOURNAMENT_HANDS = 3000


class AcpcTournamentTest(unittest.TestCase):
    def test_kuhn_simple_portfolio_tournament_full(self):
        portfolio_path = 'verification/implicit_agent/portfolios/kuhn_simple_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        agents = implicit_agents + opponent_agents
        self.run_tournament({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'name': 'kuhn_simple_portfolio-full',
            'row_agents': agents,
            'column_agents': agents,
            'confidence': 0.95,
            'max_confidence_interval_half_size': 15 / 1000,
        })

    def test_kuhn_simple_portfolio_tournament_medium(self):
        portfolio_path = 'verification/implicit_agent/portfolios/kuhn_simple_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        self.run_tournament({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'name': 'kuhn_simple_portfolio-medium',
            'row_agents': implicit_agents,
            'column_agents': implicit_agents[1:] + opponent_agents,
            'confidence': 0.95,
            'max_confidence_interval_half_size': 10 / 1000,
        })

    def test_leduc_simple_portfolio_tournament_medium(self):
        portfolio_path = 'verification/implicit_agent/portfolios/leduc_simple_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        self.run_tournament({
            'game_file_path': 'games/leduc.limit.2p.game',
            'name': 'leduc_simple_portfolio-medium',
            'row_agents': implicit_agents,
            'column_agents': implicit_agents[1:] + opponent_agents,
            'confidence': 0.95,
            'max_confidence_interval_half_size': 15 / 1000,
        })

    def test_leduc_small_portfolio_tournament_medium(self):
        portfolio_path = 'verification/implicit_agent/portfolios/leduc_small_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        self.run_tournament({
            'game_file_path': 'games/leduc.limit.2p.game',
            'name': 'leduc_small_portfolio-medium',
            'row_agents': implicit_agents,
            'column_agents': implicit_agents[1:] + opponent_agents,
            'confidence': 0.95,
            'max_confidence_interval_half_size': 15 / 1000,
        })

    def test_leduc_small_portfolio_aivat_tournament(self):
        portfolio_path = 'verification/implicit_agent/portfolios/leduc_small_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        self.run_tournament({
            'game_file_path': 'games/leduc.limit.2p.game',
            'name': 'leduc_small_portfolio-aivat',
            'row_agents': list(filter(lambda a: 'aivat' in a[0], implicit_agents)),
            'column_agents': opponent_agents,
            'confidence': 0.95,
            'max_confidence_interval_half_size': 15 / 1000,
        })

    def test_leduc_small_hard_portfolio_tournament(self):
        portfolio_path = 'verification/implicit_agent/portfolios/leduc_small_hard_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        self.run_tournament({
            'game_file_path': 'games/leduc.limit.2p.game',
            'name': 'leduc_small_hard_portfolio-medium',
            'row_agents': implicit_agents,
            'column_agents': opponent_agents,
            'confidence': 0.95,
            'max_confidence_interval_half_size': 15 / 1000,
        })

    def test_leduc_small_hard_portfolio_aivat_tournament(self):
        portfolio_path = 'verification/implicit_agent/portfolios/leduc_small_hard_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        self.run_tournament({
            'game_file_path': 'games/leduc.limit.2p.game',
            'name': 'leduc_small_hard_portfolio-aivat-medium',
            'row_agents': list(filter(lambda a: 'aivat' in a[0], implicit_agents)),
            'column_agents': opponent_agents,
            'confidence': 0.95,
            'max_confidence_interval_half_size': 15 / 1000,
        })

    def _get_portfolio_agents(self, portfolio_path):
        implicit_agents = []
        opponent_agents = []
        for file in os.listdir(portfolio_path):
            if file.endswith('.sh'):
                agent_name = file[:-len('.sh')]
                if  any(char.isdigit() for char in agent_name):
                    # Weak evaluation agent
                    opponent_agents += [(agent_name, agent_name, '/'.join([portfolio_path, file]))]
                else:
                    # Implicit modelling agent
                    print_agent_name = agent_name.replace('_', ' ').replace('-', '-\n')
                    implicit_agents += [(agent_name, print_agent_name, '/'.join([portfolio_path, file]))]
        return implicit_agents, opponent_agents

    def run_tournament(self, test_spec):
        workspace_dir = os.getcwd()

        game_file_path = workspace_dir + '/' + test_spec['game_file_path']
        game = acpc.read_game_file(game_file_path)
        if game.get_num_players() != 2:
            raise AttributeError('Only games with 2 players are supported')

        tournament_name = test_spec['name']
        confidence = test_spec['confidence']
        max_confidence_interval_half_size = test_spec['max_confidence_interval_half_size']

        logs_base_dir = get_new_path('%s/%s/%s-%s+-%s' % (
            workspace_dir,
            FILES_PATH,
            tournament_name,
            int(confidence * 100),
            int(max_confidence_interval_half_size * 1000)))

        if not os.path.exists(logs_base_dir):
            os.makedirs(logs_base_dir)

        row_agents = test_spec['row_agents']
        row_num_agents = len(row_agents)
        row_agent_scripts_paths = [workspace_dir + '/' + agent[2] for agent in row_agents]

        column_agents = test_spec['column_agents']
        column_num_agents = len(column_agents)
        column_agent_scripts_paths = [workspace_dir + '/' + agent[2] for agent in column_agents]

        seeds = []

        seeds_file_path = '%s/%s/seeds.log' % (workspace_dir, FILES_PATH)
        if not os.path.exists(seeds_file_path):
            max_seed = (2**30) - 1
            for _ in range(5000):
                seeds += [random.randint(1, max_seed)]
            with open(seeds_file_path, 'w') as file:
                for seed in seeds:
                    file.write(str(seed) + '\n')
        else:
            with open(seeds_file_path, 'r') as seeds_file:
                for seed in seeds_file:
                    seeds += [int(float(seed))]

        scores_table = [[None for j in range(column_num_agents)] for i in range(row_num_agents)]

        agent_pairs_evaluated = []

        env = os.environ.copy()
        env['PATH'] = os.path.dirname(sys.executable) + ':' + env['PATH']

        for i in range(row_num_agents):
            for j in range(column_num_agents):
                row_agent_name = row_agents[i][0]
                column_agent_name = column_agents[j][0]
                if row_agent_name == column_agent_name:
                    continue

                agent_pair_key = tuple(sorted([row_agent_name, column_agent_name]))
                if agent_pair_key in agent_pairs_evaluated:
                    continue

                row_agent_script_path = row_agent_scripts_paths[i]
                column_agent_script_path = column_agent_scripts_paths[j]

                match_name = '%s-vs-%s' % (row_agent_name, column_agent_name)
                match_name_reversed = '%s-vs-%s' % (column_agent_name, row_agent_name)
                match_logs_dir = ('%s/%s' % (logs_base_dir, match_name)).replace('\n', '')

                print()
                print('Evaluating %s' % match_name)

                best_confidence_interval_half_size = float('inf')
                row_player_mean_utility = -1
                run_counter = 0
                log_readings = []
                while best_confidence_interval_half_size > max_confidence_interval_half_size:
                    run_counter += 1
                    run_logs_dir = '%s/run_%s' % (match_logs_dir, run_counter)
                    os.makedirs(run_logs_dir)

                    if len(seeds) < run_counter:
                        seeds += [int(datetime.now().timestamp())]
                    seed = seeds[run_counter - 1]

                    normal_order_logs_name = '%s/%s' % (run_logs_dir, match_name)
                    proc = subprocess.Popen(
                        [
                            MATCH_SCRIPT,
                            normal_order_logs_name,
                            game_file_path,
                            str(NUM_TOURNAMENT_HANDS),
                            str(seed),
                            row_agent_name,
                            row_agent_script_path,
                            column_agent_name,
                            column_agent_script_path],
                        cwd=ACPC_INFRASTRUCTURE_DIR,
                        env=env,
                        stdout=subprocess.PIPE)
                    proc.stdout.readline().decode('utf-8').strip()
                    log_readings += [get_player_utilities_from_log_file(normal_order_logs_name + '.log')]

                    reversed_order_logs_name = '%s/%s' % (run_logs_dir, match_name_reversed)
                    proc = subprocess.Popen(
                        [
                            MATCH_SCRIPT,
                            reversed_order_logs_name,
                            game_file_path,
                            str(NUM_TOURNAMENT_HANDS),
                            str(seed),
                            column_agent_name,
                            column_agent_script_path,
                            row_agent_name,
                            row_agent_script_path],
                        cwd=ACPC_INFRASTRUCTURE_DIR,
                        env=env,
                        stdout=subprocess.PIPE)
                    proc.stdout.readline().decode('utf-8').strip()
                    log_readings += [get_player_utilities_from_log_file(reversed_order_logs_name + '.log')]

                    data, player_names = get_logs_data(*log_readings)
                    means, interval_half_size, _, _ = calculate_confidence_interval(data, confidence)

                    print('Run %s, current confidence interval half size: %s' % (run_counter, interval_half_size[0]))
                    best_confidence_interval_half_size = interval_half_size[0]
                    row_player_index = player_names.index(row_agent_name)
                    row_player_mean_utility = means[row_player_index]

                scores_table[i][j] = row_player_mean_utility

                agent_pairs_evaluated += [agent_pair_key]

        print()
        print()
        scores_copy = copy.deepcopy(scores_table)
        for i in range(row_num_agents):
            scores_copy[i] = [row_agents[i][1]] + [None if score is None else score * 1000 for score in scores_copy[i]]
        column_agent_names = [agent[1] for agent in column_agents]
        avg_results_table_string = tabulate(scores_copy, headers=column_agent_names, tablefmt='grid')
        print(avg_results_table_string)

        confidence_line = 'Confidence interval: %s%% +- %s' % (int(confidence * 100), int(max_confidence_interval_half_size * 1000))
        print(confidence_line)

        with open('%s/results.log' % logs_base_dir, 'w') as file:
            file.write(avg_results_table_string)
            file.write('\n')
            file.write('All utilities in mbb/g\n')
            file.write(confidence_line)
            file.write('\n')
