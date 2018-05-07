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


FILES_PATH = 'verification/tournaments'
ACPC_INFRASTRUCTURE_DIR = os.getcwd() + '/../acpc-python-client/acpc_infrastructure'
MATCH_SCRIPT = './play_match.pl'

NUM_TOURNAMENT_HANDS = 3000


class AcpcTournamentTest(unittest.TestCase):
    def _get_portfolio_agents(self, portfolio_path):
        implicit_agents = []
        opponent_agents = []
        for file in os.listdir(portfolio_path):
            if file.endswith('.sh'):
                agent_name = file[:-len('.sh')]
                if agent_name[-1].isdigit():
                    # Weak evaluation agent
                    opponent_agents += [(agent_name, agent_name, '/'.join([portfolio_path, file]))]
                else:
                    # Implicit modelling agent
                    print_agent_name = agent_name.replace('_', ' ').replace('-', '-\n')
                    implicit_agents += [(agent_name, print_agent_name, '/'.join([portfolio_path, file]))]
        return implicit_agents, opponent_agents

    def test_kuhn_simple_portfolio_tournament_full(self):
        portfolio_path = 'verification/implicit_agent/portfolios/kuhn_simple_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        agents = implicit_agents + opponent_agents
        self.run_tournament({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'name': 'kuhn_simple_portfolio-full',
            'row_agents': agents,
            'column_agents': agents,
        })

    def test_kuhn_simple_portfolio_tournament_medium(self):
        portfolio_path = 'verification/implicit_agent/portfolios/kuhn_simple_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        self.run_tournament({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'name': 'kuhn_simple_portfolio-medium',
            'row_agents': implicit_agents,
            'column_agents': implicit_agents[1:] + opponent_agents,
        })

    def test_kuhn_simple_portfolio_tournament_short(self):
        portfolio_path = 'verification/implicit_agent/portfolios/kuhn_simple_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        self.run_tournament({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'name': 'kuhn_simple_portfolio-short',
            'row_agents': implicit_agents,
            'column_agents': opponent_agents,
        })

    def test_leduc_simple_portfolio_tournament_medium(self):
        portfolio_path = 'verification/implicit_agent/portfolios/leduc_simple_portfolio'
        implicit_agents, opponent_agents = self._get_portfolio_agents(portfolio_path)
        self.run_tournament({
            'game_file_path': 'games/leduc.limit.2p.game',
            'name': 'leduc_simple_portfolio-medium',
            'row_agents': implicit_agents,
            'column_agents': implicit_agents[1:] + opponent_agents,
        })

    def run_tournament(self, test_spec):
        workspace_dir = os.getcwd()

        game_file_path = workspace_dir + '/' + test_spec['game_file_path']
        game = acpc.read_game_file(game_file_path)
        if game.get_num_players() != 2:
            raise AttributeError('Only games with 2 players are supported')

        tournament_name = test_spec['name']

        logs_base_dir = '/'.join([workspace_dir, FILES_PATH, tournament_name])
        if os.path.exists(logs_base_dir):
            shutil.rmtree(logs_base_dir)

        row_agents = test_spec['row_agents']
        row_num_agents = len(row_agents)
        row_agent_scripts_paths = [workspace_dir + '/' + agent[2] for agent in row_agents]

        column_agents = test_spec['column_agents']
        column_num_agents = len(column_agents)
        column_agent_scripts_paths = [workspace_dir + '/' + agent[2] for agent in column_agents]

        seed = int(datetime.now().timestamp())

        scores_table = [[None for j in range(column_num_agents)] for i in range(row_num_agents)]

        agent_pairs_evaluated = []

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
                match_logs_dir = ('%s/%s' % (logs_base_dir, match_name)).replace('\n', '')
                if os.path.exists(match_logs_dir):
                    shutil.rmtree(match_logs_dir)
                os.makedirs(match_logs_dir)

                proc = subprocess.Popen(
                    [
                        MATCH_SCRIPT,
                        '%s/order-normal' % match_logs_dir,
                        game_file_path,
                        str(NUM_TOURNAMENT_HANDS),
                        str(seed),
                        row_agent_name,
                        row_agent_script_path,
                        column_agent_name,
                        column_agent_script_path],
                    cwd=ACPC_INFRASTRUCTURE_DIR,
                    stdout=subprocess.PIPE)
                scores_line = proc.stdout.readline().decode('utf-8').strip()
                row_agent_score = float(scores_line.split(':')[1].split('|')[0])

                proc = subprocess.Popen(
                    [
                        MATCH_SCRIPT,
                        '%s/order-reversed' % match_logs_dir,
                        game_file_path,
                        str(NUM_TOURNAMENT_HANDS),
                        str(seed),
                        column_agent_name,
                        column_agent_script_path,
                        row_agent_name,
                        row_agent_script_path],
                    cwd=ACPC_INFRASTRUCTURE_DIR,
                    stdout=subprocess.PIPE)
                scores_line = proc.stdout.readline().decode('utf-8').strip()
                row_agent_score += float(scores_line.split(':')[1].split('|')[1])
                scores_table[i][j] = row_agent_score / 2

                agent_pairs_evaluated += [agent_pair_key]

        print()
        print()
        scores_copy = copy.deepcopy(scores_table)
        for i in range(row_num_agents):
            scores_copy[i] = [row_agents[i][1]] + scores_copy[i]
        column_agent_names = [agent[1] for agent in column_agents]
        avg_results_table_string = tabulate(scores_copy, headers=column_agent_names, tablefmt='grid')
        print(avg_results_table_string)

        with open('%s/results.log' % logs_base_dir, 'w') as file:
            file.write(avg_results_table_string)
            file.write('\n')
