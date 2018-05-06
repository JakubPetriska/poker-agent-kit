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

import acpc_python_client as acpc


FILES_PATH = 'verification/tournaments'
ACPC_INFRASTRUCTURE_DIR = os.getcwd() + '/../acpc-python-client/acpc_infrastructure'
MATCH_SCRIPT = './play_match.pl'

NUM_TOURNAMENT_HANDS = 3000


class AcpcTournamentTest(unittest.TestCase):
    def test_kuhn_simple_portfolio_tournament(self):
        portfolio_path = 'verification/implicit_agent/portfolios/kuhn_simple_portfolio'
        agents = [
            ('kuhn_simple_portfolio',
                    '%s/agent.sh' % portfolio_path)]
        for file in os.listdir(portfolio_path):
            if not file == 'agent.sh' and file.endswith('.sh'):
                agent_name = file[:-len('.sh')]
                agents += [(agent_name, '/'.join([portfolio_path, file]))]
        self.run_tournament({
            'game_file_path': 'games/kuhn.limit.2p.game',
            'name': 'kuhn_simple_portfolio',
            'agents': agents,
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

        agents = test_spec['agents']
        num_agents = len(agents)
        agent_scripts_paths = [workspace_dir + '/' + agent[1] for agent in agents]

        seed = int(datetime.now().timestamp())

        scores_table = np.zeros([num_agents] * 2)

        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                first_agent_name = agents[i][0]
                second_agent_name = agents[j][0]
                match_name = '%s-vs-%s' % (first_agent_name, second_agent_name)
                match_logs_dir = '%s/%s' % (logs_base_dir, match_name)
                if not os.path.exists(match_logs_dir):
                    os.makedirs(match_logs_dir)

                proc = subprocess.Popen(
                    [
                        MATCH_SCRIPT,
                        '%s/order-normal' % match_logs_dir,
                        game_file_path,
                        str(NUM_TOURNAMENT_HANDS),
                        str(seed),
                        first_agent_name,
                        agent_scripts_paths[i],
                        second_agent_name,
                        agent_scripts_paths[j]],
                    cwd=ACPC_INFRASTRUCTURE_DIR,
                    stdout=subprocess.PIPE)
                scores_line = proc.stdout.readline().decode('utf-8').strip()
                score = float(scores_line.split(':')[1].split('|')[0])
                scores_table[i, j] = score

                proc = subprocess.Popen(
                    [
                        MATCH_SCRIPT,
                        '%s/order-reversed' % match_logs_dir,
                        game_file_path,
                        str(NUM_TOURNAMENT_HANDS),
                        str(seed),
                        second_agent_name,
                        agent_scripts_paths[j],
                        first_agent_name,
                        agent_scripts_paths[i]],
                    cwd=ACPC_INFRASTRUCTURE_DIR,
                    stdout=subprocess.PIPE)
                scores_line = proc.stdout.readline().decode('utf-8').strip()
                score = float(scores_line.split(':')[1].split('|')[0])
                scores_table[j, i] = score

        print()
        print()
        print('All results:')
        agent_names = [agents[i][0] for i in range(num_agents)]
        table = np.zeros([num_agents, num_agents + 1])
        table[:, 1:] = scores_table
        table = table.tolist()
        for i in range(num_agents):
            table[i][0] = agent_names[i]
            table[i][i + 1] = None
        all_results_table_string = tabulate(table, headers=agent_names, tablefmt='grid')
        print(all_results_table_string)

        print()
        print('Averaged results:')
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                table[i][j + 1] = (scores_table[i, j] - scores_table[j, i]) / 2
                table[j][i + 1] = None
        avg_results_table_string = tabulate(table, headers=agent_names, tablefmt='grid')
        print(avg_results_table_string)

        with open('%s/results.log' % logs_base_dir, 'w') as file:
            file.write('All results:\n')
            file.write(all_results_table_string)
            file.write('\n\n')
            file.write('Averaged results:\n')
            file.write(avg_results_table_string)
            file.write('\n')
