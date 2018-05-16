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
from tools.io_util import read_strategy_from_file
from implicit_modelling.build_portfolio import build_portfolio
from tools.io_util import write_strategy_to_file
from implicit_modelling.implicit_modelling_agent import ImplicitModellingAgent
from tools.game_utils import get_big_blind_size
from utility_estimation.simple import SimpleUtilityEstimator
from utility_estimation.imaginary_observations import ImaginaryObservationsUtilityEstimator


TEST_DIRECTORY = 'verification/implicit_agent'
PORTFOLIOS_DIRECTORY = '%s/portfolios' % TEST_DIRECTORY
GAME_LOGS_DIRECTORY = '%s/logs' % TEST_DIRECTORY

START_DEALER_AND_OPPONENT_SCRIPT_PATH = './scripts/start_dealer_and_other_players.sh'

NUM_EVAL_HANDS = 3000


class ImplicitAgentTest(unittest.TestCase):
    def test_kuhn_simple_portfolio(self):
        self.evaluate_agent({
            'portfolio_name': 'kuhn_simple_portfolio',
            'game_file_path': 'games/kuhn.limit.2p.game',
            'utility_estimator_class': SimpleUtilityEstimator,
        })

    def test_kuhn_simple_portfolio_imaginary_observations(self):
        self.evaluate_agent({
            'portfolio_name': 'kuhn_simple_portfolio',
            'game_file_path': 'games/kuhn.limit.2p.game',
            'utility_estimator_class': ImaginaryObservationsUtilityEstimator,
        })

    def test_leduc_simple_portfolio(self):
        self.evaluate_agent({
            'portfolio_name': 'leduc_simple_portfolio',
            'game_file_path': 'games/leduc.limit.2p.game',
            'utility_estimator_class': SimpleUtilityEstimator,
        })

    def test_leduc_simple_portfolio_imaginary_observations(self):
        self.evaluate_agent({
            'portfolio_name': 'leduc_simple_portfolio',
            'game_file_path': 'games/leduc.limit.2p.game',
            'utility_estimator_class': ImaginaryObservationsUtilityEstimator,
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

        portfolio_size = len(response_strategy_paths)

        logs_dir = '/'.join([GAME_LOGS_DIRECTORY, portfolio_name])
        if os.path.exists(logs_dir):
            shutil.rmtree(logs_dir)
        os.makedirs(logs_dir)

        big_blind_size = get_big_blind_size(game)

        print()
        for i in range(portfolio_size):
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
                stdout=subprocess.PIPE)
            port_number = proc.stdout.readline().decode('utf-8').strip()

            client = acpc.Client(game_file_path, '127.0.1.1', port_number)

            full_response_strategy_paths = ['%s/%s' % (portfolio_directory, s) for s in response_strategy_paths]
            client.play(ImplicitModellingAgent(game_file_path, full_response_strategy_paths, utility_estimator_class=test_spec['utility_estimator_class']))

            scores_line = proc.stdout.readline().decode('utf-8').strip()
            agent_score = float(scores_line.split(':')[1].split('|')[1])
            agent_score_mbb_per_game = (agent_score / NUM_EVAL_HANDS) * big_blind_size
            print('%s vs %s: %s' % (portfolio_name, opponent_name, agent_score_mbb_per_game))
