import unittest
import os
import sys
import shutil
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import acpc_python_client as acpc

from tools.constants import Action
from weak_agents.action_tilted_agent import create_agent_strategy_from_trained_strategy, TiltType
from tools.io_util import read_strategy_from_file
from implicit_modelling.build_portfolio import build_portfolio
from tools.io_util import write_strategy_to_file
from implicit_modelling.implicit_modelling_agent import ImplicitModellingAgent


TEST_DIRECTORY = 'verification/implicit_agent'
PORTFOLIOS_DIRECTORY = '%s/portfolios' % TEST_DIRECTORY
GAME_LOGS_DIRECTORY = '%s/logs' % TEST_DIRECTORY

BASE_AGENT_SCRIPT_PATH = '%s/base_agent_script.sh' % TEST_DIRECTORY
BASE_OPPONENT_SCRIPT_PATH = '%s/base_opponent_script.sh' % TEST_DIRECTORY

EVALUATE_SCRIPT_PATH = './implicit_modelling/evaluate.sh'

START_DEALER_AND_OPPONENT_SCRIPT_PATH = './scripts/start_dealer_and_other_players.sh'

REPLACE_STRING_COMMENT = '###COMMENT###'
REPLACE_STRING_GAME_FILE_PATH = '###GAME_FILE_PATH###'
REPLACE_STRING_ENVIRONMENT_ACTIVATION = '###ENVIRONMENT_ACTIVATION###'
OPPONENT_SCRIPT_REPLACE_STRING_COMMENT_FILE_PATH = '###STRATEGY_FILE_PATH###'
AGENT_SCRIPT_PORTFOLIO_STRATEGIES_PATHS = "###PORTFOLIO_STRATEGIES_PATHS###"

OPPONENT_SCRIPT_REPLACE_STRINGS = [
    REPLACE_STRING_COMMENT,
    REPLACE_STRING_GAME_FILE_PATH,
    OPPONENT_SCRIPT_REPLACE_STRING_COMMENT_FILE_PATH]

AGENT_SCRIPT_REPLACE_STRINGS = [
    REPLACE_STRING_COMMENT,
    REPLACE_STRING_GAME_FILE_PATH,
    AGENT_SCRIPT_PORTFOLIO_STRATEGIES_PATHS]

WARNING_COMMENT = 'This file is generated. Do not edit!'


def replace_in_file(filename, old_strings, new_strings):
    with open(filename) as f:
        s = f.read()

    with open(filename, 'w') as f:
        for i in range(len(old_strings)):
            s = s.replace(old_strings[i], new_strings[i])
        f.write(s)


class TestMode(Enum):
    EVAL = 0
    DEBUG = 1


class ImplicitAgentTest(unittest.TestCase):
    def test_kuhn_simple_portfolio(self):
        self.evaluate_agent({
            'portfolio_name': 'kuhn_simple_portfolio',
            'game_file_path': 'games/kuhn.limit.2p.game',
            'test_mode': TestMode.DEBUG,
            # 'test_mode': TestMode.EVAL,
        })

    def evaluate_agent(self, test_spec):
        portfolio_name = test_spec['portfolio_name']
        portfolio_directory = '%s/%s' % (PORTFOLIOS_DIRECTORY, portfolio_name)

        game_file_path = test_spec['game_file_path']

        anaconda_env_name = None
        if 'anaconda3/envs' in sys.executable:
            anaconda_env_name = sys.executable.split('/anaconda3/envs/')[1].split('/')[0]

        response_strategy_paths = []
        opponent_strategy_paths = []
        opponent_names = []
        opponent_script_paths = []
        for file in os.listdir(portfolio_directory):
            if file.endswith('-response.strategy'):
                response_strategy_paths += [file]
            elif file.endswith('-opponent.strategy'):
                opponent_strategy_paths += [file]
                opponent_name = file[:-len('-opponent.strategy')]
                opponent_names += [opponent_name]

                opponent_script_path = '%s/%s.sh' % (portfolio_directory, opponent_name)
                opponent_script_paths += [opponent_script_path]
                shutil.copy(BASE_OPPONENT_SCRIPT_PATH, opponent_script_path)
                replace_in_file(
                    opponent_script_path,
                    OPPONENT_SCRIPT_REPLACE_STRINGS,
                    [
                        WARNING_COMMENT,
                        game_file_path,
                        '%s/%s' % (portfolio_directory, file)])
                if anaconda_env_name:
                    replace_in_file(
                        opponent_script_path,
                        [REPLACE_STRING_ENVIRONMENT_ACTIVATION],
                        ['source activate %s' % anaconda_env_name])
        agent_script_path = '%s/agent.sh' % portfolio_directory
        shutil.copy(BASE_AGENT_SCRIPT_PATH, agent_script_path)

        portfolio_size = len(response_strategy_paths)

        strategies_replacement = ''
        for i in range(portfolio_size):
            strategies_replacement += '        "${WORKSPACE_DIR}/%s/%s"' % (portfolio_directory, response_strategy_paths[i])
            if i < (portfolio_size - 1):
                strategies_replacement += ' \\\n'
        replace_in_file(
            agent_script_path,
            AGENT_SCRIPT_REPLACE_STRINGS,
            [
                WARNING_COMMENT,
                game_file_path,
                strategies_replacement])
        if anaconda_env_name:
            replace_in_file(
                agent_script_path,
                [REPLACE_STRING_ENVIRONMENT_ACTIVATION],
                ['source activate %s' % anaconda_env_name])

        portfolio_size = len(response_strategy_paths)

        logs_dir = '/'.join([GAME_LOGS_DIRECTORY, portfolio_name])
        if os.path.exists(logs_dir):
            shutil.rmtree(logs_dir)
        os.makedirs(logs_dir)

        test_mode = test_spec['test_mode']
        for i in range(portfolio_size):
            opponent_name = opponent_names[i]
            logs_path = '%s/%s' % (logs_dir, opponent_name)

            if test_mode == TestMode.EVAL:
                rc = subprocess.call([
                    EVALUATE_SCRIPT_PATH,
                    game_file_path,
                    logs_path,
                    portfolio_name,
                    agent_script_path,
                    opponent_name,
                    opponent_script_paths[i]])
                self.assertEqual(rc, 0)
            elif test_mode == TestMode.DEBUG:
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
                client.play(ImplicitModellingAgent(game_file_path, full_response_strategy_paths))
