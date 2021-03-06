import sys
import numpy as np
import os

import acpc_python_client as acpc

from tools.constants import NUM_ACTIONS
from tools.io_util import read_strategy_from_file
from tools.agent_utils import get_info_set, select_action, convert_action_to_int
from implicit_modelling.strategies_weighted_mixeture import StrategiesWeightedMixture
from implicit_modelling.exp3g import Exp3G
from utility_estimation.simple import SimpleUtilityEstimator
from utility_estimation.imaginary_observations import ImaginaryObservationsUtilityEstimator
from utility_estimation.aivat import AivatUtilityEstimator


class ImplicitModellingAgent(acpc.Agent):
    def __init__(
            self,
            game_file_path,
            portfolio_strategy_files_paths,
            exp3g_gamma=0.02,
            exp3g_eta=0.025,
            utility_estimator_class=SimpleUtilityEstimator,
            utility_estimator_args=None):
        super().__init__()
        self.portfolio_size = len(portfolio_strategy_files_paths)
        self.bandit_algorithm = Exp3G(exp3g_gamma, exp3g_eta, self.portfolio_size)
        game = acpc.read_game_file(game_file_path)
        if utility_estimator_args is None:
            self.utility_estimator = utility_estimator_class(game, True)
        else:
            self.utility_estimator = utility_estimator_class(game, True, **utility_estimator_args)

        self.portfolio_trees = []
        self.portfolio_dicts = []
        for portfolio_strategy_file_path in portfolio_strategy_files_paths:
            strategy_tree, strategy_dict = read_strategy_from_file(game_file_path, portfolio_strategy_file_path)
            self.portfolio_trees += [strategy_tree]
            self.portfolio_dicts += [strategy_dict]

        self.portfolio_strategies_mixture = StrategiesWeightedMixture(game, self.portfolio_trees)

    def on_game_start(self, game):
        pass

    def _get_info_set_current_strategy(self, info_set):
        experts_weights = self.bandit_algorithm.get_current_expert_probabilities()
        current_strategy = np.zeros(NUM_ACTIONS)
        for i in range(self.portfolio_size):
            current_strategy += experts_weights[i] * np.array(self.portfolio_dicts[i][info_set])
        return current_strategy

    def on_next_turn(self, game, match_state, is_acting_player):
        if not is_acting_player:
            return

        info_set = get_info_set(game, match_state)
        current_strategy = self._get_info_set_current_strategy(info_set)
        selected_action = select_action(current_strategy)
        self.set_next_action(selected_action)

    def on_game_finished(self, game, match_state):
        expert_probabilities = self.bandit_algorithm.get_current_expert_probabilities()
        self.portfolio_strategies_mixture.update_weights(expert_probabilities)
        expert_utility_estimates = self.utility_estimator.get_utility_estimations(
            match_state.get_state(),
            match_state.get_viewing_player(),
            self.portfolio_strategies_mixture.strategy,
            self.portfolio_trees)
        if expert_utility_estimates is not None:
            self.bandit_algorithm.update_weights(expert_utility_estimates)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage {game_file_path} {dealer_hostname} {dealer_port} [none|imaginary_observations] *{portfolio_strategy_files_paths}")
        sys.exit(1)

    utility_estimator_type = sys.argv[4]
    utility_estimator_class = None
    utility_estimator_args = None
    if utility_estimator_type == 'none':
        utility_estimator_class = SimpleUtilityEstimator
    elif utility_estimator_type == 'imaginary_observations':
        utility_estimator_class = ImaginaryObservationsUtilityEstimator
    elif utility_estimator_type == 'aivat':
        utility_estimator_class = AivatUtilityEstimator

        game_name = os.path.basename(sys.argv[1])[:-len('.game')]
        equilibrium_strategy_path = '%s/../strategies/%s-equilibrium.strategy' % (os.path.dirname(sys.argv[1]), game_name)
        utility_estimator_args = {
            'equilibirum_strategy_path': equilibrium_strategy_path,
        }
    else:
        raise AttributeError('Invalid utility estimation method type %s' % utility_estimator_type)

    client = acpc.Client(sys.argv[1], sys.argv[2], sys.argv[3])
    client.play(ImplicitModellingAgent(
        sys.argv[1],
        sys.argv[5:],
        utility_estimator_class=utility_estimator_class,
        utility_estimator_args=utility_estimator_args))
