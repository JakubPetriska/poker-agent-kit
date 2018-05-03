import sys
import numpy as np
import matplotlib.pyplot as plt

import acpc_python_client as acpc

from response.restricted_nash_response import RestrictedNashResponse
from evaluation.exploitability import Exploitability
from tools.game_utils import is_strategies_equal


def build_portfolio(
        game_file_path,
        opponent_strategy_trees,
        rnr_params,
        portfolio_size=-1,
        portfolio_cut_improvement_threshold=0.05,
        log=False):
    if portfolio_size <= 0 \
        and not portfolio_cut_improvement_threshold or portfolio_cut_improvement_threshold <= 0:
        raise AttributeError('Either portfolio_size or portfolio_cut_improvement_threshold larger than 0 must be provided')

    num_opponents = len(opponent_strategy_trees)

    game = acpc.read_game_file(game_file_path)
    exp = Exploitability(game)

    responses = []
    for i in range(num_opponents):
        current_rnr_params = rnr_params[i]
        rnr = RestrictedNashResponse(game, opponent_strategy_trees[i], current_rnr_params[0], show_progress=log)
        response_strategy = None
        if len(current_rnr_params) > 2:
            response_strategy = rnr.train(current_rnr_params[1], weight_delay=current_rnr_params[2])
        else:
            response_strategy = rnr.train(current_rnr_params[1])
        responses += [response_strategy]

    utilities = np.zeros([num_opponents, num_opponents])
    for i in range(num_opponents):
        for j in range(num_opponents):
            utilities[i, j] = exp.evaluate(opponent_strategy_trees[j], responses[i])

    portfolio_utilities = np.zeros(num_opponents)
    response_added = np.ones(num_opponents, dtype=np.intp) * -1

    response_total_utility = np.mean(utilities, axis=1)
    best_response_index = np.argmax(response_total_utility)

    portfolio_utilities[0] = response_total_utility[best_response_index]
    response_added[0] = best_response_index

    max_utilities = np.zeros(num_opponents)
    np.copyto(max_utilities, utilities[best_response_index])

    response_available = [True] * num_opponents
    response_available[best_response_index] = False
    for i in range(1, num_opponents):
        best_portfolio_utility = None
        best_max_utilities = None
        best_response_to_add = None
        for j in range(num_opponents):
            if response_available[j]:
                new_max_utilities = np.maximum(max_utilities, utilities[j])
                new_portfolio_utility = np.mean(new_max_utilities)
                if not best_portfolio_utility or new_portfolio_utility > best_portfolio_utility:
                    best_portfolio_utility = new_portfolio_utility
                    best_max_utilities = new_max_utilities
                    best_response_to_add = j
        response_available[best_response_to_add] = False
        max_utilities = best_max_utilities
        portfolio_utilities[i] = best_portfolio_utility
        response_added[i] = best_response_to_add

    final_portfolio_size = None

    if portfolio_size > 0:
        final_portfolio_size = portfolio_size
    else:
        min_portfolio_utility = portfolio_utilities[0]
        max_portfolio_utility = portfolio_utilities[-1]
        total_utility_improvement = max_portfolio_utility - min_portfolio_utility
        minimal_improvement = total_utility_improvement * portfolio_cut_improvement_threshold
        final_portfolio_size = 1
        for i in range(1, num_opponents):
            if (portfolio_utilities[i] - portfolio_utilities[i - 1]) >= minimal_improvement:
                final_portfolio_size += 1
            else:
                break

    if log:
        print('Utilities table:')
        for i in range(num_opponents):
            print('\t'.join([str(u) for u in utilities[i]]))
        print('Response added: %s' % response_added)
        print('Final portfolio size: %s' % final_portfolio_size)

        plt.figure(dpi=160)
        plt.plot(np.arange(num_opponents, dtype=np.intp) + 1, portfolio_utilities)
        plt.plot(
            final_portfolio_size,
            portfolio_utilities[final_portfolio_size - 1],
            marker='o',
            color='r')
        plt.title('Portfolio utility')
        plt.xlabel('Portfolio size')
        plt.ylabel('Portfolio value [mbb/g]')
        plt.grid()
        plt.show()

    response_indices = response_added[:final_portfolio_size]
    return np.take(responses, response_indices), response_indices
