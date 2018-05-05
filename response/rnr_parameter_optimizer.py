from response.restricted_nash_response import RestrictedNashResponse
from evaluation.exploitability import Exploitability
from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import StrategyTreeNodeProvider
from tools.game_utils import copy_strategy


class RnrParameterOptimizer():
    def __init__(
            self,
            game,
            iterations=1500,
            checkpoint_iterations=10,
            show_progress=True):
        self.game = game
        self.iterations = iterations
        self.checkpoint_iterations = checkpoint_iterations
        self.show_progress = show_progress
        self.exp = Exploitability(game)

    def train(
            self,
            opponent_strategy,
            exploitability,
            max_exploitability_delta):

        result_strategy = GameTreeBuilder(self.game, StrategyTreeNodeProvider()).build_tree()
        best_exploitability = float('inf')
        best_exploitability_delta = float('inf')

        def checkpoint_callback(game_tree, checkpoint_index, iterations):
            if iterations <= ((3 / 4) * self.iterations):
                # Make sure the strategy at least partially converged
                return

            nonlocal result_strategy
            nonlocal best_exploitability_delta
            nonlocal best_exploitability

            current_exploitability = self.exp.evaluate(game_tree)
            current_exploitability_delta = abs(current_exploitability - exploitability)
            if current_exploitability_delta < best_exploitability_delta:
                if current_exploitability_delta <= max_exploitability_delta:
                    copy_strategy(result_strategy, game_tree)
                best_exploitability_delta = current_exploitability_delta
                best_exploitability = current_exploitability

        iteration = 0
        p_low = 0
        p_high = 1

        if self.show_progress:
            print()

        while True:
            if self.show_progress:
                iteration += 1
                print('Run %s' % iteration)
                print('Interval: %s - %s' % (p_low, p_high))
            p_current = p_low + (p_high - p_low) / 2
            rnr = RestrictedNashResponse(
                self.game,
                opponent_strategy,
                p_current,
                show_progress=self.show_progress)
            rnr.train(
                self.iterations,
                checkpoint_iterations=self.checkpoint_iterations,
                checkpoint_callback=checkpoint_callback)

            if best_exploitability_delta < max_exploitability_delta:
                return result_strategy, best_exploitability, p_current

            if self.show_progress:
                print('Exploitability: %s, p=%s, current_delta=%s' % (best_exploitability, p_current, best_exploitability_delta))

            if best_exploitability > exploitability:
                p_high = p_current
            else:
                p_low = p_current
            best_exploitability = float('inf')
            best_exploitability_delta = float('inf')
