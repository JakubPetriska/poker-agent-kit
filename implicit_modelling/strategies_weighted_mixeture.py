import numpy as np

from tools.walk_trees import walk_trees
from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.nodes import ActionNode
from tools.game_tree.node_provider import NodeProvider


class StrategiesWeightedMixtureActionNode(ActionNode):
    def __init__(self, parent, player):
        super().__init__(parent, player)
        self.strategy_nodes = None
        self.weights = None

    def __getattr__(self, attr):
        if attr == 'strategy':
            strategy_sum = np.zeros(3)
            for i, node in enumerate(self.strategy_nodes):
                strategy_sum += self.weights[i] * node.strategy
            return strategy_sum


class StrategiesWeightedMixtureTreeNodeProvider(NodeProvider):
    def create_action_node(self, parent, player):
        return StrategiesWeightedMixtureActionNode(parent, player)


class StrategiesWeightedMixture():
    def __init__(self, game, strategies):
        self.strategy = GameTreeBuilder(game, StrategiesWeightedMixtureTreeNodeProvider()).build_tree()
        self.weights = np.ones(len(strategies)) / len(strategies)

        def on_nodes(*nodes):
            nodes[0].weights = self.weights
            nodes[0].strategy_nodes = nodes[1:]
        walk_trees(on_nodes, self.strategy, *strategies)

    def update_weights(self, weights):
        np.copyto(self.weights, weights)
