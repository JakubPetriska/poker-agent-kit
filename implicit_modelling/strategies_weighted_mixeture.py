import numpy as np

from tools.walk_trees import walk_trees
from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.nodes import ActionNode
from tools.game_tree.node_provider import NodeProvider


class StrategiesWeightedMixtureActionNode(ActionNode):
    def __init__(self, parent, player):
        super().__init__(parent, player)
        self.strategies = None
        self.weights = None

    def __getattr__(self, attr):
        if attr == 'strategy':
            return np.average(self.strategies, axis=0, weights=self.weights)


class StrategiesWeightedMixtureTreeNodeProvider(NodeProvider):
    def create_action_node(self, parent, player):
        return StrategiesWeightedMixtureActionNode(parent, player)


class StrategiesWeightedMixture():
    def __init__(self, game, strategies):
        self.strategy = GameTreeBuilder(game, StrategiesWeightedMixtureTreeNodeProvider()).build_tree()
        self.weights = np.ones(len(strategies)) / len(strategies)

        def on_nodes(*nodes):
            mixture_node = nodes[0]
            if isinstance(mixture_node, ActionNode):
                mixture_node.weights = self.weights
                mixture_node.strategies = np.zeros([len(strategies), 3])
                for i, node in enumerate(nodes[1:]):
                    mixture_node.strategies[i, :] = node.strategy
        walk_trees(on_nodes, self.strategy, *strategies)

    def update_weights(self, weights):
        np.copyto(self.weights, weights)
