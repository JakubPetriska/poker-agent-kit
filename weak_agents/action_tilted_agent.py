from enum import Enum
import numpy as np

import acpc_python_client as acpc

from cfr.main import Cfr
from tools.walk_trees import walk_trees
from tools.io_util import read_strategy_from_file
from tools.game_utils import copy_strategy
from tools.game_tree.builder import GameTreeBuilder
from tools.game_tree.node_provider import StrategyTreeNodeProvider


class TiltType(Enum):
    ADD = 1,
    MULTIPLY = 2,


def create_agent_strategy(
        game_file_path,
        tilt_action,
        tilt_type,
        tilt_probability,
        cfr_iterations=20000,
        show_progress=True):

    game = acpc.read_game_file(game_file_path)
    cfr = Cfr(game, show_progress=show_progress)
    cfr.train(cfr_iterations)
    return create_agent_strategy_from_trained_strategy(
        game_file_path,
        cfr.game_tree,
        tilt_action,
        tilt_type,
        tilt_probability,
        True)


def create_agent_strategy_from_strategy_file(
        game_file_path,
        strategy_path,
        tilt_action,
        tilt_type,
        tilt_probability):

    strategy_tree = read_strategy_from_file(game_file_path, strategy_path)
    return create_agent_strategy_from_trained_strategy(
        game_file_path,
        strategy_tree,
        tilt_action,
        tilt_type,
        tilt_probability,
        True)


def create_agent_strategy_from_trained_strategy(
        game_file_path,
        strategy_tree,
        tilt_action,
        tilt_type,
        tilt_probability,
        in_place=False):

    tilt_action_index = tilt_action.value

    def on_node(node):
        if tilt_action_index in node.children:
            original_tilt_action_probability = node.strategy[tilt_action_index]
            new_tilt_action_probability = -1
            if tilt_type == TiltType.ADD:
                new_tilt_action_probability = np.clip(original_tilt_action_probability + tilt_probability, 0, 1)
            elif tilt_type == TiltType.MULTIPLY:
                new_tilt_action_probability = np.clip(
                    original_tilt_action_probability + original_tilt_action_probability * tilt_probability, 0, 1)
            node.strategy[tilt_action_index] = new_tilt_action_probability
            diff = original_tilt_action_probability - new_tilt_action_probability
            per_action_diff = diff / (len(node.children) - 1)
            for a in node.children:
                if a != tilt_action_index:
                    node.strategy[a] += per_action_diff

    result_strategy = None
    if in_place:
        result_strategy = strategy_tree
    else:
        game = acpc.read_game_file(game_file_path)
        result_strategy = GameTreeBuilder(game, StrategyTreeNodeProvider()).build_tree()
        copy_strategy(result_strategy, strategy_tree)

    walk_trees(on_node, result_strategy)
    return result_strategy
