#!/bin/bash
# TODO

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
WORKSPACE_DIR="${SCRIPT_DIR}/.."

GAME_FILE_PATH="${WORKSPACE_DIR}/games/kuhn.limit.2p.game"
STRATEGY_FILE_PATH="${SCRIPT_DIR}/kuhn.limit.2p-equilibrium.strategy"

PYTHONPATH="${PYTHONPATH}:${WORKSPACE_DIR}" python "${WORKSPACE_DIR}/tools/strategy_agent.py" \
  "${GAME_FILE_PATH}" \
  "${STRATEGY_FILE_PATH}" \
  $1 $2