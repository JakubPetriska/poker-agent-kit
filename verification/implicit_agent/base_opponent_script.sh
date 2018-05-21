#!/bin/bash
# TODO
# ###COMMENT###

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
WORKSPACE_DIR="${SCRIPT_DIR}/../../../.."

GAME_FILE_PATH="###GAME_FILE_PATH###"
STRATEGY_FILE_PATH="${SCRIPT_DIR}/###STRATEGY_FILE_PATH###"

PYTHONPATH="${PYTHONPATH}:${WORKSPACE_DIR}" python "${WORKSPACE_DIR}/tools/strategy_agent.py" \
  "${WORKSPACE_DIR}/${GAME_FILE_PATH}" \
  "${STRATEGY_FILE_PATH}" \
  $1 $2