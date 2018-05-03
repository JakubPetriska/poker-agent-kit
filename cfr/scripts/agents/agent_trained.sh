#!/bin/bash
# ACPC agent launching script.
# This script is used to play game of poker with this agent through ACPC poker infrastructure.
#
# This general trained agent launching script. It's 1st argument is full path to
# trained agent script for specific game.

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
WORKSPACE_DIR="${SCRIPT_DIR}/../../../"

PYTHONPATH="${PYTHONPATH}:${WORKSPACE_DIR}" python "${WORKSPACE_DIR}/tools/strategy_agent.py" \
  "${WORKSPACE_DIR}/games/$1.game" \
  $2 $3 $4