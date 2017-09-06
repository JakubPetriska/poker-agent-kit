#!/bin/bash
# ACPC agent launching script.
# This script is used to play game of poker with this agent through ACPC poker infrastructure.

SCRIPT_DIR=`dirname $0`
GAME_NAME="$(basename ${SCRIPT_DIR})"

GAME_FILE_PATH="${SCRIPT_DIR}/${GAME_NAME}.game"
STRATEGY_FILE_PATH="${SCRIPT_DIR}/${GAME_NAME}.strategy"

python "${SCRIPT_DIR}/../../agent/strategy_agent.py" \
  ${GAME_FILE_PATH} \
  ${STRATEGY_FILE_PATH} \
  $1 $2