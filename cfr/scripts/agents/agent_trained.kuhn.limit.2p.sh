#!/bin/bash
# ACPC agent launching script.
# This script is used to play game of poker with this agent through ACPC poker infrastructure.

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"

GAME_NAME="kuhn.limit.2p"
STRATEGY_FILE_PATH="${SCRIPT_DIR}/../../strategies/${GAME_NAME}.strategy"

${SCRIPT_DIR}/agent_trained.sh $GAME_NAME $STRATEGY_FILE_PATH $1 $2
