#!/bin/bash
# ACPC agent launching script.
# This script is used to play game of poker with this agent through ACPC poker infrastructure.
#
# This general random agent launching script. It's 1st argument is full path to
# random agent script for specific game.

SCRIPT_DIR="$( cd "$(dirname "$1")" ; pwd -P )"
SCRIPT_NAME="$(basename $1)"
GAME_NAME="$(sed 's/agent_random.\(.*\).sh/\1/' <<< ${SCRIPT_NAME})"

# Following file is to be found in acpc-python-client directory
AGENT_FILE="../random_agent.py"
if [ ! -f ${AGENT_FILE} ]; then
  AGENT_FILE="./random_agent.py"
fi

python ${AGENT_FILE} \
  "${SCRIPT_DIR}/../../games/${GAME_NAME}.game" \
  $2 $3