#!/bin/bash

SCRIPT_DIR=`dirname $0`
GAME_NAME="$(basename ${SCRIPT_DIR})"

AGENT_FILE="../random_agent.py"
if [ ! -f ${AGENT_FILE} ]; then
  AGENT_FILE="./random_agent.py"
fi

python ${AGENT_FILE} \
  "${SCRIPT_DIR}/${GAME_NAME}.game" \
  $1 $2