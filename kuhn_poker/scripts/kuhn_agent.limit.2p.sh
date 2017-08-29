#!/bin/bash

SCRIPT_DIR=`dirname $0`
AGENTS_DIR=${SCRIPT_DIR}/../agents
export PYTHONPATH=${PYTHONPATH}:${SCRIPT_DIR}/..

GAME_FILE_PATH=${SCRIPT_DIR}/../kuhn.limit.2p.game
STRATEGY_FILE_PATH=${SCRIPT_DIR}/../kuhn.limit.2p.strategy

python ${AGENTS_DIR}/kuhn_agent.py ${GAME_FILE_PATH} ${STRATEGY_FILE_PATH} $1 $2