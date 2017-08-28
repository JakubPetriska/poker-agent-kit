#!/bin/bash

SCRIPT_DIR=`dirname $0`
AGENTS_DIR=${SCRIPT_DIR}/../agents
export PYTHONPATH=${PYTHONPATH}:${SCRIPT_DIR}/..

python ${AGENTS_DIR}/kuhn_random_agent.py ${SCRIPT_DIR}/../kuhn.limit.2p.game $1 $2