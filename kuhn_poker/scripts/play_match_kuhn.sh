#!/bin/bash

MATCH_NAME='kuhn.2p.limit'
LOGS_DIR='../../logs'
ACPC_DIR='../../../acpc-python-client/acpc_infrastructure'

if [ ! -d ${LOGS_DIR} ]; then
  mkdir ${LOGS_DIR}
fi

SCRIPT_DIR=$(pwd)
cd ${ACPC_DIR}

TIMESTAMP=$(date +%s)

./play_match.pl ${SCRIPT_DIR}/${LOGS_DIR}/${MATCH_NAME} \
    ${SCRIPT_DIR}/../kuhn.limit.2p.game \
    1000 \
    ${TIMESTAMP} \
    Random ${SCRIPT_DIR}/kuhn_random_agent.limit.2p.sh \
    CFR_trained ${SCRIPT_DIR}/kuhn_agent.limit.2p.sh
