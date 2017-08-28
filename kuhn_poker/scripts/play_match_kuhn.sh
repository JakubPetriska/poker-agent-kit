#!/bin/bash

MATCH_NAME='kuhn.2p.limit'
LOGS_DIR='../../logs'
ACPC_DIR='../../../acpc-python-client/acpc_infrastructure'

if [ ! -d ${LOGS_DIR} ]; then
  mkdir ${LOGS_DIR}
fi

CURRENT_DIR=$(pwd)
cd ${ACPC_DIR}

./play_match.pl ${CURRENT_DIR}/${LOGS_DIR}/${MATCH_NAME} \
    ${CURRENT_DIR}/../kuhn.limit.2p.game \
    1 0 \
    CFR_Agent ${CURRENT_DIR}/kuhn_agent.limit.2p.sh \
    Random ${CURRENT_DIR}/kuhn_random_agent.limit.2p.sh
