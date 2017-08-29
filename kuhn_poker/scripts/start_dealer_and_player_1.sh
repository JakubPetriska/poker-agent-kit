#!/bin/bash

MATCH_NAME='kuhn.2p.limit'
LOGS_DIR='../../logs'
ACPC_CLIENT_DIR='../../../acpc-python-client'

if [ ! -d ${LOGS_DIR} ]; then
  mkdir ${LOGS_DIR}
fi

SCRIPT_DIR=$(pwd)
cd ${ACPC_CLIENT_DIR}

TIMESTAMP=$(date +%s)

./scripts/_start_dealer_and_player_1.pl \
  ${SCRIPT_DIR}/${LOGS_DIR}/${MATCH_NAME} \
  ${SCRIPT_DIR}/../kuhn.limit.2p.game \
  1000 \
  ${TIMESTAMP} \
  Random ${SCRIPT_DIR}/kuhn_random_agent.limit.2p.sh \
  CFR_trained