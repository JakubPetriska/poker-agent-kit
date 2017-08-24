#!/bin/bash

ACPC_DIR=../../acpc-python-client/acpc_infrastructure

CURRENT_DIR=$(pwd)

if [ ! -d ../logs ]; then
  mkdir ../logs
fi

cd ${ACPC_DIR}

./play_match.pl ${CURRENT_DIR}/../logs/matchName \
    ${CURRENT_DIR}/kuhn.limit.2p.game \
    1 0 \
    Alice ${CURRENT_DIR}/kuhn_agent.limit.2p.sh \
    Bob ${CURRENT_DIR}/kuhn_agent.limit.2p.sh
