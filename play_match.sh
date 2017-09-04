#!/bin/bash

if [ "$#" != 1 ]; then
  echo 'Usage: ./play_match.sh {game_dir}'
  exit 1
fi

ACPC_DIR="../acpc-python-client/acpc_infrastructure"

SCRIPT_DIR=$(pwd)
GAME_DIR_PATH=${SCRIPT_DIR}/$1
GAME_NAME="$(basename ${GAME_DIR_PATH})"

LOGS_DIR="${SCRIPT_DIR}/logs"

if [ ! -d ${LOGS_DIR} ]; then
  mkdir ${LOGS_DIR}
fi

cd ${ACPC_DIR}

TIMESTAMP=$(date +%s)

./play_match.pl \
  "${LOGS_DIR}/${GAME_NAME}" \
  "${GAME_DIR_PATH}/${GAME_NAME}.game" \
  1000 \
  ${TIMESTAMP} \
  Random "${GAME_DIR_PATH}/agent_random.${GAME_NAME}.sh" \
  CFR_trained "${GAME_DIR_PATH}/agent_trained.${GAME_NAME}.sh"
