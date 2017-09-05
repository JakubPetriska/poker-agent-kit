#!/bin/bash

if [ "$#" != 1 ]; then
  echo 'Usage: ./play_against_random.sh {game_dir}'
  exit 1
fi

ACPC_DIR="../acpc-python-client/acpc_infrastructure"

SCRIPT_DIR=$(pwd)
GAME_DIR_PATH=${SCRIPT_DIR}/$1
GAME_NAME="$(basename ${GAME_DIR_PATH})"
GAME_FILE_PATH="${GAME_DIR_PATH}/${GAME_NAME}.game"

# Get number of players
NUM_PLAYERS=$(cat ${GAME_FILE_PATH} | grep numPlayers | sed 's/\s*numPlayers\s*=\s*\([0-9]*\)\s*/\1/')
NUM_RANDOM_PLAYERS=$((${NUM_PLAYERS} - 1))
PLAYERS=""
for i in `seq 1 ${NUM_RANDOM_PLAYERS}`;
do
  PLAYERS="${PLAYERS} Random_${i} "${GAME_DIR_PATH}/agent_random.${GAME_NAME}.sh""
done

LOGS_DIR="${SCRIPT_DIR}/logs"

if [ ! -d ${LOGS_DIR} ]; then
  mkdir ${LOGS_DIR}
fi

cd ${ACPC_DIR}

TIMESTAMP=$(date +%s)

eval "./play_match.pl" \
  "${LOGS_DIR}/${GAME_NAME}" \
  ${GAME_FILE_PATH} \
  1000 \
  ${TIMESTAMP} \
  ${PLAYERS} \
  CFR_trained "${GAME_DIR_PATH}/agent_trained.${GAME_NAME}.sh"
