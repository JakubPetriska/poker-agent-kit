#!/bin/bash
# TODO

if [ "$#" != 5 ]; then
  echo 'Usage: ./start_dealer_and_other_players.sh {game_file_path} {logs_path} {opponent_name} {opponent_script} {player_name}'
  exit 1
fi

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
WORKSPACE_DIR="${SCRIPT_DIR}/.."
ACPC_CLIENT_DIR="${SCRIPT_DIR}/../../acpc-python-client"

GAME_FILE_PATH="${SCRIPT_DIR}/../$1"
GAME_FILE_NAME="$(basename ${GAME_FILE_PATH})"
GAME_NAME=$(sed 's/\(.*\).game/\1/' <<< ${GAME_FILE_NAME})

if [ ! -d ${LOGS_DIR} ]; then
  mkdir ${LOGS_DIR}
fi

cd ${ACPC_CLIENT_DIR}

TIMESTAMP=$(date +%s)

eval ./scripts/_start_dealer_and_player_1.pl \
  "${WORKSPACE_DIR}/$2" \
  ${GAME_FILE_PATH} \
  3000 \
  ${TIMESTAMP} \
  $3 "${WORKSPACE_DIR}/$4" \
  $5
