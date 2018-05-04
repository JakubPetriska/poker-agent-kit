#!/bin/bash
# TODO

if [ "$#" != 6 ]; then
  echo 'Usage: ./evaluate.sh {game_file_path} {logs_dir} {agent_name} {agent_script} {opponent_name} {opponent_script}'
  exit 1
fi

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
WORKSPACE_DIR="${SCRIPT_DIR}/.."
ACPC_DIR="${WORKSPACE_DIR}/../acpc-python-client/acpc_infrastructure"

GAME_FILE_PATH="${WORKSPACE_DIR}/$1"

cd ${ACPC_DIR}

TIMESTAMP=$(date +%s)

eval "./play_match.pl" \
  "${WORKSPACE_DIR}/$2" \
  ${GAME_FILE_PATH} \
  3000 \
  ${TIMESTAMP} \
  $3 "${WORKSPACE_DIR}/$4" \
  $5 "${WORKSPACE_DIR}/$6"