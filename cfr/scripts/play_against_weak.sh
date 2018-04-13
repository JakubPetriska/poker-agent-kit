#!/bin/bash
# Play 1000 hands of a poker game through ACPC infrastructure and print results.
# Agent trained using CFR is pitted against required number of agents (given by game's player count)
# that are also trained using CFR.
# However opponent agents are launched using different script so they can be trained with different number of
# iterations, usually smaller.
#
# Game file path provided as an argument must be in the games diretory in the root
# of the project and must be relative to this project's directory.

if [ "$#" != 1 ]; then
  echo 'Usage: ./play_against_weak.sh {game_file_path}'
  exit 1
fi

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
WORKSPACE_DIR="${SCRIPT_DIR}/../.."
ACPC_DIR="${WORKSPACE_DIR}/../acpc-python-client/acpc_infrastructure"

GAME_FILE_PATH="${WORKSPACE_DIR}/$1"
GAME_FILE_NAME="$(basename ${GAME_FILE_PATH})"
GAME_NAME=$(sed 's/\(.*\).game/\1/' <<< ${GAME_FILE_NAME})

# Get number of players
NUM_PLAYERS=$(cat ${GAME_FILE_PATH} | grep numPlayers | sed 's/\s*numPlayers\s*=\s*\([0-9]*\)\s*/\1/')
NUM_OTHER_PLAYERS=$((${NUM_PLAYERS} - 1))
PLAYERS=""
for i in `seq 1 ${NUM_OTHER_PLAYERS}`;
do
  PLAYERS="${PLAYERS} CFR_trained_weak_${i} "${SCRIPT_DIR}/agents/agent_trained_weak.${GAME_NAME}.sh""
done

LOGS_DIR="${WORKSPACE_DIR}/logs"

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
  CFR_trained "${SCRIPT_DIR}/agents/agent_trained.${GAME_NAME}.sh"
