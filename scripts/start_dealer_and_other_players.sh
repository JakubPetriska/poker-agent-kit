#!/bin/bash
# Play 1000 hands of a poker game through ACPC infrastructure and print results.
# Script launches the dealer and random agents such that one place is left free
# in the game. The agent can then join using the printed port number.
#
# Game file path provided as an argument must be in the games diretory in the root
# of the project and must be relative to this project's directory.
#
# Use this to debug the agent.

if [ "$#" != 1 ]; then
  echo 'Usage: ./play_match.sh {game_file_path}'
  exit 1
fi

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
ACPC_CLIENT_DIR="${SCRIPT_DIR}/../../acpc-python-client"

GAME_FILE_PATH="${SCRIPT_DIR}/../$1"
GAME_FILE_NAME="$(basename ${GAME_FILE_PATH})"
GAME_NAME=$(sed 's/\(.*\).game/\1/' <<< ${GAME_FILE_NAME})

# Get number of players
NUM_PLAYERS=$(cat ${GAME_FILE_PATH} | grep numPlayers | sed 's/\s*numPlayers\s*=\s*\([0-9]*\)\s*/\1/')
NUM_RANDOM_PLAYERS=$((${NUM_PLAYERS} - 1))
PLAYERS=""
for i in `seq 1 ${NUM_RANDOM_PLAYERS}`;
do
  PLAYERS="${PLAYERS} Random_${i} "${SCRIPT_DIR}/agents/agent_random.${GAME_NAME}.sh""
done
echo $PLAYERS

LOGS_DIR="${SCRIPT_DIR}/../logs"

if [ ! -d ${LOGS_DIR} ]; then
  mkdir ${LOGS_DIR}
fi

cd ${ACPC_CLIENT_DIR}

TIMESTAMP=$(date +%s)

eval ./scripts/_start_dealer_and_player_1.pl \
  "${LOGS_DIR}/${GAME_NAME}" \
  ${GAME_FILE_PATH} \
  1000 \
  ${TIMESTAMP} \
  ${PLAYERS} \
  Tested_agent