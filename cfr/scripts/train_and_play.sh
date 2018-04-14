#!/bin/bash
# Train and test agent for single game.
#
# In general used to quickly verify nothing is fundamentally broken with CFR training algorithm.

if [ "$#" != 3 ]; then
  echo 'Usage: ./train_and_play.sh {game_file_path} {iterations} {game_play_count}'
  exit 1
fi

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
WORKSPACE_DIR="${SCRIPT_DIR}/../.."

GAME_FILE_PATH="${WORKSPACE_DIR}/$1"
GAME_FILE_NAME="$(basename ${GAME_FILE_PATH})"
GAME_NAME=$(sed 's/\(.*\).game/\1/' <<< ${GAME_FILE_NAME})

ITERATIONS=$2
GAME_PLAY_COUNT=$3

export PYTHONPATH="${PYTHONPATH}:${WORKSPACE_DIR}"

python ${WORKSPACE_DIR}/cfr/train.py \
  ${GAME_FILE_PATH} \
  ${ITERATIONS} \
  "${WORKSPACE_DIR}/cfr/strategies/${GAME_NAME}.strategy"

for i in $(seq 1 ${GAME_PLAY_COUNT}); do
  ${SCRIPT_DIR}/play_against_random.sh $1
done
