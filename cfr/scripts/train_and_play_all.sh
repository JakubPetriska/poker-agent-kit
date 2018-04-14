#!/bin/bash
# Train and test agents for games given by GAMES variable.
# For each game agent is trained with number of iterations given by ITERATIONS variable and
# then 5 matches are played against random agents each with 1000 hands.
#
# In general used to quickly verify nothing is fundamentally broken with CFR training algorithm.

GAMES=("kuhn.limit.2p" "leduc.limit.2p")
ITERATIONS=(100000 100000)
GAME_PLAYS=5

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
WORKSPACE_DIR="${SCRIPT_DIR}/../.."

export PYTHONPATH="${PYTHONPATH}:${WORKSPACE_DIR}"

for index in ${!GAMES[@]}; do
  python ${WORKSPACE_DIR}/cfr/train.py \
    "${WORKSPACE_DIR}/games/${GAMES[index]}.game" \
    ${ITERATIONS[index]} \
    "${WORKSPACE_DIR}/cfr/strategies/${GAMES[index]}.strategy"

  for i in $(seq 1 ${GAME_PLAYS}); do
    ${SCRIPT_DIR}/play_against_random.sh "games/${GAMES[index]}.game"
  done
done
