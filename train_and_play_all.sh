#!/bin/bash
# Train and test agents for games given by GAMES variable.
# For each game agent is trained with number of iterations given by ITERATIONS variable and
# then 5 matches are played against random agents each with 1000 hands.
#
# In general used to quickly verify nothing is fundamentally broken with CFR training algorithm.

GAMES=("kuhn.limit.2p" "kuhn.limit.3p" "leduc.limit.2p")
ITERATIONS=(100000 10000 100000)
GAME_PLAYS=5

for index in ${!GAMES[@]}; do
  python train.py \
    "./games/${GAMES[index]}/${GAMES[index]}.game" \
    ${ITERATIONS[index]} \
    "./games/${GAMES[index]}/${GAMES[index]}.strategy"

  for i in $(seq 1 ${GAME_PLAYS}); do
    ./play_against_random.sh "./games/${GAMES[index]}"
  done
done
