#!/bin/bash

GAMES=("kuhn.limit.2p" "kuhn.limit.3p" "leduc.limit.2p")
ITERATIONS=(100000 10000 100000)
GAME_PLAYS=5

for index in ${!GAMES[@]}; do
  python train.py \
    "./games/${GAMES[index]}/${GAMES[index]}.game" \
    ${ITERATIONS[index]} \
    "./games/${GAMES[index]}/${GAMES[index]}.strategy"

  for i in $(seq 1 ${GAME_PLAYS}); do
    ./play_match.sh "./games/${GAMES[index]}"
  done
done
