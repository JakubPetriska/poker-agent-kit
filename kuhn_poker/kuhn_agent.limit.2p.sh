#!/bin/bash

SCRIPT_DIR=`dirname $0`
export PYTHONPATH=${PYTHONPATH}:${SCRIPT_DIR}/..

python ${SCRIPT_DIR}/main.py ${SCRIPT_DIR}/kuhn.limit.2p.game $1 $2