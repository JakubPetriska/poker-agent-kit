#!/bin/bash
# ACPC agent launching script.
# This script is used to play game of poker with this agent through ACPC poker infrastructure.

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
${SCRIPT_DIR}/agent_random.sh $0 $1 $2
