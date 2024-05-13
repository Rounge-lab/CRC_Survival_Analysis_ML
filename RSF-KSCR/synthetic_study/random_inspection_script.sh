#!/bin/sh

RANDOM=49
SEED=435
rand=bash -c '$RANDOM=SEED;'

run=$((0 + $RANDOM % 49))

# second input is max number of cores to use
./random_inspection.sh $run 8



