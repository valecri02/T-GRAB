#!/bin/bash

if [[ "$PWD" != */T-GRAB ]]; then
    echo "Error: Please run this script from the T-GRAB directory."
    exit 1
fi

echo "Start submitting ordered-long-range dataset generation..."
sleep 2

NUM_NODES=100
NUM_BRANCHES=8
NUM_SAMPLES=4000
VAL_RATIO=0.1
TEST_RATIO=0.1

for LAG in 0 4 8 16
do
    for BRANCH_LEN in 4 8
    do
        DATASET_PATTERN="($LAG, $BRANCH_LEN)"
        sbatch \
            --mem=4G \
            --output="logs/ordered_long_range/($LAG, $BRANCH_LEN)_${NUM_NODES}nn_${NUM_BRANCHES}nb_${NUM_SAMPLES}ns/%j-o.out" \
            --error="logs/ordered_long_range/($LAG, $BRANCH_LEN)_${NUM_NODES}nn_${NUM_BRANCHES}nb_${NUM_SAMPLES}ns/%j-e.out" \
            scripts/dataset/ordered_long_range/_main.sh \
                "$DATASET_PATTERN" \
                $NUM_NODES \
                $VAL_RATIO \
                $TEST_RATIO \
                $NUM_BRANCHES \
                $NUM_SAMPLES
    done
done
