#!/bin/bash

if [[ "$PWD" != */T-GRAB ]]; then
    echo "Error: Please run this script from the T-GRAB directory."
    exit 1
fi

echo "Start submitting sparse associative-recall dataset generation..."
sleep 2

NUM_NODES=100
ACTIVE_NODES=16
PAIRS_PER_STEP=2
QUERY_RATIO=1.0
NUM_SAMPLES=4000
VAL_RATIO=0.1
TEST_RATIO=0.1

CONFIGS=(
    "4 4"
    "8 8"
    "16 8"
)

for CONFIG in "${CONFIGS[@]}"
do
    set -- $CONFIG
    LAG=$1
    NUM_WRITE_STEPS=$2
    DATASET_PATTERN="($LAG, $NUM_WRITE_STEPS)"
    sbatch \
        --mem=4G \
        --output="logs/associative_recall_sparse/($LAG, $NUM_WRITE_STEPS)_${NUM_NODES}nn_${ACTIVE_NODES}an_${PAIRS_PER_STEP}pps_${NUM_SAMPLES}ns/%j-o.out" \
        --error="logs/associative_recall_sparse/($LAG, $NUM_WRITE_STEPS)_${NUM_NODES}nn_${ACTIVE_NODES}an_${PAIRS_PER_STEP}pps_${NUM_SAMPLES}ns/%j-e.out" \
        scripts/dataset/associative_recall_sparse/_main.sh \
            "$DATASET_PATTERN" \
            $NUM_NODES \
            $VAL_RATIO \
            $TEST_RATIO \
            $ACTIVE_NODES \
            $PAIRS_PER_STEP \
            $QUERY_RATIO \
            $NUM_SAMPLES
done
