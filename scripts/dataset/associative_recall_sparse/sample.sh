#!/bin/bash
#SBATCH --job-name=DATA_ARS_SAMPLE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu

if command -v module >/dev/null 2>&1; then
    module load python/3.8
fi
if [ -f "$PWD/tgrab/bin/activate" ]; then
    source $PWD/tgrab/bin/activate
fi
cd ../

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
    python -m T-GRAB.dataset.DTDG.graph_generation.run associative_recall_sparse \
        --num-nodes=$NUM_NODES \
        --dataset-name="($LAG, $NUM_WRITE_STEPS)" \
        --seed=12345 \
        --val-ratio=$VAL_RATIO \
        --test-ratio=$TEST_RATIO \
        --active-nodes=$ACTIVE_NODES \
        --pairs-per-step=$PAIRS_PER_STEP \
        --query-ratio=$QUERY_RATIO \
        --num-samples=$NUM_SAMPLES \
        --save-dir=$PWD/T-GRAB/scratch/data/
done
