#!/bin/bash
#SBATCH --job-name=DATA_AR_SAMPLE
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
NUM_SAMPLES=1000
VAL_RATIO=0.1
TEST_RATIO=0.1

for LAG in 4 8 16
do
    for NUM_WRITE_STEPS in 4 8
    do
        python -m T-GRAB.dataset.DTDG.graph_generation.run associative_recall \
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
done
