#!/bin/bash
#SBATCH --job-name=DATA_OLR_SAMPLE
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
NUM_BRANCHES=6
NUM_SAMPLES=1000
VAL_RATIO=0.1
TEST_RATIO=0.1
BRANCH_LEN=4

for LAG in 0 4 8
do
    python -m T-GRAB.dataset.DTDG.graph_generation.run ordered_long_range \
        --num-nodes=$NUM_NODES \
        --dataset-name="($LAG, $BRANCH_LEN)" \
        --seed=12345 \
        --val-ratio=$VAL_RATIO \
        --test-ratio=$TEST_RATIO \
        --num-branches=$NUM_BRANCHES \
        --num-samples=$NUM_SAMPLES \
        --save-dir=$PWD/T-GRAB/scratch/data/
done
