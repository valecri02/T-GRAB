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
NUM_KEYS=16
NUM_VALUES=16
NUM_DISTRACTOR_EDGES=4
NUM_SAMPLES=1000
VAL_RATIO=0.1
TEST_RATIO=0.1

for LAG in 8 16
do
    for NUM_PAIRS in 4 8 16
    do
        python -m T-GRAB.dataset.DTDG.graph_generation.run associative_recall \
            --num-nodes=$NUM_NODES \
            --dataset-name="($LAG, $NUM_PAIRS)" \
            --seed=12345 \
            --val-ratio=$VAL_RATIO \
            --test-ratio=$TEST_RATIO \
            --num-keys=$NUM_KEYS \
            --num-values=$NUM_VALUES \
            --num-distractor-edges=$NUM_DISTRACTOR_EDGES \
            --num-samples=$NUM_SAMPLES \
            --save-dir=$PWD/T-GRAB/scratch/data/
    done
done
