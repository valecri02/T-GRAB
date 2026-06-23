#!/bin/bash
#SBATCH --job-name=DATA_OLR
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

python -m T-GRAB.dataset.DTDG.graph_generation.run ordered_long_range \
    --num-nodes=$2 \
    --dataset-name="${1}" \
    --seed=12345 \
    \
    --val-ratio=$3 \
    --test-ratio=$4 \
    \
    --num-branches=$5 \
    --num-samples=$6 \
    --save-dir=$PWD/T-GRAB/scratch/data/
