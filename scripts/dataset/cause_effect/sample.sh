# Load module and environment

if [[ "$PWD" != */T-GRAB ]]; then
    echo "Error: Please run this script from the T-GRAB directory."
    exit 1
fi

source $PWD/tgrab/bin/activate
cd ../

LAG=256
NUM_PATTERNS=4000

python -m T-GRAB.dataset.DTDG.graph_generation.run cause_effect \
    --num-nodes=100 \
    --dataset-name="($LAG, $NUM_PATTERNS)" \
    --seed=12345 \
    --val-ratio=0.1 \
    --test-ratio=0.1 \
    --test-inductive-ratio=0.0 \
    --test-inductive-num-nodes-ratio=0.0 \
    \
    --er-prob=0.002 \
    --er-prob-inductive=0.0 \
    --save-dir=$PWD/T-GRAB/scratch/data/