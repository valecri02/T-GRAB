#!/bin/bash
#SBATCH --job-name=CT_CE_TGN_PROVIDS_NOMEMORY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

DATA_LOC=data/
RUN_SCRIPT=T-GRAB.train.run
NODE_POS=circular_layout

cd ../

DATA="$1"
SEED=$2
NODE_FEAT=$3
NODE_FEAT_DIM=$4
EVAL_MODE=$5
NUM_EPOCHS_TO_VIS=$6
ROOT_LOAD_SAVE_DIR=$7
VAL_FIRST_METRIC=${8}
NUM_UNITS=${12}
NUM_HEADS=${13}
TIME_FEAT_DIM=${14}
NUM_NEIGHBORS=${15}
TRAIN_BATCH_SIZE=${16}
TRAIN_SNAPSHOT_BASED=${17}
CLEAR_RESULT=${20}
WANDB_ENTITY=${22}

echo "@@@ RUNNING TGN ProvIDS without memory on $DATA @@@"

ARGS=(
    CTDG.link_pred.memory_node.tgn_provids_nomemory
    --data="$DATA"
    --seed=$SEED
    --node-feat=$NODE_FEAT
    --data-loc=$DATA_LOC
    --num-units=$NUM_UNITS
    --val-first-metric=$VAL_FIRST_METRIC
    --node-pos=$NODE_POS
    --node-feat-dim=$NODE_FEAT_DIM
    --patience=50
    --num-epoch=100000
    --root-load-save-dir=$ROOT_LOAD_SAVE_DIR
    --num-neighbors=$NUM_NEIGHBORS
    --num-heads=$NUM_HEADS
    --dropout=0.1
    --time-feat-dim=$TIME_FEAT_DIM
    --train-batch-size=$TRAIN_BATCH_SIZE
    --wandb-entity=$WANDB_ENTITY
    --wandb-project="T-GRAB_ce_final"
)

TRAIN_ARGS=("${ARGS[@]}" "--num-epochs-to-visualize=$NUM_EPOCHS_TO_VIS" "--replay-memory-before-eval")
EVAL_ARGS=("${ARGS[@]}" "--num-epochs-to-visualize=0" "--eval-mode")

if [ "$CLEAR_RESULT" == "true" ]; then
    TRAIN_ARGS=("${TRAIN_ARGS[@]}" --clear-results)
fi
if [ "$TRAIN_SNAPSHOT_BASED" == "true" ]; then
    TRAIN_ARGS=("${TRAIN_ARGS[@]}" "--train-snapshot-based")
    EVAL_ARGS=("${EVAL_ARGS[@]}" "--train-snapshot-based")
fi

if [ "$EVAL_MODE" == "false" ]; then
    python -m $RUN_SCRIPT "${TRAIN_ARGS[@]}"
else
    python -m $RUN_SCRIPT "${EVAL_ARGS[@]}"
fi
