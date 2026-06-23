export ROOT_LOAD_SAVE_DIR="$PWD/scratch/"
export SCRIPT_LOC=scripts/task/link_pred/
export DATA_LOC=$PWD/scratch/data/
export PYENV=$PWD/tgrab/

if [[ "$PWD" != */T-GRAB ]]; then
    echo "Error: Please run this script from the T-GRAB directory."
    exit 1
fi

task=ordered_long_range
NUM_EPOCHS_TO_VIS=0


EVAL_MODE=false
CTDG_DO_SNAPSHOT_TRAINING=true
# METHODS_TO_RUN=("CTDG/_tgn_provids" "CTDG/_tgn_provids_mlstm")
METHODS_TO_RUN=("CTDG/_tgn_provids_mlstm")
CLEAR_RESULT=true
WANDB_ENTITY="cristoferivalentina5-danmarks-tekniske-universitet-dtu"
MESSAGE_AGGREGATOR="sequence"
MLSTM_NUM_HEADS=2
TGN_LR=5e-6
MLSTM_LR=5e-6

VAL_FIRST_METRIC="memnode_avg_ap"

NUM_NODES=100
NUM_BRANCHES=6
NUM_SAMPLES=1000

for VAL_RATIO in 0.1
do
    for TEST_RATIO in 0.1
    do
        for LAG in 4
        do
            for BRANCH_LEN in 4
            do
                DATA="($LAG, $BRANCH_LEN)/ordered_long_range-${NUM_SAMPLES}ns-${NUM_NODES}nn-${NUM_BRANCHES}nb-${VAL_RATIO}vr-${TEST_RATIO}tr"

                RAW_MEM=$(echo "0.05 * $NUM_BRANCHES * $BRANCH_LEN" | bc)
                RAW_MEM=$(printf "%.0f" "$RAW_MEM")

                for SEED in 1235 #2346 3457
                do
                    for NODE_FEAT in "ONE_HOT"
                    do
                        for NODE_FEAT_DIM in 1
                        do
                            for model in "CTDG/_tgn_provids" "CTDG/_tgn_provids_mlstm"
                            do
                                if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                    if (( $(echo "$RAW_MEM < 4" | bc -l) )); then
                                        MEM=4
                                    else
                                        MEM=$RAW_MEM
                                    fi
                                    GPU=40
                                    MAX_GPU=40
                                    NUM_UNITS=1
                                    NUM_HEADS=2
                                    TIME_FEAT_DIM=100
                                    NUM_NEIGHBORS=20
                                    CHANNEL_EMBEDDING_DIM=50
                                    MAX_INPUT_SEQ_LEN=20
                                    TRAIN_BATCH_SIZE=1

                                    EXTRA_ARGS=()
                                    if [[ "${model}" = "CTDG/_tgn_provids" ]]; then
                                        EXTRA_ARGS=("$MESSAGE_AGGREGATOR" "$TGN_LR")
                                    elif [[ "${model}" = "CTDG/_tgn_provids_mlstm" ]]; then
                                        EXTRA_ARGS=("$MLSTM_NUM_HEADS" "$MESSAGE_AGGREGATOR" "$MLSTM_LR")
                                    fi

                                    ./scripts/task/link_pred/ordered_long_range/$model.sh \
                                        "$DATA" \
                                        $SEED \
                                        $NODE_FEAT \
                                        $NODE_FEAT_DIM \
                                        $EVAL_MODE \
                                        $NUM_EPOCHS_TO_VIS \
                                        $ROOT_LOAD_SAVE_DIR \
                                        "$VAL_FIRST_METRIC" \
                                        $MEM \
                                        $MAX_GPU \
                                        $GPU \
                                        $NUM_UNITS \
                                        $NUM_HEADS \
                                        $TIME_FEAT_DIM \
                                        $NUM_NEIGHBORS \
                                        $TRAIN_BATCH_SIZE \
                                        $CTDG_DO_SNAPSHOT_TRAINING \
                                        $CHANNEL_EMBEDDING_DIM \
                                        $MAX_INPUT_SEQ_LEN \
                                        $CLEAR_RESULT \
                                        $NUM_NODES \
                                        $WANDB_ENTITY \
                                        "${EXTRA_ARGS[@]}"
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done
