export ROOT_LOAD_SAVE_DIR="$PWD/scratch/"
export SCRIPT_LOC=scripts/task/link_pred/
export DATA_LOC=$PWD/scratch/data/
export PYENV=$PWD/tgrab/

if [[ "$PWD" != */T-GRAB ]]; then
    echo "Error: Please run this script from the T-GRAB directory."
    exit 1
fi

task=associative_recall
NUM_EPOCHS_TO_VIS=0

EVAL_MODE=false
CTDG_DO_SNAPSHOT_TRAINING=true
METHODS_TO_RUN=("CTDG/_tgn_provids" "CTDG/_tgn_provids_mlstm")
CLEAR_RESULT=true
WANDB_ENTITY="cristoferivalentina5-danmarks-tekniske-universitet-dtu"
MESSAGE_AGGREGATOR="sequence"
MLSTM_NUM_HEADS=1
TGN_LR=5e-5
MLSTM_LR=2e-5

VAL_FIRST_METRIC="memnode_avg_ap"

NUM_NODES=100
ACTIVE_NODES=16
PAIRS_PER_STEP=2
QUERY_RATIO=1.0
NUM_SAMPLES=1000

for VAL_RATIO in 0.1
do
    for TEST_RATIO in 0.1
    do
        for LAG in 4 8 16
        do
            for NUM_WRITE_STEPS in 4 8
            do
                DATA="($LAG, $NUM_WRITE_STEPS)/associative_recall-${NUM_SAMPLES}ns-${NUM_NODES}nn-${ACTIVE_NODES}an-${PAIRS_PER_STEP}pps-${QUERY_RATIO}qr-${VAL_RATIO}vr-${TEST_RATIO}tr"

                for SEED in 1235
                do
                    for NODE_FEAT in "ONE_HOT"
                    do
                        for NODE_FEAT_DIM in 1
                        do
                            for model in "CTDG/_tgn_provids" "CTDG/_tgn_provids_mlstm"
                            do
                                if [[ " ${METHODS_TO_RUN[@]} " =~ " ${model} " ]]; then
                                    MEM=4
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

                                    ./scripts/task/link_pred/associative_recall/$model.sh \
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
