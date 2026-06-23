#!/bin/bash
### LSF options
#BSUB -q gpua100
#BSUB -J lr_44_3b_l
#BSUB -o lr_44_3b_l_%J.out
#BSUB -e lr_44_3b_l_%J.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -M 5GB
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -B
#BSUB -N

set -euo pipefail

module purge

module load cuda/11.7

# Conda setup
source ~/miniforge3/bin/activate
conda activate tgrab

# Run from anywhere inside this repository.
REPO_DIR=/work3/s253892/T-GRAB/
PROJECT_PARENT=/work3/s253892/

cd "$PROJECT_PARENT"

# Keep Hydra/joblib runs from competing for CPU threads inside each process.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TGRAB_ROOT_LOAD_SAVE_DIR="${TGRAB_ROOT_LOAD_SAVE_DIR:-$REPO_DIR/scratch}"

# Choose one of: cause_effect, long_range, ordered_long_range
DATASET="${DATASET:-cause_effect}"

# Choose one or more model configs.
# MODELS="${MODELS:-tgn_provids,tgn_provids_mlstm}"
MODELS="${MODELS:-tgn_provids,tgn_provids_mlstm}"

# Choose one or more message aggregators: last, mean, sequence.
MESSAGE_AGGREGATOR="${MESSAGE_AGGREGATOR:-sequence}"

# Choose one or more memory modes. 0 is plain TGN-style memory, 2 is m2/ProvIDS-enhanced memory.
MEMORY_ENHANCEMENT="${MEMORY_ENHANCEMENT:-2}"

# Choose one or more ProvIDS GNN layers.
# "model.num_units=$NUM_UNITS"
# NUM_UNITS="${NUM_UNITS:-2}"

# Number of concurrent Hydra jobs inside this shell/slurm allocation.
N_JOBS="${N_JOBS:-4}"

COMMON_OVERRIDES=(
  "model=$MODELS"
  "model.message_aggregator=$MESSAGE_AGGREGATOR"
  "model.memory_enhancement=$MEMORY_ENHANCEMENT"
  "hydra.launcher.n_jobs=$N_JOBS"
  "training.replay_memory_before_eval"=false
  "training.lr"=5e-5
  "training.clear_results"=false
)

if [[ "$DATASET" == "cause_effect" ]]; then
  # lag 4,8,16,32 \
  python -m T-GRAB.train.hydra_multirun --multirun \
    dataset=cause_effect \
    dataset.lag=4,8,16 \
    dataset.er_prob=0.002 \
    "${COMMON_OVERRIDES[@]}"
elif [[ "$DATASET" == "long_range" ]]; then
  python -m T-GRAB.train.hydra_multirun --multirun \
    dataset=long_range \
    dataset.lag=8,16,64 \
    dataset.branch_len=4,8,16 \
    dataset.num_branches=3,6 \
    "${COMMON_OVERRIDES[@]}"
elif [[ "$DATASET" == "ordered_long_range" ]]; then
  python -m T-GRAB.train.hydra_multirun --multirun \
    dataset=ordered_long_range \
    dataset.lag=8,16 \
    dataset.branch_len=8,16 \
    dataset.num_branches=6,8 \
    dataset.num_samples=1000 \
    training.val_first_metric=memnode_avg_ap \
    "${COMMON_OVERRIDES[@]}"
else
  echo "Unknown DATASET=$DATASET. Use DATASET=cause_effect, DATASET=long_range, or DATASET=ordered_long_range." >&2
  exit 1
fi
