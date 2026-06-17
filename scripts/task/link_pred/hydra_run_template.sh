#!/bin/bash

set -euo pipefail

# Run from anywhere inside this repository.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_PARENT="$(dirname "$REPO_DIR")"

source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate tgrab

cd "$PROJECT_PARENT"

# Keep Hydra/joblib runs from competing for CPU threads inside each process.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Choose one of: cause_effect, long_range
DATASET="${DATASET:-cause_effect}"

# Choose one or more model configs.
MODELS="${MODELS:-tgn_provids,tgn_provids_mlstm}"

# Number of concurrent Hydra jobs inside this shell/slurm allocation.
N_JOBS="${N_JOBS:-2}"

COMMON_OVERRIDES=(
  "model=$MODELS"
  "hydra.launcher.n_jobs=$N_JOBS"
)

if [[ "$DATASET" == "cause_effect" ]]; then
  python -m T-GRAB.train.hydra_multirun --multirun \
    dataset=cause_effect \
    dataset.lag=1,4,8,16,32,64 \
    dataset.er_prob=0.002 \
    "${COMMON_OVERRIDES[@]}"
elif [[ "$DATASET" == "long_range" ]]; then
  python -m T-GRAB.train.hydra_multirun --multirun \
    dataset=long_range \
    dataset.lag=8,16,64 \
    dataset.branch_len=4,8,16 \
    dataset.num_branches=3,6 \
    "${COMMON_OVERRIDES[@]}"
else
  echo "Unknown DATASET=$DATASET. Use DATASET=cause_effect or DATASET=long_range." >&2
  exit 1
fi
