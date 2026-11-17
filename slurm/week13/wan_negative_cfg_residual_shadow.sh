#!/bin/bash
#SBATCH --job-name=wan-negative-cfg-residual
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=wan-negative-cfg-residual-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week13/wan_negative_cfg_residual_shadow/${SLURM_JOB_ID:-manual}"
setup_sglang_env

python3 -u "$REPO_DIR/slurm/week13/wan_negative_cfg_residual_shadow.py" \
  --prompt-manifest "$REPO_DIR/slurm/week12/experiment4/p10_multiple_objects.json" \
  --output-dir "$OUTPUT_DIR" "$@" \
  2>&1 | tee "$OUTPUT_DIR/experiment.log"
