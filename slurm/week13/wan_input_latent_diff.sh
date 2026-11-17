#!/bin/bash
#SBATCH --job-name=wan-input-latent-diff
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=wan-input-latent-diff-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week13/wan_input_latent_diff/${SLURM_JOB_ID:-manual}"
setup_sglang_env

python3 -u "$REPO_DIR/slurm/week13/wan_input_latent_diff.py" \
  --prompt-manifest "$REPO_DIR/slurm/vbench_33_prompts_wan_chai_kv.json" \
  --output-csv "$OUTPUT_DIR/input_latent_diff.csv" "$@" \
  2>&1 | tee "$OUTPUT_DIR/experiment.log"
