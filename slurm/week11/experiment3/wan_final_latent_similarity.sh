#!/bin/bash
#SBATCH --job-name=wan-final-latent-similarity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH --output=wan-final-latent-similarity-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week11/experiment3/${SLURM_JOB_ID:-manual}"
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

setup_sglang_env
mkdir -p "$OUTPUT_DIR"
cp "$REPO_DIR/slurm/week10/vbench_33_prompts_wan_chai_kv.json" "$OUTPUT_DIR/prompts.json"

# Compare final normalized latents, before VAE scaling/decoding, for all 33 prompts.
python3 -u "$REPO_DIR/slurm/week11/experiment3/wan_final_latent_similarity.py" \
  --model-path "$MODEL_PATH" \
  --prompt-manifest "$OUTPUT_DIR/prompts.json" \
  --output-csv "$OUTPUT_DIR/similarity.csv" \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --flow-shift 8 \
  --guidance-scale 6 2>&1 | tee "$OUTPUT_DIR/experiment.log"

echo "Results: $OUTPUT_DIR/similarity.csv"
