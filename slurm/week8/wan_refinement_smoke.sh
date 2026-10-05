#!/bin/bash
#SBATCH --job-name=wan-refinement-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=wan-refinement-smoke-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week8/wan_refinement_smoke"
RUN_PREFIX="wan_refinement_smoke_${SLURM_JOB_ID:-manual}"
SUMMARY_CSV="$OUTPUT_DIR/${RUN_PREFIX}_summary.csv"

FASTWAN_MODEL_PATH="$SCRATCH/models/FastWan2.1-T2V-14B-Diffusers"
MODEL_ID="Wan-AI/Wan2.1-T2V-14B-Diffusers"
PROMPT="A red tram moves slowly through a sunlit city square"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
SEED=42
NUM_GPUS=1
ULYSSES_DEGREE=1
RING_DEGREE=1
DMD_DENOISING_STEPS="1000"
REFINE_STEPS=50
REFINE_SIGMA=0.75 # only retains 1 - refine_sigma of latent
DRAFT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_draft.pt"

# No limit on consecutive cache-dit hits.
export SGLANG_CACHE_DIT_MC=-1

setup_sglang_env

# 1. Generate the FastWan draft and save its latent.
LABEL="draft"
RUN_ID=1
NUM_INFERENCE_STEPS=1
PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_perf.json"
OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}.mp4"
LOG_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}.log"

echo "Starting FastWan draft"

sglang generate \
  --model-path "$FASTWAN_MODEL_PATH" \
  --model-id "$MODEL_ID" \
  --pipeline WanDMDPipeline \
  --num-gpus "$NUM_GPUS" \
  --sp-degree "$NUM_GPUS" \
  --ulysses-degree "$ULYSSES_DEGREE" \
  --ring-degree "$RING_DEGREE" \
  --encoder-parallel replicate \
  --prompt "$PROMPT" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" \
  --fps "$FPS" \
  --num-inference-steps "$NUM_INFERENCE_STEPS" \
  --seed "$SEED" \
  --dmd-denoising-steps "$DMD_DENOISING_STEPS" \
  --enable-cache-dit false \
  --wan-save-latent-path "$DRAFT_PATH" \
  --save-output \
  --output-file-path "$OUTPUT_PATH" \
  --perf-dump-path "$PERF_PATH" 2>&1 | tee "$LOG_PATH"

append_perf_summary
test -s "$DRAFT_PATH"

# 2. Refine the draft with base Wan and cache-dit.
LABEL="refined"
RUN_ID=2
NUM_INFERENCE_STEPS="$REFINE_STEPS"
PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_perf.json"
OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}.mp4"
LOG_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}.log"

echo "Starting Wan refinement"

sglang generate \
  --model-path "$MODEL_ID" \
  --num-gpus "$NUM_GPUS" \
  --sp-degree "$NUM_GPUS" \
  --ulysses-degree "$ULYSSES_DEGREE" \
  --ring-degree "$RING_DEGREE" \
  --encoder-parallel replicate \
  --prompt "$PROMPT" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" \
  --fps "$FPS" \
  --num-inference-steps "$NUM_INFERENCE_STEPS" \
  --seed "$SEED" \
  --enable-cache-dit false \
  --wan-init-latent-path "$DRAFT_PATH" \
  --wan-refine-sigma "$REFINE_SIGMA" \
  --save-output \
  --output-file-path "$OUTPUT_PATH" \
  --perf-dump-path "$PERF_PATH" 2>&1 | tee "$LOG_PATH"

append_perf_summary

echo "Smoke test completed. Outputs: $OUTPUT_DIR"
