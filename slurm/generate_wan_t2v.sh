#!/bin/bash
#SBATCH --job-name=wan-t2v
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=wan-t2v-%j.out

set -e

cd ~/cp4101/week2/sglang

# Avoid cuDNN v8 frontend failures seen on some cluster CUDA/cuDNN stacks.
export TORCH_CUDNN_V8_API_DISABLED=1

export SCRATCH=/mnt/scratch/n/$USER
export HF_HOME=$SCRATCH/sglang/hf-cache
export TMPDIR=$SCRATCH/sglang/tmp
export XDG_CACHE_HOME=$SCRATCH/sglang/cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/sglang/torch-cache/inductor
export TRITON_CACHE_DIR=$SCRATCH/sglang/torch-cache/triton
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1
mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$SCRATCH/sglang/outputs"
RUN_PREFIX="wan_t2v_${SLURM_JOB_ID:-manual}"

source "$SCRATCH/sglang/venv/bin/activate"

nvidia-smi

for RUN_ID in 1 2 3; do
  echo "Starting Wan T2V run ${RUN_ID}/3"

  sglang generate \
    --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --num-gpus 1 \
    --prompt "A red tram moves slowly through a sunlit city square" \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 50 \
    --save-output \
    --perf-dump-path "$SCRATCH/sglang/outputs/${RUN_PREFIX}_perf_run_${RUN_ID}.json" \
    --output-file-path "$SCRATCH/sglang/outputs/${RUN_PREFIX}_run_${RUN_ID}.mp4"
done
