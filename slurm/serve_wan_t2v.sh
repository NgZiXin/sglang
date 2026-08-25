#!/bin/bash
#SBATCH --job-name=wan-t2v-serve
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=wan-t2v-serve-%j.out

set -euo pipefail

cd ~/cp4101/week2/sglang

# Avoid cuDNN v8 frontend failures seen on some cluster CUDA/cuDNN stacks.
export TORCH_CUDNN_V8_API_DISABLED=1

export SCRATCH=/mnt/scratch/n/$USER
export HF_HOME=$SCRATCH/sglang/hf-cache
export TMPDIR=$SCRATCH/sglang/tmp
export XDG_CACHE_HOME=$SCRATCH/sglang/cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/sglang/torch-cache/inductor
export TRITON_CACHE_DIR=$SCRATCH/sglang/torch-cache/triton
mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$SCRATCH/sglang/benchmark"
RUN_PREFIX="wan_t2v_${SLURM_JOB_ID:-manual}"

source "$SCRATCH/sglang/venv/bin/activate"

nvidia-smi

echo "Starting Wan T2V server for Wan-AI/Wan2.1-T2V-14B-Diffusers"

sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --num-gpus 1 \
  --port 30000 > server.log 2>&1 &

SERVER_PID=$!

for i in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:30000/health >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

echo "Starting Wan T2V server is ready, running benchmark"

python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
  --dataset vbench \
  --task text-to-video \
  --num-prompts 1 \
  --max-concurrency 1 \
  --port 30000 \
  --output-file "$SCRATCH/sglang/benchmark/${RUN_PREFIX}_bench.json"

echo "Benchmark completed, shutting down server"

kill "$SERVER_PID" || true
wait "$SERVER_PID" || true
