#!/bin/bash
#SBATCH --job-name=wan-t2v-concurrency
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=wan-t2v-concurrency-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week9/serving-concurrency"
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
NUM_INFERENCE_STEPS=50
NUM_PROMPTS=2 # Total measured requests at each concurrency level.
PORT=30000

setup_sglang_env

RUN_DIR="$OUTPUT_DIR/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
cp "$REPO_DIR/slurm/vbench_11_prompts.json" "$RUN_DIR/prompts.json"

# The serving benchmark expects one prompt per line.
python3 - "$RUN_DIR/prompts.json" > "$RUN_DIR/prompts.txt" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    for prompt in json.load(file)["prompts"]:
        print(prompt["prompt_en"])
PY

sglang serve \
  --model-path "$MODEL_PATH" \
  --num-gpus 1 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --batching-mode dynamic \
  --batching-max-size 3 \
  --batching-delay-ms 5 \
  --enable-batching-metrics \
  > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!

# Stop the server even if a benchmark fails.
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Reuse the same server and prompts for all three concurrency levels.
# The benchmark waits for /health and excludes warmup from its measurements.
# Latency includes server queueing and generation, with completion polled every second.
for CONCURRENCY in 1 2; do
  echo "Benchmarking $NUM_PROMPTS requests, concurrency=$CONCURRENCY"
  python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --model "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --task text-to-video \
    --dataset vbench \
    --dataset-path "$RUN_DIR/prompts.txt" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --request-rate inf \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --extra-body '{"seed": 42, "enable_cache_dit": false}' \
    --warmup-requests 1 \
    --warmup-inference-steps "$NUM_INFERENCE_STEPS" \
    --output-file "$RUN_DIR/concurrency_${CONCURRENCY}.json" \
    2>&1 | tee "$RUN_DIR/concurrency_${CONCURRENCY}.log"
done

echo "Benchmark completed. Latency statistics (seconds) and logs: $RUN_DIR"
