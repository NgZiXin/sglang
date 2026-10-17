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

OUTPUT_DIR="$SCRATCH/sglang/outputs/week9/serving-concurrency-2"
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT=480
WIDTH=832
NUM_FRAMES=33
FPS=16
NUM_INFERENCE_STEPS=50
NUM_PROMPTS=2 # Total measured requests at each concurrency level.
PORT=30000

setup_sglang_env
# Synchronize stage/step timers to attribute GPU work accurately.
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1

RUN_DIR="$OUTPUT_DIR/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
NSYS_SESSION="wan_${SLURM_JOB_ID}"
NSYS_ACTIVE=0
nsys --version > "$RUN_DIR/nsys.log"
cp "$REPO_DIR/slurm/vbench_11_prompts.json" "$RUN_DIR/prompts.json"

# The serving benchmark expects one prompt per line.
python3 - "$RUN_DIR/prompts.json" > "$RUN_DIR/prompts.txt" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    for prompt in json.load(file)["prompts"]:
        print(prompt["prompt_en"])
PY

# Launch under Nsight now; start each capture after the server is ready.
nsys launch --session-new "$NSYS_SESSION" --trace cuda,nvtx \
  sglang serve \
  --model-path "$MODEL_PATH" \
  --num-gpus 1 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --batching-mode dynamic \
  --batching-max-size 3 \
  --batching-delay-ms 5 \
  --enable-batching-metrics \
  --enable-layerwise-nvtx-marker \
  > "$RUN_DIR/server.log" 2>&1 &
NSYS_PID=$!

# Finalize any active capture before shutting down the server process group.
cleanup() {
  if (( NSYS_ACTIVE )); then
    nsys stop --session "$NSYS_SESSION" >> "$RUN_DIR/nsys.log" 2>&1 || true
  fi
  nsys shutdown --session "$NSYS_SESSION" --kill sigterm >> "$RUN_DIR/nsys.log" 2>&1 || true
  wait "$NSYS_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Wait for /health before collecting, so model loading is excluded from the traces.
python3 - "$PORT" <<'PY'
import sys
from sglang.multimodal_gen.benchmarks.bench_serving import wait_for_service

wait_for_service(f"http://127.0.0.1:{sys.argv[1]}")
PY

# Use the same benchmark command for warmup and measurement.
run_benchmark() {
  local concurrency="$1" num_prompts="$2" label="$3"
  python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --model "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --task text-to-video \
    --dataset vbench \
    --dataset-path "$RUN_DIR/prompts.txt" \
    --num-prompts "$num_prompts" \
    --max-concurrency "$concurrency" \
    --request-rate inf \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --extra-body '{"seed": 42, "enable_cache_dit": false}' \
    --warmup-requests 0 \
    --output-file "$RUN_DIR/${label}.json" \
    2>&1 | tee "$RUN_DIR/${label}.log"
}

# Warm the actual concurrency before capture; built-in warmups are sequential.
# Nsight captures and measured statistics exclude these separate warmup runs.
for CONCURRENCY in 1 2; do
  echo "Warming concurrency=$CONCURRENCY"
  run_benchmark "$CONCURRENCY" "$CONCURRENCY" "warmup_${CONCURRENCY}"

  echo "Benchmarking $NUM_PROMPTS requests, concurrency=$CONCURRENCY"
  nsys start --session "$NSYS_SESSION" \
    --sample none --cpuctxsw none \
    --output "$RUN_DIR/concurrency_${CONCURRENCY}" >> "$RUN_DIR/nsys.log" 2>&1
  NSYS_ACTIVE=1
  run_benchmark "$CONCURRENCY" "$NUM_PROMPTS" "concurrency_${CONCURRENCY}"
  nsys stop --session "$NSYS_SESSION" >> "$RUN_DIR/nsys.log" 2>&1
  NSYS_ACTIVE=0
  test -s "$RUN_DIR/concurrency_${CONCURRENCY}.nsys-rep"
done

echo "Benchmark completed. Latency statistics, Nsight reports, and logs: $RUN_DIR"
