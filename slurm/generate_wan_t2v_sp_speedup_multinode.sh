#!/bin/bash
#SBATCH --job-name=wan-t2v-sp-mn
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-47:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --output=wan-t2v-sp-mn-%j.out

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
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1

OUTPUT_DIR=${OUTPUT_DIR:-"$SCRATCH/sglang/outputs/sp_speedup_multinode"}
mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

RUN_PREFIX=${RUN_PREFIX:-"wan_t2v_sp_speedup_multinode_${SLURM_JOB_ID:-manual}"}
SUMMARY_CSV="$OUTPUT_DIR/${RUN_PREFIX}_summary.csv"

MODEL_PATH=${MODEL_PATH:-"Wan-AI/Wan2.1-T2V-14B-Diffusers"}
PROMPT=${PROMPT:-"A red tram moves slowly through a sunlit city square"}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-81}
FPS=${FPS:-16}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
SEED=${SEED:-42}
REPEATS=${REPEATS:-3}
SAVE_OUTPUT=${SAVE_OUTPUT:-1}
NNODES=${NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
DIST_PORT_BASE=${DIST_PORT_BASE:-23456}
WORKER_STARTUP_DELAY=${WORKER_STARTUP_DELAY:-10}

# label,num_gpus,ulysses_degree,ring_degree
RUN_CONFIGS=(
  "ring_4,4,1,4"
  "ulysses_4,4,4,1"
  "ulysses2_ring2,4,2,2"
)

write_csv_header() {
  python3 - "$SUMMARY_CSV" <<'PY'
import csv
import sys

columns = [
    "label",
    "num_gpus",
    "nnodes",
    "gpus_per_node",
    "ulysses_degree",
    "ring_degree",
    "run_id",
    "seed",
    "height",
    "width",
    "num_frames",
    "fps",
    "num_inference_steps",
    "total_duration_ms",
    "input_validation_ms",
    "text_encoding_ms",
    "latent_preparation_ms",
    "timestep_preparation_ms",
    "denoising_ms",
    "decoding_ms",
    "scheduler_return_result_spill_arrays_ms",
    "scheduler_client_materialize_file_refs_ms",
    "stage_durations_json",
    "perf_path",
    "output_path",
    "start_time",
    "end_time",
    "dist_init_addr",
    "status",
]

with open(sys.argv[1], "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=columns).writeheader()
PY
}

append_csv_row() {
  python3 - "$SUMMARY_CSV" "$PERF_PATH" "$LABEL" "$NUM_GPUS" "$NNODES" "$GPUS_PER_NODE" "$ULYSSES_DEGREE" "$RING_DEGREE" "$RUN_ID" "$SEED" "$HEIGHT" "$WIDTH" "$NUM_FRAMES" "$FPS" "$NUM_INFERENCE_STEPS" "$OUTPUT_PATH" "$START_TIME" "$END_TIME" "$DIST_INIT_ADDR" <<'PY'
import csv
import json
import sys

(
    summary_csv,
    perf_path,
    label,
    num_gpus,
    nnodes,
    gpus_per_node,
    ulysses_degree,
    ring_degree,
    run_id,
    seed,
    height,
    width,
    num_frames,
    fps,
    num_inference_steps,
    output_path,
    start_time,
    end_time,
    dist_init_addr,
) = sys.argv[1:]

stage_columns = {
    "InputValidationStage": "input_validation_ms",
    "TextEncodingStage": "text_encoding_ms",
    "LatentPreparationStage": "latent_preparation_ms",
    "TimestepPreparationStage": "timestep_preparation_ms",
    "DenoisingStage": "denoising_ms",
    "DecodingStage": "decoding_ms",
    "Scheduler.return_result.spill_arrays": "scheduler_return_result_spill_arrays_ms",
    "SchedulerClient.materialize_file_refs": "scheduler_client_materialize_file_refs_ms",
}

columns = [
    "label",
    "num_gpus",
    "nnodes",
    "gpus_per_node",
    "ulysses_degree",
    "ring_degree",
    "run_id",
    "seed",
    "height",
    "width",
    "num_frames",
    "fps",
    "num_inference_steps",
    "total_duration_ms",
    "input_validation_ms",
    "text_encoding_ms",
    "latent_preparation_ms",
    "timestep_preparation_ms",
    "denoising_ms",
    "decoding_ms",
    "scheduler_return_result_spill_arrays_ms",
    "scheduler_client_materialize_file_refs_ms",
    "stage_durations_json",
    "perf_path",
    "output_path",
    "start_time",
    "end_time",
    "dist_init_addr",
    "status",
]

with open(perf_path, "r", encoding="utf-8") as f:
    perf = json.load(f)

stage_durations = {}
for item in perf.get("steps", []):
    name = item.get("name")
    if name:
        stage_durations[name] = item.get("duration_ms", "")

row = {
    "label": label,
    "num_gpus": num_gpus,
    "nnodes": nnodes,
    "gpus_per_node": gpus_per_node,
    "ulysses_degree": ulysses_degree,
    "ring_degree": ring_degree,
    "run_id": run_id,
    "seed": seed,
    "height": height,
    "width": width,
    "num_frames": num_frames,
    "fps": fps,
    "num_inference_steps": num_inference_steps,
    "total_duration_ms": perf.get("total_duration_ms", ""),
    "stage_durations_json": json.dumps(stage_durations, sort_keys=True),
    "perf_path": perf_path,
    "output_path": output_path,
    "start_time": start_time,
    "end_time": end_time,
    "dist_init_addr": dist_init_addr,
    "status": "success",
}

for stage_name, column_name in stage_columns.items():
    row[column_name] = stage_durations.get(stage_name, "")

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writerow(row)
PY
}

cleanup_worker() {
  if [[ -n "${WORKER_SRUN_PID:-}" ]]; then
    kill "$WORKER_SRUN_PID" >/dev/null 2>&1 || true
    wait "$WORKER_SRUN_PID" >/dev/null 2>&1 || true
    WORKER_SRUN_PID=""
  fi
}

trap cleanup_worker EXIT

source "$SCRATCH/sglang/venv/bin/activate"

mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if (( ${#ALLOCATED_NODES[@]} < NNODES )); then
  echo "ERROR: Expected ${NNODES} allocated node(s), got ${#ALLOCATED_NODES[@]}."
  printf 'Allocated nodes: %s\n' "${ALLOCATED_NODES[@]}"
  exit 1
fi

HEAD_NODE=${ALLOCATED_NODES[0]}
WORKER_NODE=${ALLOCATED_NODES[1]}

echo "Allocated nodes:"
printf '  %s\n' "${ALLOCATED_NODES[@]}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "WORKER_NODE=${WORKER_NODE}"
echo "TOTAL_GPUS=${TOTAL_GPUS}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"

write_csv_header

CONFIG_INDEX=0
for CONFIG in "${RUN_CONFIGS[@]}"; do
  IFS=',' read -r LABEL NUM_GPUS ULYSSES_DEGREE RING_DEGREE <<< "$CONFIG"

  if (( NUM_GPUS != TOTAL_GPUS )); then
    echo "ERROR: ${LABEL} requests ${NUM_GPUS} total GPU(s), but this script allocated ${TOTAL_GPUS}."
    exit 1
  fi

  for RUN_ID in $(seq 1 "$REPEATS"); do
    PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_perf_run_${RUN_ID}.json"
    OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_run_${RUN_ID}.mp4"
    WORKER_LOG="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_run_${RUN_ID}_worker.log"
    DIST_INIT_ADDR="${HEAD_NODE}:$((DIST_PORT_BASE + CONFIG_INDEX * 100 + RUN_ID))"
    START_TIME=$(date -Is)

    export MODEL_PATH PROMPT HEIGHT WIDTH NUM_FRAMES FPS NUM_INFERENCE_STEPS SEED SAVE_OUTPUT
    export NUM_GPUS NNODES GPUS_PER_NODE ULYSSES_DEGREE RING_DEGREE DIST_INIT_ADDR
    export PERF_PATH OUTPUT_PATH

    echo "Starting ${LABEL} run ${RUN_ID}/${REPEATS}: num_gpus=${NUM_GPUS}, nnodes=${NNODES}, gpus_per_node=${GPUS_PER_NODE}, ulysses_degree=${ULYSSES_DEGREE}, ring_degree=${RING_DEGREE}, dist_init_addr=${DIST_INIT_ADDR}"

    srun --exclusive --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:h100-96:${GPUS_PER_NODE} --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" --nodelist="$WORKER_NODE" --export=ALL,NODE_RANK_VALUE=1 bash -lc '
      set -euo pipefail
      cd ~/cp4101/week2/sglang
      source "$SCRATCH/sglang/venv/bin/activate"
      echo "[worker] hostname=$(hostname)"
      echo "[worker] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
      python3 - <<'"'"'PY'"'"'
import torch

print(f"[worker] torch.cuda.device_count()={torch.cuda.device_count()}")
PY
      sglang serve \
        --model-path "$MODEL_PATH" \
        --num-gpus "$NUM_GPUS" \
        --nnodes "$NNODES" \
        --node-rank "$NODE_RANK_VALUE" \
        --dist-init-addr "$DIST_INIT_ADDR" \
        --sp-degree "$NUM_GPUS" \
        --ulysses-degree "$ULYSSES_DEGREE" \
        --ring-degree "$RING_DEGREE" \
        --encoder-parallel replicate \
        --cfg-parallel-size 1
    ' > "$WORKER_LOG" 2>&1 &

    WORKER_SRUN_PID=$!
    sleep "$WORKER_STARTUP_DELAY"

    srun --exclusive --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:h100-96:${GPUS_PER_NODE} --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" --nodelist="$HEAD_NODE" --export=ALL,NODE_RANK_VALUE=0 bash -lc '
      set -euo pipefail
      cd ~/cp4101/week2/sglang
      source "$SCRATCH/sglang/venv/bin/activate"
      echo "[head] hostname=$(hostname)"
      echo "[head] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
      python3 - <<'"'"'PY'"'"'
import torch

print(f"[head] torch.cuda.device_count()={torch.cuda.device_count()}")
PY
      OUTPUT_ARGS=(--no-save-output)
      if [[ "$SAVE_OUTPUT" == "1" ]]; then
        OUTPUT_ARGS=(--save-output --output-file-path "$OUTPUT_PATH")
      fi

      sglang generate \
        --model-path "$MODEL_PATH" \
        --num-gpus "$NUM_GPUS" \
        --nnodes "$NNODES" \
        --node-rank "$NODE_RANK_VALUE" \
        --dist-init-addr "$DIST_INIT_ADDR" \
        --sp-degree "$NUM_GPUS" \
        --ulysses-degree "$ULYSSES_DEGREE" \
        --ring-degree "$RING_DEGREE" \
        --encoder-parallel replicate \
        --cfg-parallel-size 1 \
        --prompt "$PROMPT" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --num-frames "$NUM_FRAMES" \
        --fps "$FPS" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --seed "$SEED" \
        "${OUTPUT_ARGS[@]}" \
        --perf-dump-path "$PERF_PATH"
    '

    END_TIME=$(date -Is)
    append_csv_row
    cleanup_worker
  done

  CONFIG_INDEX=$((CONFIG_INDEX + 1))
done

echo "Multi-node sequence-parallel speedup experiment completed."
echo "Summary CSV: $SUMMARY_CSV"
echo "Perf JSON files: $OUTPUT_DIR/${RUN_PREFIX}_*_perf_run_*.json"
