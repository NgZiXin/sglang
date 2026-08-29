#!/bin/bash
#SBATCH --job-name=wan-t2v-sp-speed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=wan-t2v-sp-speed-%j.out

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

OUTPUT_DIR=${OUTPUT_DIR:-"$SCRATCH/sglang/outputs/sp_speedup"}
mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

RUN_PREFIX=${RUN_PREFIX:-"wan_t2v_sp_speedup_${SLURM_JOB_ID:-manual}"}
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
REQUIRED_GPUS=${REQUIRED_GPUS:-2}

# label,num_gpus,ulysses_degree,ring_degree
RUN_CONFIGS=(
  "baseline,1,1,1"
  "ulysses_2,2,2,1"
  "ring_2,2,1,2"
)

write_csv_header() {
  python3 - "$SUMMARY_CSV" <<'PY'
import csv
import sys

columns = [
    "label",
    "num_gpus",
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
    "status",
]

with open(sys.argv[1], "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=columns).writeheader()
PY
}

append_csv_row() {
  python3 - "$SUMMARY_CSV" "$PERF_PATH" "$LABEL" "$NUM_GPUS" "$ULYSSES_DEGREE" "$RING_DEGREE" "$RUN_ID" "$SEED" "$HEIGHT" "$WIDTH" "$NUM_FRAMES" "$FPS" "$NUM_INFERENCE_STEPS" "$OUTPUT_PATH" "$START_TIME" "$END_TIME" <<'PY'
import csv
import json
import sys

(
    summary_csv,
    perf_path,
    label,
    num_gpus,
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
    "status": "success",
}

for stage_name, column_name in stage_columns.items():
    row[column_name] = stage_durations.get(stage_name, "")

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writerow(row)
PY
}

source "$SCRATCH/sglang/venv/bin/activate"

nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_GPUS=${SLURM_GPUS:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

VISIBLE_GPUS=$(python3 - <<'PY'
import torch

print(torch.cuda.device_count())
PY
)

echo "torch.cuda.device_count()=${VISIBLE_GPUS}"

if (( VISIBLE_GPUS < REQUIRED_GPUS )); then
  echo "ERROR: This experiment needs ${REQUIRED_GPUS} visible GPU(s), but only ${VISIBLE_GPUS} are visible."
  echo "Check the Slurm GPU request syntax for NUS SOC. Try --gpus=2 or --gres=gpu:h100-96:2 if --gpus=h100-96:2 exposes only one GPU."
  exit 1
fi

write_csv_header

for CONFIG in "${RUN_CONFIGS[@]}"; do
  IFS=',' read -r LABEL NUM_GPUS ULYSSES_DEGREE RING_DEGREE <<< "$CONFIG"

  for RUN_ID in $(seq 1 "$REPEATS"); do
    PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_perf_run_${RUN_ID}.json"
    OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_run_${RUN_ID}.mp4"
    START_TIME=$(date -Is)

    echo "Starting ${LABEL} run ${RUN_ID}/${REPEATS}: num_gpus=${NUM_GPUS}, ulysses_degree=${ULYSSES_DEGREE}, ring_degree=${RING_DEGREE}"

    OUTPUT_ARGS=(--no-save-output)
    if [[ "$SAVE_OUTPUT" == "1" ]]; then
      OUTPUT_ARGS=(--save-output --output-file-path "$OUTPUT_PATH")
    fi

    sglang generate \
      --model-path "$MODEL_PATH" \
      --num-gpus "$NUM_GPUS" \
      --sp-degree "$NUM_GPUS" \
      --ulysses-degree "$ULYSSES_DEGREE" \
      --ring-degree "$RING_DEGREE" \
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

    END_TIME=$(date -Is)
    append_csv_row
  done
done

echo "Sequence-parallel speedup experiment completed."
echo "Summary CSV: $SUMMARY_CSV"
echo "Perf JSON files: $OUTPUT_DIR/${RUN_PREFIX}_*_perf_run_*.json"
