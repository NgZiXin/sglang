#!/bin/bash
#SBATCH --job-name=wan-t2v-cache-dit
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7:30:00
#SBATCH --output=wan-t2v-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

# Scheduler params
FLOW_SHIFT=5 # Model default; sets the range before Beta sampling.

OUTPUT_DIR="$SCRATCH/sglang/outputs/week10/vary_denoise_cache_dit_2"
RUN_ID=1 # Required by perf summary csv
CACHE_MISS_RUNS_CSV="$OUTPUT_DIR/cache_miss_runs_${SLURM_JOB_ID}.csv"

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
NUM_GPUS=1
ULYSSES_DEGREE=1
RING_DEGREE=1
WAN_TIMESTEP_SCHEDULE=0 # 0=flow shift, 1=early-fine, 2=Beta(0.6, 0.6), 3=Beta(0.6, 0.4).

# Fixed cache settings
export CACHE_DIT_LOG_LEVEL=DEBUG
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_MC=-1
export SGLANG_CACHE_DIT_RDT=0.48
export SGLANG_CACHE_DIT_TAYLORSEER=false
export SGLANG_CACHE_DIT_TS_ORDER=1
export SGLANG_CACHE_DIT_SCM_PRESET=none
export SGLANG_CACHE_DIT_SCM_POLICY=dynamic

record_cache_miss_run() {
  python3 - "$1" "$CACHE_MISS_RUNS_CSV" "$NUM_INFERENCE_STEPS" "$LABEL" "$SEED" <<'PY'
import csv
import re
import sys

log_path, csv_path, steps, label, seed = sys.argv[1:]
pattern = re.compile(
    r"cache-dit summary: .*?context=(\d+) hits=(\d+) calls=(\d+)"
)
with open(log_path, encoding="utf-8") as log_file:
    stats = [
        (int(context), int(hits), int(calls))
        for line in log_file
        if (match := pattern.search(line))
        for context, hits, calls in [match.groups()]
    ]
if not stats:
    raise SystemExit(f"ERROR: No Cache-DiT summary found in {log_path}")

counts = {(hits, calls) for _, hits, calls in stats}
if len(counts) != 1:
    raise SystemExit(
        f"ERROR: Cache-DiT context summaries disagree in {log_path}: {stats}"
    )

# Cache-DiT logs the same request totals once per context. Use one context
# rather than summing those duplicate summaries.
_, hits, calls = stats[0]
if hits > calls:
    raise SystemExit(
        f"ERROR: Cache-DiT hits ({hits}) exceed transformer calls ({calls}) in {log_path}"
    )

with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([steps, label, seed, hits, calls, calls - hits])
PY
}

setup_sglang_env
printf 'num_inference_steps,prompt_id,seed,cache_hits,transformer_calls,cache_misses\n' \
  > "$CACHE_MISS_RUNS_CSV"

for NUM_INFERENCE_STEPS in 40; do
RUN_DIR="$OUTPUT_DIR/steps_${NUM_INFERENCE_STEPS}_schedule_${WAN_TIMESTEP_SCHEDULE}_shift_${FLOW_SHIFT}_${SLURM_JOB_ID}"
PROMPT_MANIFEST="$RUN_DIR/prompts.json"
SUMMARY_CSV="$RUN_DIR/perf_summary.csv"
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/videos" "$RUN_DIR/perf" "$RUN_DIR/logs"
# Save the prompt manifest to the run directory for reproducibility.
cp "$REPO_DIR/slurm/week10/vbench_33_prompts_wan_chai_kv.json" "$PROMPT_MANIFEST"

# Bash reads the prompt rows directly from this inline JSON reader.
PROMPTS=$(python3 - "$PROMPT_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    manifest = json.load(file)
for prompt in manifest["prompts"]:
    print(prompt["prompt_id"], prompt["prompt_en"],
          manifest["seed"], manifest["sample_index"], sep="\t")
PY
)

while IFS=$'\t' read -r LABEL PROMPT SEED SAMPLE_INDEX; do
  FILE_NAME="${LABEL}_seed${SEED}"
  PERF_PATH="$RUN_DIR/perf/$FILE_NAME.json"
  GENERATED_VIDEO="$RUN_DIR/videos/$FILE_NAME.mp4"
  VIDEO_PATH="$RUN_DIR/videos/${PROMPT}-${SAMPLE_INDEX}.mp4" # VBench filename format.
  LOG_PATH="$RUN_DIR/logs/$FILE_NAME.log"

  echo "Generating ${LABEL}, seed=${SEED}: ${PROMPT}"
  sglang generate \
    --model-path "$MODEL_PATH" \
    --num-gpus "$NUM_GPUS" \
    --sp-degree "$NUM_GPUS" \
    --ulysses-degree "$ULYSSES_DEGREE" \
    --ring-degree "$RING_DEGREE" \
    --prompt "$PROMPT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --flow-shift "$FLOW_SHIFT" \
    --wan-timestep-schedule "$WAN_TIMESTEP_SCHEDULE" \
    --seed "$SEED" \
    --enable-cache-dit true \
    --save-output \
    --perf-dump-path "$PERF_PATH" \
    --output-file-path "$GENERATED_VIDEO" 2>&1 | tee "$LOG_PATH"
  if ! test -s "$GENERATED_VIDEO"; then
    echo "ERROR: Generated video is missing or empty: $GENERATED_VIDEO" >&2
    exit 1
  fi
  record_cache_miss_run "$LOG_PATH"
  # Rename after saving because SGLang replaces spaces and punctuation.
  mv "$GENERATED_VIDEO" "$VIDEO_PATH"
  append_perf_summary
done <<< "$PROMPTS"

# Keep evaluation code available for when VBench is needed again.
set +e
(
  set -e
  run_vbench_evaluation \
    --video-dir "$RUN_DIR/videos" \
    --manifest "$PROMPT_MANIFEST" \
    --output-file "$RUN_DIR/vbench_scores.json"
) 2>&1 | tee "$RUN_DIR/logs/vbench.log"
EVALUATION_STATUS=("${PIPESTATUS[@]}")
set -e

if (( EVALUATION_STATUS[0] == 0 && EVALUATION_STATUS[1] == 0 )); then
  printf '{"status":"completed","evaluation_exit_code":0,"log_exit_code":0,"log_file":"logs/vbench.log"}\n' \
    > "$RUN_DIR/vbench_status.json"
  echo "Generation and evaluation completed: $RUN_DIR"
else
  printf '{"status":"failed","evaluation_exit_code":%d,"log_exit_code":%d,"log_file":"logs/vbench.log"}\n' \
    "${EVALUATION_STATUS[0]}" "${EVALUATION_STATUS[1]}" > "$RUN_DIR/vbench_status.json"
  echo "WARNING: Evaluation failed for ${NUM_INFERENCE_STEPS} steps; generated videos are saved. See $RUN_DIR/logs/vbench.log. Continuing the sweep." >&2
fi
done

echo "Cache miss data: $CACHE_MISS_RUNS_CSV"
