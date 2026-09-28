#!/bin/bash
#SBATCH --job-name=wan-cache-vbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=wan-cache-vbench-%j.out

set -euo pipefail

# Paths: change the environment path here if you installed elsewhere.
export SCRATCH="/mnt/scratch/n/$USER"
source "$SCRATCH/vbench/bin/activate"

VIDEO_DIR="$SCRATCH/sglang/outputs/week7/cache_dit_csv_2"
RESULT_DIR="$VIDEO_DIR/vbench_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

# Evaluate the folder, then combine individual video scores into one CSV.
python - "$VIDEO_DIR" "$RESULT_DIR" <<'PY'
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

video_dir, result_dir = map(Path, sys.argv[1:])
dimensions = [
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality",
]

# Read prompt, seed and MC from each filename, and latency from its perf JSON.
rows = {}
for video in sorted(video_dir.glob("*.mp4")):
    match = re.search(r"_(p\d+_.+)_seed(\d+)_mc(-?\d+)_run_\d+$", video.stem)
    if not match:
        raise ValueError(f"Unexpected video filename: {video.name}")
    prompt, seed, mc = match.groups()
    perf = video.with_name(video.stem.replace("_run_", "_perf_run_") + ".json")
    latency = json.loads(perf.read_text())["total_duration_ms"] / 1000 if perf.exists() else ""
    rows[video.name] = dict(video=video.name, prompt_id=prompt,
                           seed=int(seed), mc=int(mc), latency_s=latency)
if not rows:
    raise ValueError(f"No MP4 videos found in {video_dir}")

# Each metric processes all videos together, loading its model once.
for dimension in dimensions:
    metric_dir = result_dir / dimension
    subprocess.run([
        "vbench", "evaluate", "--ngpus", "1",
        "--videos_path", str(video_dir), "--mode", "custom_input",
        "--dimension", dimension, "--output_path", str(metric_dir),
    ], check=True)

    # Require exactly one result file and a score for every input video.
    result_file, = metric_dir.glob("*_eval_results.json")
    results = json.loads(result_file.read_text())[dimension][1]
    scores = {Path(item["video_path"]).name: float(item["video_results"])
              for item in results}
    if scores.keys() != rows.keys():
        raise ValueError(f"{dimension}: results do not match the input videos")
    for name, score in scores.items():
        # VBench's per-video imaging scores need /100 to match its aggregate scale.
        rows[name][dimension] = score / 100 if dimension == "imaging_quality" else score

csv_path = result_dir / "video_scores.csv"
with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["video", "prompt_id", "seed", "mc", "latency_s"] + dimensions)
    writer.writeheader()
    writer.writerows(rows.values())
print(f"Saved {len(rows)} video scores to {csv_path}")
PY
