#!/usr/bin/env python3
"""Append one SGLang perf dump to a CSV summary."""

import csv
import json
import sys
from pathlib import Path

# Parse command line arguments
summary_csv = Path(sys.argv[1])
perf_path = Path(sys.argv[2])
(
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
) = sys.argv[3:]

# Read the perf JSON file and construct a row for the CSV summary
with perf_path.open("r", encoding="utf-8") as f:
    perf = json.load(f)

stage_durations = {
    step["name"]: step.get("duration_ms", "")
    for step in perf.get("steps", [])
    if step.get("name")
}

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
    "input_validation_ms": stage_durations.get("InputValidationStage", ""),
    "text_encoding_ms": stage_durations.get("TextEncodingStage", ""),
    "latent_preparation_ms": stage_durations.get("LatentPreparationStage", ""),
    "timestep_preparation_ms": stage_durations.get("TimestepPreparationStage", ""),
    "denoising_ms": stage_durations.get(
        "DenoisingStage", stage_durations.get("DmdDenoisingStage", "")
    ),
    "decoding_ms": stage_durations.get("DecodingStage", ""),
    "scheduler_return_result_spill_arrays_ms": stage_durations.get(
        "Scheduler.return_result.spill_arrays", ""
    ),
    "scheduler_client_materialize_file_refs_ms": stage_durations.get(
        "SchedulerClient.materialize_file_refs", ""
    ),
    "stage_durations_json": json.dumps(stage_durations, sort_keys=True),
    "perf_path": str(perf_path),
}

# Append the row to the CSV summary file
with summary_csv.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if not summary_csv.exists():
        writer.writeheader()
    writer.writerow(row)
