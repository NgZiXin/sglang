#!/usr/bin/env python3
"""Evaluate a VBench manifest and save one JSON summary."""

import argparse
import json
import math
import tempfile
from pathlib import Path

QUALITY_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]
SEMANTIC_DIMENSIONS = [
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]

# Official T2V constants: (reference minimum, reference maximum, weight).
# https://github.com/Vchitect/VBench/blob/master/scripts/constant.py
SCORING_PARAMETERS = {
    "subject_consistency": (0.1462, 1.0, 1.0),
    "background_consistency": (0.2615, 1.0, 1.0),
    "temporal_flickering": (0.6293, 1.0, 1.0),
    "motion_smoothness": (0.706, 0.9975, 1.0),
    "dynamic_degree": (0.0, 1.0, 0.5),
    "aesthetic_quality": (0.0, 1.0, 1.0),
    "imaging_quality": (0.0, 1.0, 1.0),
    "object_class": (0.0, 1.0, 1.0),
    "multiple_objects": (0.0, 1.0, 1.0),
    "human_action": (0.0, 1.0, 1.0),
    "color": (0.0, 1.0, 1.0),
    "spatial_relationship": (0.0, 1.0, 1.0),
    "scene": (0.0, 0.8222, 1.0),
    "appearance_style": (0.0009, 0.2855, 1.0),
    "temporal_style": (0.0, 0.364, 1.0),
    "overall_consistency": (0.0, 0.364, 1.0),
}


def prepare_inputs(video_dir: Path, prompts: list):
    """Require at least one available VBench sample for each prompt."""
    if not prompts:
        raise ValueError("The manifest has no prompts")

    full_info = []
    dimensions = set()
    for prompt in prompts:
        # Human-action evaluation reads its label from this filename.
        videos = []
        for index in range(5):
            video = video_dir / f"{prompt['prompt_en']}-{index}.mp4"
            if video.is_file():
                if video.stat().st_size == 0:
                    raise ValueError(f"Empty video: {video}")
                videos.append(str(video))
        if not videos:
            raise ValueError(f"No video found for prompt: {prompt['prompt_en']}")
        full_info.append({**prompt, "video_list": videos})
        dimensions.update(prompt["dimension"])

    if dimensions != set(SCORING_PARAMETERS):
        raise ValueError("The manifest must cover exactly the 16 VBench dimensions")
    return full_info


def evaluate_videos(video_dir: Path, full_info: list, work_dir: Path):
    """Evaluate all 16 dimensions in one VBench API call with safe score handling."""
    import torch
    from vbench import VBench

    metadata_path = work_dir / "videos.json"
    metadata_path.write_text(json.dumps(full_info))

    if not torch.cuda.is_available():
        raise RuntimeError("VBench evaluation requires a CUDA GPU")

    benchmark = VBench(
        torch.device("cuda"), str(metadata_path), str(work_dir)
    )
    benchmark.evaluate(
        videos_path=str(video_dir),
        name="result",
        dimension_list=list(SCORING_PARAMETERS),
    )

    results = json.loads((work_dir / "result_eval_results.json").read_text())
    scores = {}

    for dimension in SCORING_PARAMETERS:
        expected_videos = set()
        for prompt in full_info:
            if dimension in prompt["dimension"]:
                expected_videos.update(prompt["video_list"])

        aggregate, details = results[dimension]
        scored_videos = {item["video_path"] for item in details}

        # Check 1: Ensure no duplicate video entries in details
        if len(details) != len(scored_videos):
            raise ValueError(f"{dimension}: duplicate video entries in result details")

        # Check 2: Ensure all scored videos are part of the expected set
        if not scored_videos.issubset(expected_videos):
            unexpected = scored_videos - expected_videos
            raise ValueError(f"{dimension}: contains {len(unexpected)} unexpected video paths")

        # Dimension-specific evaluation logic
        if dimension == "color":
            if not expected_videos:
                raise ValueError("color: no input videos defined in manifest")

            missing_videos = sorted(expected_videos - scored_videos)

            # Safeguard: Official VBench averages only over detected videos.
            # If no videos had detections (details is empty), aggregate is null or NaN.
            if details and aggregate is not None and not math.isnan(float(aggregate)):
                score = float(aggregate)
            else:
                score = 0.0

            if missing_videos:
                print(
                    f"[Notice] color: {len(missing_videos)}/{len(expected_videos)} "
                    f"videos lacked object detection by GRiT. "
                    f"Official score is averaged over the remaining {len(details)} detected videos."
                )

        else:
            # For all 15 other dimensions, 100% of the expected videos must be evaluated
            if scored_videos != expected_videos:
                missing = expected_videos - scored_videos
                raise ValueError(
                    f"{dimension}: evaluated {len(scored_videos)}/{len(expected_videos)} videos. "
                    f"Missing: {sorted(missing)[:3]}..."
                )
            score = float(aggregate if aggregate is not None else 0.0)

            # Safeguard: MUSIQ outputs imaging quality on [0, 100].
            # Official VBench divides by 100.0 before normalization.
            if dimension == "imaging_quality" and score > 1.0:
                score = score / 100.0

        if not math.isfinite(score):
            raise ValueError(f"{dimension}: non-finite score encountered ({score})")

        scores[dimension] = score

    return scores


def calculate_total_scores(scores: dict):
    """Normalize the 16 aggregate scores and apply VBench's 80/20 category weighting."""
    normalized = {}

    for dimension, (minimum, maximum, _) in SCORING_PARAMETERS.items():
        val = scores[dimension]
        denom = maximum - minimum
        if denom == 0.0:
            norm_val = 0.0
        else:
            norm_val = (val - minimum) / denom

        normalized[dimension] = norm_val

    def weighted_average(scores_dict: dict, dimension_group: list):
        weighted_sum = 0.0
        total_weight = 0.0
        for dim in dimension_group:
            weight = SCORING_PARAMETERS[dim][2]
            weighted_sum += scores_dict[dim] * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0.0 else 0.0

    quality = weighted_average(normalized, QUALITY_DIMENSIONS)
    semantic = weighted_average(normalized, SEMANTIC_DIMENSIONS)

    return {
        "quality_score": quality,
        "semantic_score": semantic,
        "total_score": (4.0 * quality + semantic) / 5.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    seed = manifest.get("seed", 0)
    video_dir = args.video_dir.resolve()
    full_info = prepare_inputs(video_dir, manifest["prompts"])

    # VBench's input and raw result JSON files are deleted after evaluation
    with tempfile.TemporaryDirectory(prefix="vbench-") as temporary_dir:
        scores = evaluate_videos(video_dir, full_info, Path(temporary_dir))

    summary = {
        "seed": seed,
        "num_videos": sum(len(prompt["video_list"]) for prompt in full_info),
        "dimension_scores": scores,
        **calculate_total_scores(scores),
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved VBench scores to {args.output_file}")


if __name__ == "__main__":
    main()