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


def prepare_inputs(video_dir, prompts):
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


def evaluate_videos(video_dir, full_info, work_dir):
    """Evaluate all 16 dimensions in one VBench API call."""
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
        videos_path=str(video_dir), name="result",
        dimension_list=list(SCORING_PARAMETERS),
    )

    results = json.loads((work_dir / "result_eval_results.json").read_text())
    scores = {}
    color_evaluation = None
    for dimension in SCORING_PARAMETERS:
        expected_videos = set()
        for prompt in full_info:
            if dimension in prompt["dimension"]:
                expected_videos.update(prompt["video_list"])
        aggregate, details = results[dimension]
        scored_videos = {item["video_path"] for item in details}
        if len(details) != len(scored_videos) or not scored_videos <= expected_videos:
            raise ValueError(f"{dimension}: duplicate or unexpected result videos")
        if dimension == "color":
            if not expected_videos:
                raise ValueError("color: no input videos")
            missing_videos = sorted(expected_videos - scored_videos)
            score = sum(float(item["video_results"]) for item in details) / len(expected_videos)
            color_evaluation = {
                "official_score": float(aggregate) if details else None,
                "missing_videos": missing_videos,
            }
            if missing_videos:
                print(f"color: counting {len(missing_videos)} missing video results as zero")
        else:
            if scored_videos != expected_videos:
                raise ValueError(f"{dimension}: results do not match its input videos")
            score = float(aggregate)
        if not math.isfinite(score):
            raise ValueError(f"{dimension}: non-finite score")
        scores[dimension] = score
    return scores, color_evaluation


def calculate_total_scores(scores):
    """Normalize the 16 aggregate scores and apply VBench's 80/20 weighting."""
    normalized = {}
    for dimension, (minimum, maximum, _) in SCORING_PARAMETERS.items():
        normalized[dimension] = (scores[dimension] - minimum) / (maximum - minimum)
    
    def weighted_average(scores, dimensions):
        weighted_sum = 0.0
        total_weight = 0.0
        for dimension in dimensions:
            weight = SCORING_PARAMETERS[dimension][2]
            weighted_sum += scores[dimension] * weight
            total_weight += weight
        return weighted_sum / total_weight

    quality = weighted_average(normalized, QUALITY_DIMENSIONS)
    semantic = weighted_average(normalized, SEMANTIC_DIMENSIONS)
    return {
        "quality_score": quality,
        "semantic_score": semantic,
        "total_score": (4 * quality + semantic) / 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    seed = manifest["seed"]
    video_dir = args.video_dir.resolve()
    full_info = prepare_inputs(video_dir, manifest["prompts"])

    # VBench's input and raw result JSON files are deleted after evaluation.
    with tempfile.TemporaryDirectory(prefix="vbench-") as temporary_dir:
        scores, color_evaluation = evaluate_videos(video_dir, full_info, Path(temporary_dir))

    summary = {
        "seed": seed,
        "num_videos": sum(len(prompt["video_list"]) for prompt in full_info),
        "dimension_scores": scores,
        "score_policy": "vbench_with_missing_color_videos_scored_zero",
        "color_evaluation": color_evaluation,
        **calculate_total_scores(scores),
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved VBench scores to {args.output_file}")


if __name__ == "__main__":
    main()
