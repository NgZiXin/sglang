"""Test same-timestep negative CFG reuse between prompt_a and prompt_en.

Uses the metric and generation defaults from wan_input_latent_diff.py.
Defaults to the three prompt pairs in week12/experiment4/p10_multiple_objects.json.
For each pair, saves three decoded videos: prompt_a, prompt_en_full, and
prompt_en_reuse. The last run compares its CURRENT model input against prompt_a
at each step and skips the negative forward only when relative mean L1 < 0.05.

Reference inputs and raw negative predictions are cached on CPU for one pair.
Timings include CPU comparisons/cache transfers and video decoding/encoding;
they are experiment timings, not isolated transformer benchmarks.
"""

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from unittest.mock import patch

from wan_input_latent_diff import latent_metrics, load_pairs


STEP_FIELDS = [
    "prompt_id", "seed", "run", "step", "timestep", "mean_l1",
    "relative_mean_l1", "threshold", "below_threshold", "negative_reused",
]
RUN_FIELDS = [
    "prompt_id", "seed", "run", "steps", "positive_forwards",
    "negative_forwards", "negative_reused", "reuse_fraction", "elapsed_seconds",
    "video_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-manifest", type=Path,
        default=Path(__file__).resolve().parents[1] / "week12" / "experiment4" / "p10_multiple_objects.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("wan_negative_cfg_reuse"))
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt-id", help="Run just this pair; defaults to all pairs")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-threshold", type=float, default=0.05)
    args = parser.parse_args()
    try:
        args.prompts = load_pairs(args.prompt_manifest, args.prompt_id)
        for row in args.prompts:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", row["prompt_id"]):
                raise ValueError("prompt_id must contain only letters, digits, _ or -")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid prompt selection: {error}")
    if min(args.height, args.width, args.num_frames, args.fps, args.num_inference_steps) < 1:
        parser.error("Dimensions, frame count, fps and inference steps must be positive")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale <= 1:
        parser.error("--guidance-scale must be finite and > 1 to enable both CFG branches")
    if not math.isfinite(args.reuse_threshold) or args.reuse_threshold < 0:
        parser.error("--reuse-threshold must be finite and nonnegative")
    return args


class NegativeReuseHook:
    """Intercept raw branch outputs, preserving the pipeline's CFG combination."""

    def __init__(self, stage, args, mode, reference):
        if mode not in ("prompt_a", "prompt_en_full", "prompt_en_reuse"):
            raise ValueError(f"Unknown mode: {mode}")
        self.stage = stage
        self.args = args
        self.mode = mode
        self.reference = reference
        self.predict_cfg = stage._predict_noise_with_cfg
        self.predict_branch = stage._predict_noise
        self.rows = []
        self.positive_forwards = 0
        self.negative_forwards = 0
        self.reused = 0

    def __call__(self, *, latent_model_input, timestep_index, **kwargs):
        import torch

        if timestep_index != len(self.rows) or timestep_index >= self.args.num_inference_steps:
            raise RuntimeError(f"Unexpected denoising step index: {timestep_index}")
        policy = kwargs["cfg_policy"]
        if (
            kwargs["server_args"].enable_cfg_parallel
            or len(policy.branches) != 2
            or not policy.branches[0].is_conditional
            or policy.branches[1].is_conditional
        ):
            raise RuntimeError("This experiment requires serial positive/negative CFG")
        if kwargs.get("cfg_gate_state") and kwargs["cfg_gate_state"]["active"]:
            raise RuntimeError("Other CFG gating must be disabled for this experiment")

        latent = latent_model_input.detach().to(device="cpu", copy=True)
        timestep = kwargs["timestep"].detach().to(device="cpu", copy=True)
        mean_l1 = relative_l1 = None
        below_threshold = reuse = False
        if self.mode == "prompt_a":
            entry = {"input_latent": latent, "timestep": timestep}
        else:
            entry = self.reference[timestep_index]
            if not torch.equal(timestep, entry["timestep"]):
                raise RuntimeError(f"Step {timestep_index + 1}: schedules differ")
            if timestep_index == 0 and not torch.equal(latent, entry["input_latent"]):
                raise RuntimeError("Initial latents differ despite matching seeds")
            # Exactly the float32 CPU / symmetric relative-L1 metric used by
            # wan_input_latent_diff.py, including its 1e-8 denominator floor.
            mean_l1, relative_l1 = latent_metrics(
                entry["input_latent"].float().numpy(), latent.float().numpy()
            )
            below_threshold = relative_l1 < self.args.reuse_threshold
            reuse = self.mode == "prompt_en_reuse" and below_threshold

        calls = {"positive": 0, "negative": 0}

        def predict_branch(**branch_kwargs):
            negative = kwargs["batch"].is_cfg_negative
            calls["negative" if negative else "positive"] += 1
            if negative and reuse:
                self.reused += 1
                # Preserve the reference output dtype. The normal pipeline still
                # slices this raw prediction and applies CFG/postprocessing.
                return entry["negative_prediction"].to(
                    device=latent_model_input.device, copy=True
                )
            prediction = self.predict_branch(**branch_kwargs)
            if not isinstance(prediction, torch.Tensor):
                raise RuntimeError("Expected a single Wan prediction tensor")
            if negative:
                self.negative_forwards += 1
                if self.mode == "prompt_a":
                    entry["negative_prediction"] = prediction.detach().to(
                        device="cpu", copy=True
                    )
            else:
                self.positive_forwards += 1
            return prediction

        with patch.object(self.stage, "_predict_noise", predict_branch):
            prediction = self.predict_cfg(
                latent_model_input=latent_model_input,
                timestep_index=timestep_index, **kwargs,
            )
        if calls != {"positive": 1, "negative": 1}:
            raise RuntimeError(f"Expected one invocation per branch, got {calls}")
        if self.mode == "prompt_a":
            self.reference.append(entry)
        self.rows.append({
            "run": self.mode, "step": timestep_index + 1,
            "timestep": timestep.reshape(-1)[0].item(),
            "mean_l1": mean_l1, "relative_mean_l1": relative_l1,
            "threshold": self.args.reuse_threshold,
            "below_threshold": int(below_threshold), "negative_reused": int(reuse),
        })
        return prediction


def run_prompt(worker, stage, row, mode, args, pair_dir, reference):
    import torch

    from sglang.multimodal_gen.configs.sample.sampling_params import (
        SamplingParams,
        generate_request_id,
    )
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    params = SamplingParams.from_user_sampling_params_args(
        args.model_path, server_args=worker.server_args,
        request_id=generate_request_id(),
        prompt=row["prompt_a"] if mode == "prompt_a" else row["prompt_en"],
        seed=args.seed, height=args.height, width=args.width,
        num_frames=args.num_frames, fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, progressive_mode="fullres",
        enable_cache_dit=False, enable_teacache=False, enable_spectrum=False,
        # Explicitly prevent environment-enabled temporal CFG gating from
        # interfering with this same-timestep, cross-prompt experiment.
        cfg_gate_step=1.0,
        save_output=True, return_file_paths_only=True,
        output_path=str(pair_dir), output_file_name=f"{mode}.mp4",
        perf_dump_path=str(pair_dir / f"{mode}_perf.json"),
    )
    request = prepare_request(worker.server_args, params)
    hook = NegativeReuseHook(stage, args, mode, reference)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with patch.object(stage, "_predict_noise_with_cfg", hook):
        # execute_forward materializes the decoded video on disk; calling only
        # pipeline.forward (as in the latent-only experiment) does not do that.
        result = worker.execute_forward([request])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if result.error:
        raise RuntimeError(f"{row['prompt_id']} / {mode}: {result.error}")
    if len(hook.rows) != args.num_inference_steps:
        raise RuntimeError(f"{mode}: captured {len(hook.rows)} steps, expected {args.num_inference_steps}")
    video_path = Path(request.output_file_path())
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty video: {video_path}")
    common = {"prompt_id": row["prompt_id"], "seed": args.seed}
    summary = {
        **common, "run": mode, "steps": len(hook.rows),
        "positive_forwards": hook.positive_forwards,
        "negative_forwards": hook.negative_forwards,
        "negative_reused": hook.reused,
        "reuse_fraction": hook.reused / len(hook.rows),
        "elapsed_seconds": elapsed, "video_path": str(video_path),
    }
    print(
        f"{row['prompt_id']} / {mode}: reused {hook.reused}/{len(hook.rows)} "
        f"negative branches; {elapsed:.2f}s; {video_path}", flush=True,
    )
    return [{**common, **step} for step in hook.rows], summary


def main(args):
    from sglang.multimodal_gen.runtime.distributed import cleanup_dist_env_and_memory
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Avoid mixing videos/CSVs from different thresholds or interrupted runs.
    config_path = args.output_dir / "experiment.json"
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    with config_path.open("x", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    server_args = ServerArgs.from_kwargs(
        model_path=args.model_path, backend="sglang", num_gpus=1, tp_size=1,
        sp_degree=1, ulysses_degree=1, ring_degree=1,
        encoder_parallel="replicate", enable_cfg_parallel=False,
        enable_torch_compile=False, enable_breakable_cuda_graph=False,
        warmup_mode="off", attention_backend="fa", flow_shift=args.flow_shift,
        wan_timestep_schedule=0,
    )
    set_global_server_args(server_args)
    try:
        worker = GPUWorker(local_rank=0, rank=0,
                           master_port=server_args.master_port, server_args=server_args)
        route = worker.pipeline.get_stage("denoising_stage")
        stage = getattr(route, "standard_stage", route)
        # Keep decoding_stage installed for all three runs.
        with (
            (args.output_dir / "step_metrics.csv").open("w", encoding="utf-8", newline="") as steps_file,
            (args.output_dir / "run_summary.csv").open("w", encoding="utf-8", newline="") as runs_file,
        ):
            steps_writer = csv.DictWriter(steps_file, fieldnames=STEP_FIELDS)
            runs_writer = csv.DictWriter(runs_file, fieldnames=RUN_FIELDS)
            steps_writer.writeheader()
            runs_writer.writeheader()
            for index, row in enumerate(args.prompts, start=1):
                print(f"[{index}/{len(args.prompts)}] {row['prompt_id']}", flush=True)
                pair_dir = args.output_dir / row["prompt_id"]
                pair_dir.mkdir(parents=True, exist_ok=True)
                reference = []
                for mode in ("prompt_a", "prompt_en_full", "prompt_en_reuse"):
                    step_rows, summary = run_prompt(
                        worker, stage, row, mode, args, pair_dir, reference
                    )
                    steps_writer.writerows(step_rows)
                    runs_writer.writerow(summary)
                    steps_file.flush()
                    runs_file.flush()
                # Release the pair's CPU cache before moving to the next pair.
                reference.clear()
        print(f"Results: {args.output_dir}", flush=True)
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main(parse_args())
