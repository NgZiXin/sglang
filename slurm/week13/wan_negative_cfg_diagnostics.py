"""Diagnose cross-prompt negative CFG reuse without changing generation defaults.

Default pair: p29_color from week12/experiment4/p10_multiple_objects.json.
Runs prompt_a, an ordinary English baseline, the existing reuse hook at threshold
zero, and a shadow run that computes every branch and only MEASURES substitution
error. Optional --single-step-reuse 2 3 4 adds independent one-based substitutions.
Every run decodes an MP4. No cached tensors are persisted to disk.

Metrics use float32 CPU tensors. relative_mean_l1 has the same symmetric
denominator as wan_input_latent_diff.py; relative_to_reference instead divides
by mean(abs(reference)). Both denominators are floored at 1e-8. The shadow CSV
compares fresh English predictions (reference) with cached-A substitutions.
Trajectory CSV compares each English variant with the ordinary English baseline.
Timings include diagnostic overhead and are not acceleration benchmarks.
"""

import argparse
import copy
import csv
import json
import math
import re
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wan_input_latent_diff import latent_metrics, load_pairs
from wan_negative_cfg_reuse import NegativeReuseHook


def cpu(tensor):
    return tensor.detach().to(device="cpu", copy=True)


def metrics(reference, candidate, prefix):
    a, b = reference.float().numpy(), candidate.float().numpy()
    mean_l1, relative = latent_metrics(a, b)
    reference_scale = float(np.abs(a).mean())
    return {
        f"{prefix}_mean_l1": mean_l1,
        f"{prefix}_relative_mean_l1": relative,
        f"{prefix}_relative_to_reference": mean_l1 / max(reference_scale, 1e-8),
        f"{prefix}_reference_mean_abs": reference_scale,
        f"{prefix}_candidate_mean_abs": float(np.abs(b).mean()),
        f"{prefix}_max_abs": float(np.abs(a - b).max()),
        f"{prefix}_exact": int(np.array_equal(a, b)),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TraceHook:
    """Observe the returned prediction without replacing or recombining it."""

    def __init__(self, predict, steps, baseline=None, keep=False):
        self.predict, self.steps = predict, steps
        self.baseline, self.keep = baseline, keep
        self.trace, self.rows = [], []

    def __call__(self, *, latent_model_input, timestep_index, **kwargs):
        import torch

        if timestep_index != len(self.rows) or timestep_index >= self.steps:
            raise RuntimeError(f"Unexpected step index {timestep_index}")
        entry = {"input": cpu(latent_model_input), "timestep": cpu(kwargs["timestep"])}
        row = {"step": timestep_index + 1,
               "timestep": entry["timestep"].reshape(-1)[0].item()}
        if self.baseline is not None:
            base = self.baseline[timestep_index]
            if not torch.equal(entry["timestep"], base["timestep"]):
                raise RuntimeError("English schedules differ")
            if timestep_index == 0 and not torch.equal(entry["input"], base["input"]):
                raise RuntimeError("English initial inputs differ")
            row.update(metrics(base["input"], entry["input"], "input_vs_full"))
        prediction = self.predict(latent_model_input=latent_model_input,
                                  timestep_index=timestep_index, **kwargs)
        if not isinstance(prediction, torch.Tensor):
            raise RuntimeError("Expected one Wan guided prediction tensor")
        entry["guided"] = cpu(prediction)
        if self.baseline is not None:
            row.update(metrics(base["guided"], entry["guided"], "guided_vs_full"))
        if self.keep:
            self.trace.append(entry)
        self.rows.append(row)
        return prediction


class SingleStepHook(NegativeReuseHook):
    """Use the existing substitution path at exactly one one-based step."""

    def __init__(self, stage, args, reference, step):
        super().__init__(stage, copy.copy(args), "prompt_en_reuse", reference)
        self.step = step

    def __call__(self, *, timestep_index, **kwargs):
        # A finite metric always passes infinity, and never passes zero.
        self.args.reuse_threshold = math.inf if timestep_index + 1 == self.step else 0.0
        return super().__call__(timestep_index=timestep_index, **kwargs)


class ShadowHook:
    """Compute both branches, measure an alternative CFG output, return the real one."""

    def __init__(self, stage, args, reference):
        self.stage, self.args, self.reference = stage, args, reference
        self.predict_cfg = stage._predict_noise_with_cfg
        self.predict_branch = stage._predict_noise
        self.rows = []
        self.positive_forwards = self.negative_forwards = self.reused = 0

    def __call__(self, *, latent_model_input, timestep_index, **kwargs):
        import torch

        if timestep_index != len(self.rows) or timestep_index >= self.args.num_inference_steps:
            raise RuntimeError(f"Unexpected shadow step {timestep_index}")
        policy, batch = kwargs["cfg_policy"], kwargs["batch"]
        server = kwargs["server_args"]
        if (server.enable_cfg_parallel or len(policy.branches) != 2
                or not policy.branches[0].is_conditional or policy.branches[1].is_conditional):
            raise RuntimeError("Shadow diagnostics require serial two-branch CFG")
        if kwargs.get("cfg_gate_state") and kwargs["cfg_gate_state"]["active"]:
            raise RuntimeError("Other CFG gating must be disabled")
        reference = self.reference[timestep_index]
        timestep, latent = cpu(kwargs["timestep"]), cpu(latent_model_input)
        if not torch.equal(timestep, reference["timestep"]):
            raise RuntimeError("Reference and shadow schedules differ")
        if timestep_index == 0 and not torch.equal(latent, reference["input_latent"]):
            raise RuntimeError("Reference and shadow initial inputs differ")
        raw = {}

        def observe_branch(**branch_kwargs):
            negative = batch.is_cfg_negative
            if negative in raw:
                raise RuntimeError("Branch executed more than once in one step")
            output = self.predict_branch(**branch_kwargs)
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("Expected one Wan branch prediction tensor")
            raw[negative] = output.detach().clone()
            if negative:
                self.negative_forwards += 1
            else:
                self.positive_forwards += 1
            return output

        with patch.object(self.stage, "_predict_noise", observe_branch):
            guided = self.predict_cfg(latent_model_input=latent_model_input,
                                      timestep_index=timestep_index, **kwargs)
        if set(raw) != {False, True}:
            raise RuntimeError("Did not observe both CFG branches")
        config = server.pipeline_config
        positive = config.slice_noise_pred(raw[False], kwargs["latents"])
        negative = config.slice_noise_pred(raw[True], kwargs["latents"])
        cached_raw = reference["negative_prediction"].to(device=negative.device, copy=True)
        if cached_raw.dtype != raw[True].dtype or cached_raw.shape != raw[True].shape:
            raise RuntimeError("Cached and fresh predictions have different dtype/shape")
        cached = config.slice_noise_pred(cached_raw, kwargs["latents"])
        scale = config.get_classifier_free_guidance_scale(batch, kwargs["current_guidance_scale"])
        # Use the real policy (including its dtype arithmetic and postprocessing).
        # Clones keep hypothetical computation from changing real branch tensors.
        hypothetical = policy.combine([positive.clone(), cached.clone()], batch, scale, config)
        row = {"step": timestep_index + 1, "timestep": timestep.reshape(-1)[0].item(),
               "guidance_scale": float(scale), "threshold": self.args.reuse_threshold}
        row.update(metrics(reference["input_latent"], latent, "input_vs_a"))
        row["would_reuse"] = int(row["input_vs_a_relative_mean_l1"] < self.args.reuse_threshold)
        row.update(metrics(cpu(negative), cpu(cached), "negative"))
        row.update(metrics(cpu(guided), cpu(hypothetical), "guided"))
        # This algebraic pre-postprocessing error avoids low-precision CFG
        # rounding and gives the expected (1-s) amplification independently.
        row["ideal_guided_error_mean_l1"] = abs(1.0 - float(scale)) * row["negative_mean_l1"]
        self.rows.append(row)
        return guided  # Never send the hypothetical prediction to the scheduler.


def generate(worker, stage, row, label, args, pair_dir, hook=None, baseline=None, keep=False):
    import torch

    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams, generate_request_id
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    params = SamplingParams.from_user_sampling_params_args(
        args.model_path, server_args=worker.server_args, request_id=generate_request_id(),
        prompt=row["prompt_a"] if label == "prompt_a" else row["prompt_en"],
        seed=args.seed, height=args.height, width=args.width,
        num_frames=args.num_frames, fps=args.fps, num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, progressive_mode="fullres",
        enable_cache_dit=False, enable_teacache=False, enable_spectrum=False, cfg_gate_step=1.0,
        return_trajectory_latents=True, save_output=True, return_file_paths_only=True,
        output_path=str(pair_dir), output_file_name=f"{label}.mp4",
        perf_dump_path=str(pair_dir / f"{label}_perf.json"),
    )
    request = prepare_request(worker.server_args, params)
    observer = TraceHook(hook if hook is not None else stage._predict_noise_with_cfg,
                         args.num_inference_steps,
                         baseline=baseline[0] if baseline is not None else None, keep=keep)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with patch.object(stage, "_predict_noise_with_cfg", observer):
        result = worker.execute_forward([request])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if result.error:
        raise RuntimeError(f"{label}: {result.error}")
    trajectory, times = result.trajectory_latents, result.trajectory_timesteps
    if (len(observer.rows) != args.num_inference_steps or trajectory is None
            or trajectory.shape[1] != args.num_inference_steps or times is None
            or times.shape != (args.num_inference_steps,)):
        raise RuntimeError(f"{label}: incomplete predictions or latent trajectory")
    video = Path(request.output_file_path())
    if not video.is_file() or not video.stat().st_size:
        raise RuntimeError(f"Missing video {video}")
    summary = {
        "run": label, "steps": len(observer.rows),
        "positive_forwards": hook.positive_forwards if hook is not None else len(observer.rows),
        "negative_forwards": hook.negative_forwards if hook is not None else len(observer.rows),
        "negative_reused": hook.reused if hook is not None else 0,
        "elapsed_seconds": elapsed, "video_path": str(video),
    }
    if baseline is not None:
        if not torch.equal(times, baseline[2]):
            raise RuntimeError("Post-update schedules differ")
        for index, step in enumerate(observer.rows):
            step.update(metrics(baseline[1][:, index], trajectory[:, index], "post_latent_vs_full"))
        summary.update(metrics(baseline[1][:, -1], trajectory[:, -1], "final_latent_vs_full"))
        summary["all_steps_exact_vs_full"] = int(all(
            step["input_vs_full_exact"] and step["guided_vs_full_exact"]
            and step["post_latent_vs_full_exact"] for step in observer.rows
        ))
    print(f"{label}: {summary['negative_forwards']} negative forwards, "
          f"{summary['negative_reused']} reused, {elapsed:.2f}s", flush=True)
    return observer, trajectory, times, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "week12/experiment4/p10_multiple_objects.json")
    parser.add_argument("--output-dir", type=Path, default=Path("wan_negative_cfg_diagnostics"))
    parser.add_argument("--prompt-id", default="p29_color")
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-threshold", type=float, default=0.05,
                        help="Shadow candidate labeling only; shadow always computes both branches")
    parser.add_argument("--single-step-reuse", nargs="+", type=int, default=[], metavar="STEP",
                        help="Add a separate video for each one-based substitution step, e.g. 2 3 4")
    args = parser.parse_args()
    try:
        args.prompts = load_pairs(args.prompt_manifest, args.prompt_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prompt_id):
            raise ValueError("Unsafe prompt_id")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid prompt selection: {error}")
    if min(args.height, args.width, args.num_frames, args.fps, args.num_inference_steps) < 1:
        parser.error("Dimensions, frames, fps and steps must be positive")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale <= 1:
        parser.error("Guidance scale must be finite and greater than 1")
    if not math.isfinite(args.reuse_threshold) or args.reuse_threshold < 0:
        parser.error("Threshold must be finite and nonnegative")
    if any(step < 1 or step > args.num_inference_steps for step in args.single_step_reuse):
        parser.error("Single-step indices must be within the inference step range")
    args.single_step_reuse = sorted(set(args.single_step_reuse))
    return args


def main(args):
    from sglang.multimodal_gen.runtime.distributed import cleanup_dist_env_and_memory
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "experiment.json").open("x", encoding="utf-8") as file:
        json.dump(vars(args), file, default=str, indent=2, ensure_ascii=False)
    server = ServerArgs.from_kwargs(
        model_path=args.model_path, backend="sglang", num_gpus=1, tp_size=1,
        sp_degree=1, ulysses_degree=1, ring_degree=1, encoder_parallel="replicate",
        enable_cfg_parallel=False, enable_torch_compile=False, enable_breakable_cuda_graph=False,
        warmup_mode="off", attention_backend="fa", flow_shift=args.flow_shift, wan_timestep_schedule=0,
    )
    set_global_server_args(server)
    try:
        worker = GPUWorker(local_rank=0, rank=0, master_port=server.master_port, server_args=server)
        route = worker.pipeline.get_stage("denoising_stage")
        stage = getattr(route, "standard_stage", route)
        row = args.prompts[0]
        pair_dir = args.output_dir / row["prompt_id"]
        pair_dir.mkdir(parents=True, exist_ok=True)
        reference, summaries, comparisons = [], [], []

        def run(label, hook=None, baseline=None, keep=False):
            observer, trajectory, times, summary = generate(
                worker, stage, row, label, args, pair_dir, hook, baseline, keep
            )
            common = {"prompt_id": row["prompt_id"], "seed": args.seed}
            summaries.append({**common, **summary})
            write_csv(args.output_dir / "run_summary.csv", summaries)
            if baseline is not None:
                comparisons.extend({**common, "run": label, **step} for step in observer.rows)
                write_csv(args.output_dir / "trajectory_metrics.csv", comparisons)
            return observer.trace, trajectory, times, summary

        run("prompt_a", NegativeReuseHook(stage, args, "prompt_a", reference))
        full = run("prompt_en_full", keep=True)
        zero_args = copy.copy(args)
        zero_args.reuse_threshold = 0.0
        zero = NegativeReuseHook(stage, zero_args, "prompt_en_reuse", reference)
        zero_summary = run("prompt_en_zero_reuse", zero, full)[3]
        if zero.reused != 0 or zero.negative_forwards != args.num_inference_steps:
            raise RuntimeError("Zero-threshold control unexpectedly skipped computation")
        shadow = ShadowHook(stage, args, reference)
        shadow_summary = run("prompt_en_shadow", shadow, full)[3]
        write_csv(args.output_dir / "shadow_metrics.csv", shadow.rows)
        controls = {
            "zero_reuse": zero_summary, "shadow": shadow_summary,
            "note": "Exact=0 is a diagnostic finding, not an automatic failure: inspect numerical errors. "
                    "Both controls must follow the normal English trajectory before interpreting substitutions.",
        }
        with (args.output_dir / "control_checks.json").open("w", encoding="utf-8") as file:
            json.dump(controls, file, indent=2)
        for label, summary in (("zero_reuse", zero_summary), ("shadow", shadow_summary)):
            print(f"CONTROL {label}: all_steps_exact={summary['all_steps_exact_vs_full']}, "
                  f"final_relative_l1={summary['final_latent_vs_full_relative_mean_l1']:.8g}", flush=True)
        for step in args.single_step_reuse:
            hook = SingleStepHook(stage, args, reference, step)
            run(f"prompt_en_reuse_step_{step:02d}", hook, full)
            if hook.reused != 1 or hook.negative_forwards != args.num_inference_steps - 1:
                raise RuntimeError("Single-step run did not make exactly one substitution")
        print(f"Diagnostics: {args.output_dir}", flush=True)
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main(parse_args())
