"""Shadow-test n_en(t) ~= n_a(t) + [n_en(k) - n_a(k)] on a fresh trajectory.

All model settings match wan_negative_cfg_diagnostics.py; default pair is
p29_color, seed 42. Run prompt_a, an English baseline, and an English shadow
run. All three decode videos. Every shadow step computes both real branches
and returns the real CFG output to the scheduler: NO actual speedup or corrected
rollout is performed. Cached tensors remain in memory only.

Evaluate the Cartesian product of --warmup-steps (default 1 4 10) and
--refresh-intervals (default 2 3 5). Steps are one-based. Warmup W computes and
refreshes through step W; subsequent refreshes are W+I, W+2I, ... . Other steps
use the correction from the most recent scheduled refresh. The ground truth
at a hypothetical skipped step is used for scoring, NEVER to update that policy.

Store corrections in float32 and cast reconstructed negatives back to the real
branch dtype before the original CFG combination/postprocessing. Record both
the ideal residual-drift error and the practical, rounded prediction errors.
Summaries exclude refresh/warmup steps so zero errors cannot inflate accuracy.
relative_mean_l1 uses symmetric magnitude normalization; relative_to_reference
uses the fresh English prediction magnitude (both floored at 1e-8).
"""

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wan_input_latent_diff import load_pairs
from wan_negative_cfg_reuse import NegativeReuseHook
from wan_negative_cfg_diagnostics import cpu, generate, metrics, write_csv


@dataclass
class ResidualPolicy:
    warmup: int
    interval: int
    anchor_step: int = 0
    residual: object = None

    @property
    def name(self):
        return f"warmup_{self.warmup:02d}_interval_{self.interval:02d}"

    def refresh_at(self, step):
        return step <= self.warmup or (step - self.warmup) % self.interval == 0

    def refresh(self, step, residual):
        if not self.refresh_at(step):
            raise RuntimeError("Attempt to refresh a policy on a reuse step")
        self.anchor_step = step
        self.residual = residual.detach().clone()


class ResidualShadowHook:
    def __init__(self, stage, args, reference):
        self.stage, self.args, self.reference = stage, args, reference
        self.predict_cfg = stage._predict_noise_with_cfg
        self.predict_branch = stage._predict_noise
        self.policies = [ResidualPolicy(w, i) for w in args.warmup_steps
                         for i in args.refresh_intervals]
        self.rows = []
        self.step_count = 0
        self.positive_forwards = self.negative_forwards = self.reused = 0

    def __call__(self, *, latent_model_input, timestep_index, **kwargs):
        import torch

        if timestep_index != self.step_count or timestep_index >= self.args.num_inference_steps:
            raise RuntimeError(f"Unexpected step index {timestep_index}")
        policy, batch, server = kwargs["cfg_policy"], kwargs["batch"], kwargs["server_args"]
        if (server.enable_cfg_parallel or len(policy.branches) != 2
                or not policy.branches[0].is_conditional or policy.branches[1].is_conditional):
            raise RuntimeError("Residual diagnostics require serial positive/negative CFG")
        if kwargs.get("cfg_gate_state") and kwargs["cfg_gate_state"]["active"]:
            raise RuntimeError("Other CFG gating must be disabled")
        entry = self.reference[timestep_index]
        timestep, latent = cpu(kwargs["timestep"]), cpu(latent_model_input)
        if not torch.equal(timestep, entry["timestep"]):
            raise RuntimeError("Reference and English timesteps differ")
        if timestep_index == 0 and not torch.equal(latent, entry["input_latent"]):
            raise RuntimeError("Initial inputs differ")
        raw = {}

        def observe_branch(**branch_kwargs):
            negative = batch.is_cfg_negative
            if negative in raw:
                raise RuntimeError("A CFG branch ran more than once")
            prediction = self.predict_branch(**branch_kwargs)
            if not isinstance(prediction, torch.Tensor):
                raise RuntimeError("Expected a single Wan prediction tensor")
            raw[negative] = prediction.detach().clone()
            if negative:
                self.negative_forwards += 1
            else:
                self.positive_forwards += 1
            return prediction

        with patch.object(self.stage, "_predict_noise", observe_branch):
            guided = self.predict_cfg(latent_model_input=latent_model_input,
                                      timestep_index=timestep_index, **kwargs)
        if set(raw) != {False, True}:
            raise RuntimeError("Expected both real CFG branches")
        config = server.pipeline_config
        positive = config.slice_noise_pred(raw[False], kwargs["latents"])
        negative = config.slice_noise_pred(raw[True], kwargs["latents"])
        cached_raw = entry["negative_prediction"].to(device=negative.device, copy=True)
        if cached_raw.shape != raw[True].shape or cached_raw.dtype != raw[True].dtype:
            raise RuntimeError("Cached/fresh prediction shapes or dtypes differ")
        cached = config.slice_noise_pred(cached_raw, kwargs["latents"])
        scale = config.get_classifier_free_guidance_scale(batch, kwargs["current_guidance_scale"])
        negative_cpu, cached_cpu, guided_cpu = cpu(negative), cpu(cached), cpu(guided)
        residual = negative_cpu.float() - cached_cpu.float()
        raw_guided = policy.combine([positive.clone(), cached.clone()], batch, scale, config)
        common = {
            "step": timestep_index + 1, "timestep": timestep.reshape(-1)[0].item(),
            "guidance_scale": float(scale),
            **metrics(entry["input_latent"], latent, "input_vs_a"),
            **metrics(negative_cpu, cached_cpu, "raw_negative"),
            **metrics(guided_cpu, cpu(raw_guided), "raw_guided"),
        }

        for schedule in self.policies:
            step = timestep_index + 1
            refresh = schedule.refresh_at(step)
            row = {"policy": schedule.name, "warmup_steps": schedule.warmup,
                   "refresh_interval": schedule.interval, **common,
                   "action": "refresh" if refresh else "reuse"}
            if refresh:
                schedule.refresh(step, residual)
            else:
                if schedule.residual is None or schedule.anchor_step >= step:
                    raise RuntimeError("Missing or non-causal residual anchor")
                # The CURRENT residual is ground truth for scoring only. The
                # candidate receives only A(t) and the last REFRESH residual.
                ideal_corrected = cached_cpu.float() + schedule.residual
                corrected = ideal_corrected.to(device=negative.device, dtype=negative.dtype)
                corrected_guided = policy.combine(
                    [positive.clone(), corrected.clone()], batch, scale, config
                )
                row.update(metrics(negative_cpu, cpu(corrected), "corrected_negative"))
                row.update(metrics(guided_cpu, cpu(corrected_guided), "corrected_guided"))
                drift = metrics(residual, schedule.residual, "residual_drift")
                row["residual_drift_mean_l1"] = drift["residual_drift_mean_l1"]
                row["residual_drift_relative_to_current"] = drift["residual_drift_relative_to_reference"]
                row["ideal_correction_helps"] = int(
                    drift["residual_drift_mean_l1"] < common["raw_negative_mean_l1"]
                )
                for kind in ("negative", "guided"):
                    raw_error = row[f"raw_{kind}_mean_l1"]
                    corrected_error = row[f"corrected_{kind}_mean_l1"]
                    row[f"{kind}_improved"] = int(corrected_error < raw_error)
                    row[f"{kind}_worsened"] = int(corrected_error > raw_error)
                    row[f"{kind}_error_ratio"] = corrected_error / raw_error if raw_error else None
            row["anchor_step"] = schedule.anchor_step
            row["anchor_age_steps"] = step - schedule.anchor_step
            self.rows.append(row)
        self.step_count += 1
        return guided  # The real trajectory NEVER consumes any approximation.


def summarize(rows):
    summaries = []
    for name in dict.fromkeys(row["policy"] for row in rows):
        policy_rows = [row for row in rows if row["policy"] == name]
        skipped = [row for row in policy_rows if row["action"] == "reuse"]
        summary = {
            "policy": name, "warmup_steps": policy_rows[0]["warmup_steps"],
            "refresh_interval": policy_rows[0]["refresh_interval"],
            "steps": len(policy_rows), "hypothetical_negative_forwards": len(policy_rows) - len(skipped),
            "hypothetical_negative_skips": len(skipped),
            "hypothetical_branch_evaluations_saved_fraction": len(skipped) / (2 * len(policy_rows)),
        }
        for kind in ("negative", "guided"):
            raw_errors = [row[f"raw_{kind}_mean_l1"] for row in skipped]
            corrected_errors = [row[f"corrected_{kind}_mean_l1"] for row in skipped]
            raw_sum, corrected_sum = sum(raw_errors), sum(corrected_errors)
            summary[f"{kind}_improved_steps"] = sum(row[f"{kind}_improved"] for row in skipped)
            summary[f"{kind}_worsened_steps"] = sum(row[f"{kind}_worsened"] for row in skipped)
            summary[f"{kind}_unchanged_steps"] = len(skipped) - summary[f"{kind}_improved_steps"] - summary[f"{kind}_worsened_steps"]
            summary[f"{kind}_error_ratio_of_sums"] = corrected_sum / raw_sum if raw_sum else None
            for variant, errors in (("raw", raw_errors), ("corrected", corrected_errors)):
                summary[f"{variant}_{kind}_mean_l1_on_skips"] = float(np.mean(errors)) if errors else None
                relative = [row[f"{variant}_{kind}_relative_to_reference"] for row in skipped]
                summary[f"{variant}_{kind}_mean_relative_to_reference_on_skips"] = float(np.mean(relative)) if relative else None
                summary[f"{variant}_{kind}_max_relative_to_reference_on_skips"] = max(relative) if relative else None
        summaries.append(summary)
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "week12/experiment4/p10_multiple_objects.json")
    parser.add_argument("--output-dir", type=Path, default=Path("wan_negative_cfg_residual_shadow"))
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
    parser.add_argument("--warmup-steps", nargs="+", type=int, default=[1, 4, 10])
    parser.add_argument("--refresh-intervals", nargs="+", type=int, default=[2, 3, 5])
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
    if any(w < 1 or w > args.num_inference_steps for w in args.warmup_steps):
        parser.error("Warmup lengths must be between 1 and num-inference-steps")
    if any(i < 1 for i in args.refresh_intervals):
        parser.error("Refresh intervals must be positive (1 is a no-skip control)")
    args.warmup_steps = sorted(set(args.warmup_steps))
    args.refresh_intervals = sorted(set(args.refresh_intervals))
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
        reference, summaries = [], []

        def run(label, hook=None, baseline=None, keep=False):
            observer, trajectory, times, summary = generate(
                worker, stage, row, label, args, pair_dir, hook, baseline, keep
            )
            summaries.append({"prompt_id": row["prompt_id"], "seed": args.seed, **summary})
            write_csv(args.output_dir / "run_summary.csv", summaries)
            return observer, trajectory, times, summary

        source_args = copy.copy(args)
        source_args.reuse_threshold = 0.0  # Source-only hook never attempts reuse.
        run("prompt_a", NegativeReuseHook(stage, source_args, "prompt_a", reference))
        observer, trajectory, times, _ = run("prompt_en_full", keep=True)
        baseline = (observer.trace, trajectory, times)
        shadow = ResidualShadowHook(stage, args, reference)
        observer, _, _, control = run("prompt_en_residual_shadow", shadow, baseline)
        if (shadow.step_count != args.num_inference_steps or shadow.reused != 0
                or shadow.negative_forwards != args.num_inference_steps
                or shadow.positive_forwards != args.num_inference_steps):
            raise RuntimeError("Shadow run did not execute every real branch")
        common = {"prompt_id": row["prompt_id"], "seed": args.seed}
        write_csv(args.output_dir / "residual_step_metrics.csv", [{**common, **r} for r in shadow.rows])
        policy_summaries = summarize(shadow.rows)
        write_csv(args.output_dir / "policy_summary.csv", [{**common, **r} for r in policy_summaries])
        write_csv(args.output_dir / "trajectory_metrics.csv", [{**common, **r} for r in observer.rows])
        with (args.output_dir / "control_checks.json").open("w", encoding="utf-8") as file:
            json.dump({"shadow_vs_full": control,
                       "note": "All real branches ran. Inspect any nonzero control differences before interpreting errors. "
                               "Policy summaries score hypothetical skipped steps only. Improved local errors do not guarantee rollout quality or speedup."},
                      file, indent=2)
        print(f"CONTROL all_steps_exact={control['all_steps_exact_vs_full']}; "
              f"final_relative_l1={control['final_latent_vs_full_relative_mean_l1']:.8g}", flush=True)
        for summary in policy_summaries:
            print(f"{summary['policy']}: hypothetical skips={summary['hypothetical_negative_skips']}, "
                  f"guided improved/worsened={summary['guided_improved_steps']}/{summary['guided_worsened_steps']}, "
                  f"guided error ratio={summary['guided_error_ratio_of_sums']}", flush=True)
        print(f"Results: {args.output_dir}", flush=True)
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main(parse_args())
