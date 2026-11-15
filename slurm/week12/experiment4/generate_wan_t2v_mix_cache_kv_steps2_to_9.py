"""Start Wan B from A’s step-1 latent and reuse K/V for the first 10 B steps."""

import argparse
import csv
import json
import time
import uuid
from pathlib import Path

import numpy as np

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)


SOURCE_EVALUATIONS = 50
FLOW_SHIFT = 8.0

TARGET_STEPS = tuple(range(1, 11))
TARGET_INDICES = tuple(step - 1 for step in TARGET_STEPS)
START_TIMESTEP = 997

EXPECTED_TIMESTEPS = [
    997, 995, 991, 986, 981, 975, 969, 963, 956, 949,
    923, 888, 842, 774, 666, 470,
]

_original_forward = TimestepPreparationStage.forward
_original_latent_forward = LatentPreparationStage.forward
_original_record_trajectory = DenoisingStage._record_trajectory
# A and B execute serially on the same worker, as required for K/V reuse.
_start_latents = {}


def capture_start_latent(self, ctx, step, batch, server_args):
    kv = getattr(batch, "wan_chai_kv", None)
    if not batch.is_warmup and kv and kv["mode"] == "record" and step.step_index == 0:
        if ctx.scheduler.timesteps[1].item() != START_TIMESTEP:
            raise RuntimeError("A step 1 must produce the latent for timestep 997")
        _start_latents.clear()
        _start_latents[kv["cache_id"]] = ctx.latents.detach().clone()
    return _original_record_trajectory(self, ctx, step, batch, server_args)


def prepare_start_latent(self, batch, server_args):
    kv = getattr(batch, "wan_chai_kv", None)
    if not batch.is_warmup and kv and kv["mode"] == "reuse":
        batch.latents = _start_latents.pop(kv["cache_id"], None)
        if batch.latents is None:
            raise RuntimeError("Missing A step-1 latent for B")
    return _original_latent_forward(self, batch, server_args)


def selected_scheduler_sigmas(scheduler):
    # Keep A step 2's sigma for the handoff, then use 30-step grid
    # evaluations 2–9 and every fifth 50-step evaluation from 16 through 46.
    grid_30 = np.linspace(scheduler.sigma_max, scheduler.sigma_min, 31)
    grid_50 = np.linspace(scheduler.sigma_max, scheduler.sigma_min, SOURCE_EVALUATIONS + 1)
    return np.concatenate((grid_50[1:2], grid_30[1:9], grid_50[15:-1:5]))


def prepare_experiment_timesteps(self, batch, server_args):
    kv = getattr(batch, "wan_chai_kv", None)
    if batch.is_warmup or not kv or kv["mode"] != "reuse":
        return _original_forward(self, batch, server_args)

    batch.sigmas = selected_scheduler_sigmas(self.scheduler)
    prepared = _original_forward(self, batch, server_args)
    if prepared.timesteps.tolist() != EXPECTED_TIMESTEPS:
        raise RuntimeError(f"Unexpected B timesteps: {prepared.timesteps.tolist()}")
    return prepared


# Spawned GPU workers import this module too, so they receive the same override.
DenoisingStage._record_trajectory = capture_start_latent
LatentPreparationStage.forward = prepare_start_latent
TimestepPreparationStage.forward = prepare_experiment_timesteps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--prompt-a", default="A party on a sandy beach beside gentle waves, wide shot.")
    parser.add_argument("--prompt-b", default="Gentle waves roll onto an empty sandy beach, wide shot.")
    parser.add_argument("--negative-prompt", default=None,
                        help="Shared by all runs; omit to use the model default")
    parser.add_argument("--steps-a", type=int, required=True,
                        help="Denoising steps for source request A (50)")
    parser.add_argument("--steps-b", type=int, required=True,
                        help="Denoising steps for target request B (16)")
    parser.add_argument("--flow-shift", type=float, required=True,
                        help="Flow shift used by both A and B")
    parser.add_argument("--wan-timestep-schedule", type=int, choices=(0, 1, 2, 3), default=0,
                        help="Wan timestep schedule used by both A and B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps_a != SOURCE_EVALUATIONS or args.steps_b != len(EXPECTED_TIMESTEPS):
        parser.error("This experiment requires --steps-a 50 and --steps-b 16")
    if args.flow_shift != FLOW_SHIFT:
        parser.error("--flow-shift must be 8 to reproduce the requested timesteps")
    if args.wan_timestep_schedule != 0:
        parser.error("The experiment requires the default Wan scheduler mode")
    if args.guidance_scale <= 1:
        parser.error("--guidance-scale must exceed 1 to exercise both CFG branches")
    return args


def build_cache_mapping(config):
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler,
    )

    scheduler = FlowUniPCMultistepScheduler(
        shift=config.flow_shift, timestep_schedule=config.wan_timestep_schedule,
    )
    scheduler.set_timesteps(SOURCE_EVALUATIONS, device="cpu")
    sigmas_a = scheduler.sigmas[:-1].tolist()
    timesteps_a = scheduler.timesteps.tolist()

    scheduler.set_timesteps(sigmas=selected_scheduler_sigmas(scheduler), device="cpu")
    sigmas_b = scheduler.sigmas.tolist()
    timesteps_b = scheduler.timesteps.tolist()
    if timesteps_b != EXPECTED_TIMESTEPS or sigmas_b[-1] != 0.0:
        raise RuntimeError("The scheduler did not reproduce the requested B schedule")

    if timesteps_b[0] != START_TIMESTEP or sigmas_b[0] != sigmas_a[1]:
        raise RuntimeError("B must start at the sigma reached after A step 1")

    mapping = []
    for target_index in TARGET_INDICES:
        target_sigma = sigmas_b[target_index]
        source_index = min(
            range(len(sigmas_a)), key=lambda index: abs(sigmas_a[index] - target_sigma)
        )
        source_sigma = sigmas_a[source_index]
        mapping.append(dict(
            a_step=source_index + 1, b_step=target_index + 1,
            a_timestep=timesteps_a[source_index], timestep=timesteps_b[target_index],
            a_sigma=source_sigma, b_sigma=target_sigma,
            sigma_difference=abs(source_sigma - target_sigma),
        ))
    return mapping


def main(args):
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_id = uuid.uuid4().hex
    settings = dict(
        height=args.height, width=args.width, num_frames=args.num_frames,
        fps=args.fps, seed=args.seed, num_inference_steps=args.steps_b,
        guidance_scale=args.guidance_scale, progressive_mode="fullres",
        enable_cache_dit=False, enable_teacache=False, enable_spectrum=False,
        cfg_gate_step=1.0, save_output=True, output_path=str(output_dir),
    )
    if args.negative_prompt is not None:
        settings["negative_prompt"] = args.negative_prompt
    generator = DiffGenerator.from_pretrained(
        model_path=args.model_path, num_gpus=1, tp_size=1,
        sp_degree=1, ulysses_degree=1, ring_degree=1,
        encoder_parallel="replicate", enable_cfg_parallel=False,
        enable_torch_compile=False, enable_breakable_cuda_graph=False,
        warmup_mode="off", attention_backend="fa",
        flow_shift=args.flow_shift,
        wan_timestep_schedule=args.wan_timestep_schedule,
    )
    rows = []
    try:
        from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

        defaults = SamplingParams.from_user_sampling_params_args(
            generator.server_args.model_path, server_args=generator.server_args,
            prompt=args.prompt_b, **settings,
        )
        settings["negative_prompt"] = defaults.negative_prompt
        config = generator.server_args.pipeline_config
        cache_mapping = build_cache_mapping(config)
        source_steps = [entry["a_step"] - 1 for entry in cache_mapping]
        print(f"K/V mapping (A -> B): {cache_mapping}", flush=True)
        (output_dir / "experiment.json").write_text(json.dumps(dict(
            model_path=args.model_path, prompt_a=args.prompt_a, prompt_b=args.prompt_b,
            settings=settings, cache_id=cache_id, cache_steps=TARGET_STEPS,
            cache_mapping=cache_mapping, cache_mapping_policy="nearest_sigma",
            b_timesteps=EXPECTED_TIMESTEPS, final_sigma=0.0,
            b_initial_latent=dict(a_step=1, timestep=START_TIMESTEP),
            b_solver_history="fresh",
            steps_by_run={"a_full": args.steps_a,
                          "b_chai_kv": args.steps_b},
            block_index=0, cfg_branches=["positive", "negative"],
            cache_format="projected K after QK norm/RoPE; projected V",
            flow_shift=config.flow_shift,
            timestep_schedule=config.wan_timestep_schedule,
        ), indent=2), encoding="utf-8")

        for label, prompt, mode, num_steps in (
            ("a_full", args.prompt_a, "record", args.steps_a),
            ("b_chai_kv", args.prompt_b, "reuse", args.steps_b),
        ):
            perf_path = output_dir / f"{label}_perf.json"
            report_path = output_dir / f"{label}_kv.json"
            # A new cache ID also prevents accepting reports from an older run.
            started = time.perf_counter()
            result = generator.generate(dict(
                settings, prompt=prompt, num_inference_steps=num_steps,
                output_file_name=f"{label}.mp4",
                perf_dump_path=str(perf_path),
                wan_chai_kv=dict(mode=mode, cache_id=cache_id,
                                 reuse_strategy="replace",
                                 target_steps=list(TARGET_INDICES),
                                 source_steps=source_steps,
                                 report_path=str(report_path)),
            ))
            elapsed = time.perf_counter() - started
            if result is None:
                raise RuntimeError(f"Generation failed: {label}")
            perf = json.loads(perf_path.read_text())
            report = json.loads(report_path.read_text())
            if len(perf["denoise_steps_ms"]) != num_steps:
                raise RuntimeError(f"{label} did not execute {num_steps} denoising steps")
            expected = {
                "record": (2 * len(set(source_steps)), 0),
                "reuse": (0, 2 * len(TARGET_INDICES)),
            }[mode]
            if (report["cache_id"] != cache_id or report["mode"] != mode
                    or report["source_steps"] != [i + 1 for i in source_steps]
                    or report["target_steps"] != list(TARGET_STEPS)
                    or report["num_steps"] != num_steps
                    or report["block0_calls"] != 2 * num_steps
                    or (report["captures"], report["replacements"]) != expected):
                raise RuntimeError(f"Unexpected K/V audit report: {report}")
            if mode == "reuse":
                expected_events = {
                    (target + 1, source + 1, branch)
                    for target, source in zip(TARGET_INDICES, source_steps)
                    for branch in ("positive", "negative")
                }
                actual_events = {
                    (event["step"], event["source_step"], event["branch"])
                    for event in report["events"]
                    if event["action"] == "reuse" and event["block_index"] == 0
                    and event["reuse_strategy"] == "replace"
                }
                if actual_events != expected_events:
                    raise RuntimeError(f"Unexpected K/V replacement events: {report}")
            stages = {entry["name"]: entry["duration_ms"] for entry in perf["steps"]}
            rows.append(dict(
                label=label, actual_steps=num_steps, wall_seconds=elapsed,
                denoising_ms=stages["DenoisingStage"],
                decoding_ms=stages["DecodingStage"],
                captures=report["captures"], replacements=report["replacements"],
                cache_bytes=report["cache_bytes"],
            ))
            with (output_dir / "summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"{label}: {num_steps} steps, {report['captures']} captures, "
                  f"{report['replacements']} replacements", flush=True)
            del result
    finally:
        generator.shutdown()
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main(parse_args())
