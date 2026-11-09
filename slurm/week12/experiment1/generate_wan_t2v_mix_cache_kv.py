"""Generate mixed-schedule Wan B with first-block K/V from 50-step A."""

import argparse
import csv
import json
import time
import uuid
from pathlib import Path

import numpy as np

from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)


TARGET_INDICES = (1, 2)
TARGET_STEPS = [index + 1 for index in TARGET_INDICES]


# Start, 10-step targets 1-2, 50-step targets 11-15,
# then 10-step targets 4-10, including terminal zero.
MIXED_SIGMAS = [
    1.00000000,
    0.98630137,
    0.96969697,
    0.96594427,
    0.96202532,
    0.95792880,
    0.95364238,
    0.94915254,
    0.92307692,
    0.88888889,
    0.84210526,
    0.77419355,
    0.66666667,
    0.47058824,
    0.00000000,
]


_original_forward = TimestepPreparationStage.forward


def mixed_scheduler_sigmas(scheduler):
    # UniPC applies the shift and appends the terminal zero.
    sigmas = np.asarray(MIXED_SIGMAS[:-1], dtype=np.float64)
    shift = scheduler.config.shift
    sigmas = sigmas / (shift - (shift - 1) * sigmas)
    sigmas[0] = scheduler.sigma_max
    return sigmas


def prepare_mixed_timesteps(self, batch, server_args):
    kv = getattr(batch, "wan_chai_kv", None)
    if not batch.is_warmup and kv and kv["mode"] == "reuse":
        batch.sigmas = mixed_scheduler_sigmas(self.scheduler)
        batch.num_inference_steps = len(MIXED_SIGMAS) - 1
    return _original_forward(self, batch, server_args)


# Spawned GPU workers import this module too, so they receive the same override.
TimestepPreparationStage.forward = prepare_mixed_timesteps


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
                        help="Denoising steps for mixed-schedule B (14)")
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
    if args.steps_a != 50 or args.steps_b != len(MIXED_SIGMAS) - 1:
        parser.error("This experiment requires --steps-a 50 and --steps-b 14")
    if args.wan_timestep_schedule != 0:
        parser.error("The mixed schedule requires the default Wan scheduler mode")
    if args.guidance_scale <= 1:
        parser.error("--guidance-scale must exceed 1 to exercise both CFG branches")
    return args


def build_cache_mapping(config, steps_a):
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler,
    )

    # Match the sigmas at which the model evaluates attention.
    scheduler = FlowUniPCMultistepScheduler(
        shift=config.flow_shift, timestep_schedule=config.wan_timestep_schedule,
    )
    scheduler.set_timesteps(steps_a, device="cpu")
    sigmas_a = scheduler.sigmas[:-1].tolist()
    scheduler.set_timesteps(sigmas=mixed_scheduler_sigmas(scheduler), device="cpu")
    sigmas_b = scheduler.sigmas[:-1].tolist()
    source_steps = [
        min(range(steps_a), key=lambda i: abs(sigmas_a[i] - sigmas_b[j]))
        for j in TARGET_INDICES
    ]
    mapping = [
        dict(
            a_step=i + 1, b_step=j + 1,
            a_sigma=sigmas_a[i], b_sigma=sigmas_b[j],
            sigma_difference=abs(sigmas_a[i] - sigmas_b[j]),
        )
        for i, j in zip(source_steps, TARGET_INDICES)
    ]
    return source_steps, mapping


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
        source_steps, cache_mapping = build_cache_mapping(config, args.steps_a)
        if source_steps != [5, 10]:
            raise RuntimeError(f"Unexpected 50-step source indices: {source_steps}")
        print(f"K/V mapping (A -> B): {cache_mapping}", flush=True)
        (output_dir / "experiment.json").write_text(json.dumps(dict(
            model_path=args.model_path, prompt_a=args.prompt_a, prompt_b=args.prompt_b,
            settings=settings, cache_id=cache_id, cache_steps=TARGET_STEPS,
            cache_mapping=cache_mapping, cache_mapping_policy="nearest_sigma",
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
                    or report["target_steps"] != TARGET_STEPS
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
