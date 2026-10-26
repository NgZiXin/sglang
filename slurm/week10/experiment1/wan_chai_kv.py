"""Compare Wan B with replacement, concatenated, and no source K/V.

Copied and adapted from slurm/week9/wan_chai_kv.py.
"""

import argparse
import csv
import json
import time
import uuid
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--prompt-a", default="A party on a sandy beach beside gentle waves, wide shot.")
    parser.add_argument("--prompt-b", default="Gentle waves roll onto an empty sandy beach, wide shot.")
    parser.add_argument("--negative-prompt", default=None,
                        help="Shared by all runs; omit to use the model default")
    parser.add_argument("--steps-a", type=int, required=True,
                        help="Denoising steps for source request A (at least 4)")
    parser.add_argument("--steps-b", type=int, required=True,
                        help="Denoising steps for all three B runs (at least 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.steps_a < 4 or args.steps_b < 4:
        parser.error("--steps-a and --steps-b must be at least 4 for steps 2-4 reuse")
    if args.guidance_scale <= 1:
        parser.error("--guidance-scale must exceed 1 to exercise both CFG branches")
    return args


def build_cache_mapping(config, steps_a, steps_b):
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler,
    )

    # Construct exactly the same schedules as WanPipeline. Use sigmas rather
    # than integer timesteps, which lose precision through truncation.
    schedules = []
    for num_steps in (steps_a, steps_b):
        scheduler = FlowUniPCMultistepScheduler(
            shift=config.flow_shift, timestep_schedule=config.wan_timestep_schedule,
        )
        scheduler.set_timesteps(num_steps, device="cpu")
        schedules.append(scheduler.sigmas[:-1].tolist())
    sigmas_a, sigmas_b = schedules
    # B evaluations 2-4, excluding the terminal sigma from both schedules.
    source_steps = [
        min(range(steps_a), key=lambda i: abs(sigmas_a[i] - sigmas_b[j]))
        for j in (1, 2, 3)
    ]
    mapping = [
        dict(
            a_step=i + 1, b_step=j + 1,
            a_sigma=sigmas_a[i], b_sigma=sigmas_b[j],
            sigma_difference=abs(sigmas_a[i] - sigmas_b[j]),
        )
        for i, j in zip(source_steps, (1, 2, 3))
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
        source_steps, cache_mapping = build_cache_mapping(
            config, args.steps_a, args.steps_b,
        )
        print(f"K/V mapping (A -> B): {cache_mapping}", flush=True)
        (output_dir / "experiment.json").write_text(json.dumps(dict(
            model_path=args.model_path, prompt_a=args.prompt_a, prompt_b=args.prompt_b,
            settings=settings, cache_id=cache_id, cache_steps=[2, 3, 4],
            cache_mapping=cache_mapping, cache_mapping_policy="nearest_sigma",
            steps_by_run={"a_full": args.steps_a,
                          "b_chai_kv": args.steps_b, "b_concat": args.steps_b,
                          "b_full": args.steps_b},
            reuse_strategy_by_run={"a_full": "record", "b_chai_kv": "replace",
                                   "b_concat": "concat", "b_full": "off"},
            concat_order=["B_current", "A_cached"], concat_dimension=1,
            block_index=0, cfg_branches=["positive", "negative"],
            cache_format="projected K after QK norm/RoPE; projected V",
            flow_shift=config.flow_shift,
            timestep_schedule=config.wan_timestep_schedule,
        ), indent=2), encoding="utf-8")

        # Record A once; compare both reuse strategies against the same cache.
        # Off runs last because it clears the cached source tensors.
        for label, prompt, mode, strategy, num_steps in (
            ("a_full", args.prompt_a, "record", "replace", args.steps_a),
            ("b_chai_kv", args.prompt_b, "reuse", "replace", args.steps_b),
            ("b_concat", args.prompt_b, "reuse", "concat", args.steps_b),
            ("b_full", args.prompt_b, "off", "replace", args.steps_b),
        ):
            perf_path = output_dir / f"{label}_perf.json"
            report_path = output_dir / f"{label}_kv.json"
            # A new cache ID also prevents accepting reports from an older run.
            started = time.perf_counter()
            result = generator.generate(dict(
                settings, prompt=prompt, num_inference_steps=num_steps,
                output_file_name=f"{label}.mp4",
                perf_dump_path=str(perf_path),
                wan_chai_kv=dict(mode=mode, reuse_strategy=strategy, cache_id=cache_id,
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
                "reuse": (0, 6),
                "off": (0, 0),
            }[mode]
            if (report["cache_id"] != cache_id or report["mode"] != mode
                    or report.get("reuse_strategy") != strategy
                    or report["source_steps"] != [i + 1 for i in source_steps]
                    or report["target_steps"] != [2, 3, 4]
                    or report["num_steps"] != num_steps
                    or report["block0_calls"] != 2 * num_steps
                    or (report["captures"], report["replacements"]) != expected):
                raise RuntimeError(f"Unexpected K/V audit report: {report}")
            multiplier = 2 if mode == "reuse" and strategy == "concat" else 1
            expected_steps = (set(i + 1 for i in source_steps) if mode == "record"
                              else {2, 3, 4} if mode == "reuse" else set())
            expected_events = {(step, branch) for step in expected_steps
                               for branch in ("positive", "negative")}
            events = report["events"]
            if ({(event["step"], event["branch"]) for event in events} != expected_events
                    or len(events) != len(expected_events)):
                raise RuntimeError("Missing or duplicate K/V audit events")
            source_by_target = dict(zip((2, 3, 4), [i + 1 for i in source_steps]))
            for event in events:
                source_step = (source_by_target[event["step"]] if mode == "reuse"
                               else event["step"])
                if (event.get("reuse_strategy") != strategy
                        or event["source_step"] != source_step
                        or event["block_index"] != 0 or event["action"] != mode
                        or event["key_shape"] != event["value_shape"]
                        or event["key_shape"][1] != multiplier * event["input_tokens"]):
                    raise RuntimeError(f"Unexpected K/V bank layout: {event}")
            stages = {entry["name"]: entry["duration_ms"] for entry in perf["steps"]}
            rows.append(dict(
                label=label, reuse_strategy=strategy if mode == "reuse" else "none",
                kv_token_multiplier=multiplier,
                actual_steps=num_steps, wall_seconds=elapsed,
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
                  f"{report['replacements']} reuse calls, strategy={strategy}", flush=True)
            del result
    finally:
        generator.shutdown()
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main(parse_args())
