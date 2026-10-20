"""Balloon-to-kite comparison: original CHAI, source-masked CHAI, and uncached B."""

import argparse
import csv
import json
import re
import time
import uuid
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt-a", default="A balloon floats above a grassy field in a gentle breeze, low-angle shot.")
    parser.add_argument("--prompt-b", default="A kite floats above a grassy field in a gentle breeze, low-angle shot.")
    parser.add_argument("--exclude-word", default="balloon")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Exclude source tokens above this fraction of peak word attention")
    parser.add_argument("--flow-shift", type=float, default=None)
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
    args = parser.parse_args()
    if args.steps_a < 4 or args.steps_b < 4:
        parser.error("--steps-a and --steps-b must be at least 4 for steps 2-4 reuse")
    if args.guidance_scale <= 1:
        parser.error("--guidance-scale must exceed 1 to exercise both CFG branches")
    if not 0 < args.mask_threshold < 1:
        parser.error("--mask-threshold must be between 0 and 1")
    return args


def resolve_mask_tokens(model_path, config, prompt, word):
    """Use the pipeline's tokenizer settings; select word positions, not vocab IDs."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, subfolder="tokenizer", use_fast=True, padding_side="right",
    )
    preprocess = config.preprocess_text_funcs[0]
    text = preprocess(prompt) if preprocess else prompt
    matches = list(re.finditer(r"\b" + re.escape(word) + r"\b", text))
    if len(matches) != 1:
        raise ValueError("--exclude-word must occur exactly once in the processed A prompt")
    kwargs = dict(config.text_encoder_configs[0].tokenizer_kwargs)
    kwargs.update(config.text_encoder_extra_args[0] or {})
    tokens = config.tokenize_prompt([text], tokenizer, dict(kwargs, return_offsets_mapping=True))
    start, end = matches[0].span()
    positions = [
        i for i, (left, right) in enumerate(tokens["offset_mapping"][0].tolist())
        if right > start and left < end and right > left
    ]
    if not positions:
        raise ValueError("Excluded word was truncated or could not be mapped to text tokens")
    ids = tokens["input_ids"][0].tolist()
    pieces = tokenizer.convert_ids_to_tokens([ids[i] for i in positions])
    print(f"Source exclusion: {word!r}, positions={positions}, pieces={pieces}", flush=True)
    return positions, pieces


def save_mask_previews(output_dir):
    """Show scores (top) and excluded tokens (bottom) at three latent times."""
    import torch
    from PIL import Image, ImageDraw

    for path in sorted(output_dir.glob("source_mask_step_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=True)
        scores = data["scores"].reshape(data["grid_shape"])
        excluded = (~data["keep"]).reshape(data["grid_shape"])
        scores = scores.nan_to_num()
        scores = scores / scores.max().clamp_min(1e-8)
        preview = Image.new("RGB", (768, 350), "white")
        draw = ImageDraw.Draw(preview)
        for col, frame in enumerate((0, scores.shape[0] // 2, scores.shape[0] - 1)):
            draw.text((col * 256 + 4, 4), f"Latent frame {frame}", fill="black")
            for row, values in enumerate((scores[frame], excluded[frame])):
                tile = Image.fromarray((values.float() * 255).byte().numpy())
                preview.paste(tile.resize((256, 144), Image.Resampling.NEAREST),
                              (col * 256, 24 + row * 144))
        draw.text((4, 315), "Top: word attention. Bottom: white = excluded cached tokens.", fill="black")
        draw.text((4, 331), f"Fallback: {data['fallback_reason'] or 'none'}", fill="black")
        preview.save(path.with_suffix(".png"))


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
        **({"flow_shift": args.flow_shift} if args.flow_shift is not None else {}),
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
        mask_tokens, mask_pieces = resolve_mask_tokens(
            generator.server_args.model_path, config, args.prompt_a, args.exclude_word,
        )
        source_steps, cache_mapping = build_cache_mapping(
            config, args.steps_a, args.steps_b,
        )
        print(f"K/V mapping (A -> B): {cache_mapping}", flush=True)
        (output_dir / "experiment.json").write_text(json.dumps(dict(
            model_path=args.model_path, prompt_a=args.prompt_a, prompt_b=args.prompt_b,
            settings=settings, cache_id=cache_id, cache_steps=[2, 3, 4],
            cache_mapping=cache_mapping, cache_mapping_policy="nearest_sigma",
            steps_by_run={"a_full": args.steps_a,
                          "b_chai_kv": args.steps_b, "b_chai_mask": args.steps_b,
                          "b_uncached": args.steps_b},
            exclude_word=args.exclude_word, mask_token_indices=mask_tokens,
            mask_token_pieces=mask_pieces, mask_threshold=args.mask_threshold,
            block_index=0, cfg_branches=["positive", "negative"],
            cache_format="projected K after QK norm/RoPE; projected V",
            flow_shift=config.flow_shift,
            timestep_schedule=config.wan_timestep_schedule,
        ), indent=2), encoding="utf-8")

        for label, prompt, mode, num_steps in (
            ("a_full", args.prompt_a, "record", args.steps_a),
            ("b_chai_kv", args.prompt_b, "reuse", args.steps_b),
            ("b_chai_mask", args.prompt_b, "reuse", args.steps_b),
            ("b_uncached", args.prompt_b, "off", args.steps_b),
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
                                 source_steps=source_steps,
                                 mask_token_indices=mask_tokens if mode == "record" else None,
                                 mask_threshold=args.mask_threshold,
                                 mask_cache=label == "b_chai_mask",
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
                    or report["source_steps"] != [i + 1 for i in source_steps]
                    or report["target_steps"] != [2, 3, 4]
                    or report["num_steps"] != num_steps
                    or report["block0_calls"] != 2 * num_steps
                    or (report["captures"], report["replacements"] + report["fallbacks"]) != expected
                    or (label != "b_chai_mask" and report["fallbacks"] != 0)):
                raise RuntimeError(f"Unexpected K/V audit report: {report}")
            if mode == "record":
                if {m["source_step"] for m in report["source_masks"]} != {s + 1 for s in source_steps}:
                    raise RuntimeError("A did not capture all required source masks")
                save_mask_previews(output_dir)
            stages = {entry["name"]: entry["duration_ms"] for entry in perf["steps"]}
            rows.append(dict(
                label=label, actual_steps=num_steps, wall_seconds=elapsed,
                denoising_ms=stages["DenoisingStage"],
                decoding_ms=stages["DecodingStage"],
                captures=report["captures"], replacements=report["replacements"],
                fallbacks=report["fallbacks"],
                cache_bytes=report["cache_bytes"],
            ))
            with (output_dir / "summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"{label}: {num_steps} steps, {report['captures']} captures, "
                  f"{report['replacements']} replacements, {report['fallbacks']} fallbacks",
                  flush=True)
            del result
    finally:
        generator.shutdown()
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main(parse_args())
