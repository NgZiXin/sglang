#!/usr/bin/env python3
"""Time Wan blocks with CUDA events; reuse SGLang's existing step timings."""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


def install_profiling():
    import torch
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage

    original_denoise = DenoisingStage._denoise
    original_step = DenoisingStage._run_denoising_step
    events = []
    block_numbers = {}
    end_events = {}
    step = None
    calls = defaultdict(int)

    def before_block(module, args):
        block = block_numbers[module]
        calls[step, block] += 1
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record(torch.cuda.current_stream())
        events.append((step, block, calls[step, block], begin, end))
        end_events[module] = end

    def after_block(module, args, output):
        end_events.pop(module).record(torch.cuda.current_stream())

    def denoise(stage, batch, server_args):
        if batch.is_warmup:
            return original_denoise(stage, batch, server_args)
        if server_args.num_gpus != 1 or stage.transformer_2 is not None:
            raise ValueError("This experiment requires one GPU and one Wan transformer")
        events.clear()
        calls.clear()
        handles = []
        blocks = stage.transformer.blocks
        try:
            for number, block in enumerate(blocks, start=1):
                block_numbers[block] = number
                handles.append(block.register_forward_pre_hook(before_block))
                handles.append(block.register_forward_hook(after_block, always_call=True))
            result = original_denoise(stage, batch, server_args)
            torch.cuda.synchronize()  # Read all events once after denoising.
            timings = [
                dict(step=s, block=b, call=c, duration_ms=begin.elapsed_time(end))
                for s, b, c, begin, end in events
            ]
            with open(os.environ["WAN_PROFILE_PATH"], "x", encoding="utf-8") as file:
                json.dump(dict(block_count=len(blocks), timings=timings), file)
            return result
        finally:
            for handle in handles:
                handle.remove()
            block_numbers.clear()
            end_events.clear()

    def run_step(stage, ctx, state, batch, server_args):
        # Track the step number for block records; SGLang handles step timing.
        nonlocal step
        step = state.step_index + 1
        return original_step(stage, ctx, state, batch, server_args)

    DenoisingStage._denoise = denoise
    DenoisingStage._run_denoising_step = run_step


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run_dir):
    manifest = json.loads((run_dir / "prompts.json").read_text(encoding="utf-8"))
    ids = [p["prompt_id"] for p in manifest["prompts"]]
    if len(ids) != 33 or len(set(ids)) != 33:
        raise ValueError("Expected 33 unique prompts")
    steps, blocks, block_calls = defaultdict(list), defaultdict(list), defaultdict(list)
    block_totals = defaultdict(list)
    block_count = None
    for prompt_id in ids:
        report = json.loads((run_dir / "raw" / f"{prompt_id}.json").read_text(encoding="utf-8"))
        if block_count is None:
            block_count = report["block_count"]
        if block_count <= 0 or report["block_count"] != block_count:
            raise ValueError(f"Unexpected block count: {prompt_id}")
        perf = json.loads((run_dir / "perf" / f"{prompt_id}.json").read_text(encoding="utf-8"))
        prompt_steps, prompt_blocks = defaultdict(list), defaultdict(list)
        for row in perf["denoise_steps_ms"]:
            ms = row["duration_ms"]
            if not math.isfinite(ms) or ms < 0:
                raise ValueError(f"Invalid step duration: {prompt_id}")
            prompt_steps[row["step"] + 1].append(ms)  # Existing JSON is zero-based.
        for row in report["timings"]:
            s, b, ms = row["step"], row["block"], row["duration_ms"]
            if not math.isfinite(ms) or ms < 0:
                raise ValueError(f"Invalid duration: {prompt_id}")
            prompt_blocks[s, b].append(ms)
        if set(prompt_steps) != set(range(1, 51)) or any(len(v) != 1 for v in prompt_steps.values()):
            raise ValueError(f"Expected exactly 50 steps: {prompt_id}")
        expected = {(s, b) for s in range(1, 51) for b in range(1, block_count + 1)}
        if set(prompt_blocks) != expected:
            raise ValueError(f"Missing or unexpected blocks: {prompt_id}")
        for s, values in prompt_steps.items():
            steps[s].extend(values)
        for (s, b), values in prompt_blocks.items():
            blocks[s, b].append(sum(values))  # Include all CFG calls in this step.
            block_totals[b].append(sum(values))
            block_calls[b].extend(values)
    write_csv(run_dir / "step_summary.csv", [
        dict(step=s, samples=len(values), mean_ms=mean(values))
        for s, values in sorted(steps.items())
    ])
    write_csv(run_dir / "block_by_step_summary.csv", [
        dict(step=s, block=b, samples=len(values), mean_ms=mean(values))
        for (s, b), values in sorted(blocks.items())
    ])
    write_csv(run_dir / "block_summary.csv", [
        dict(block=b, samples=len(block_totals[b]), mean_ms=mean(block_totals[b]),
             mean_per_call_ms=mean(block_calls[b]))
        for b in range(1, block_count + 1)
    ])
    print(f"Saved block and step summaries for 33 videos in {run_dir}")


# SGLang's spawned GPU workers re-execute this script as __mp_main__.
if __name__ == "__mp_main__" and os.environ.get("WAN_PROFILE_PATH"):
    install_profiling()

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "summarize":
        summarize(Path(sys.argv[2]))
    else:
        install_profiling()
        from sglang.cli.main import main
        main()
