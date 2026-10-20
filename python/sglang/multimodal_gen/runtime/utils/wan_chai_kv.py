"""Worker-local, opt-in K/V reuse for the single-GPU Wan CHAI experiment."""

import json
from pathlib import Path


class WanChaiKVCache:
    # Default B steps 2, 3, 4. A capture steps come from scheduler sigma matching.
    STEPS = (1, 2, 3)

    def __init__(self):
        self.entries = {}
        self.cache_id = None
        self.source_signature = None
        self.request = None
        self.calls = set()
        self.events = []

    def _begin(self, request, config):
        if request.num_inference_steps < 4:
            raise ValueError("Wan CHAI requires at least 4 denoising steps")
        self.num_steps = request.num_inference_steps
        if not request.do_classifier_free_guidance or request.cfg_gate_step != 1.0:
            raise ValueError("Wan CHAI requires both CFG branches at every step")
        if any(getattr(request, flag, False) for flag in (
            "enable_teacache", "enable_cache_dit", "enable_spectrum",
            "wan_resume_latent", "wan_init_latent_path",
        )):
            raise ValueError("Wan CHAI requires full inference without other caches")
        if request.progressive_mode != "fullres":
            raise ValueError("Wan CHAI requires fullres generation")
        cache_id = config.get("cache_id")
        if not cache_id or not config.get("report_path"):
            raise ValueError("wan_chai_kv requires cache_id and report_path")
        # Zero-based B indices; omitted settings retain the original experiment.
        target_steps = config.get("target_steps", list(self.STEPS))
        if (
            not isinstance(target_steps, list)
            or not target_steps
            or any(type(i) is not int or i < 0 for i in target_steps)
            or len(set(target_steps)) != len(target_steps)
        ):
            raise ValueError("Wan CHAI requires distinct valid target step indices")
        self.target_steps = tuple(target_steps)
        if config.get("mode") in ("reuse", "off") and max(self.target_steps) >= self.num_steps:
            raise ValueError("Wan CHAI target step exceeds B's schedule")
        # Zero-based A indices, ordered by the configured B target steps.
        source_steps = config.get("source_steps")
        if (
            not isinstance(source_steps, list)
            or len(source_steps) != len(self.target_steps)
            or any(type(i) is not int or i < 0 for i in source_steps)
        ):
            raise ValueError("Wan CHAI requires one valid source step index per target step")
        self.source_steps = tuple(source_steps)
        self.calls = set()
        self.events = []

    def record(self, request, step_index, key, value):
        """Save A's selected K/V tensors and leave A's attention unchanged."""
        if request is not self.request:
            if max(self.source_steps) >= self.num_steps:
                raise ValueError("Wan CHAI source step exceeds A's schedule")
            self.entries.clear()
            self.cache_id = request.wan_chai_kv["cache_id"]
            self.source_signature = (
                request.height, request.width, request.num_frames, self.source_steps,
            )
            # Capture repeated source steps only once per CFG branch.
            self.step_map = {step: step for step in self.source_steps}
        if step_index in self.step_map:
            # K is already normalized and rotated; never rotate it again.
            call = (step_index, bool(request.is_cfg_negative))
            self.entries[call] = (key.detach().clone(), value.detach().clone())
        return key, value

    def reuse(self, request, step_index, key, value):
        """Use A's cached K/V for the configured B steps, matching the CFG branch."""
        if request is not self.request:
            expected = {(step, neg) for step in self.source_steps for neg in (False, True)}
            signature = (request.height, request.width, request.num_frames, self.source_steps)
            if (self.cache_id != request.wan_chai_kv["cache_id"]
                    or set(self.entries) != expected or self.source_signature != signature):
                raise ValueError("Missing, incomplete, or incompatible A K/V cache")
            self.step_map = dict(zip(self.target_steps, self.source_steps))
        if step_index in self.step_map:
            source_step = self.step_map[step_index]
            cached = self.entries[(source_step, bool(request.is_cfg_negative))]
            for current, saved in zip((key, value), cached):
                if (current.shape != saved.shape or current.dtype != saved.dtype
                        or current.device != saved.device):
                    raise ValueError("A/B K/V shape, dtype, or device mismatch")
            return cached
        return key, value

    def off(self, request, step_index, key, value):
        """Clear A's cache and pass through B's own K/V for the baseline."""
        if request is not self.request:
            self.entries.clear()
            self.cache_id = self.source_signature = None
            self.step_map = {}
        return key, value

    def apply(self, request, step_index, key, value):
        config = getattr(request, "wan_chai_kv", None)
        if config is None:
            # Normal requests never read or retain an experiment's cache.
            self.entries.clear()
            self.cache_id = self.source_signature = self.request = None
            return key, value
        mode = config.get("mode")
        operation = {"record": self.record, "reuse": self.reuse, "off": self.off}.get(mode)
        if operation is None:
            raise ValueError("wan_chai_kv mode must be record, reuse, or off")
        if request is not self.request:
            self._begin(request, config)
        negative = bool(request.is_cfg_negative)
        call = (step_index, negative)
        if call in self.calls or not 0 <= step_index < self.num_steps:
            raise ValueError(f"Unexpected Wan CHAI block-0 invocation: {call}")
        key, value = operation(request, step_index, key, value)
        self.request = request
        self.calls.add(call)
        if step_index in self.step_map:
            source_step = self.step_map[step_index]
            self.events.append({
                "step": step_index + 1,
                "source_step": source_step + 1,
                "block_index": 0,
                "branch": "negative" if negative else "positive",
                "action": mode,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "dtype": str(key.dtype),
            })
        if step_index == self.num_steps - 1 and negative:
            expected_calls = {(i, n) for i in range(self.num_steps) for n in (False, True)}
            if self.calls != expected_calls:
                raise ValueError(f"Wan CHAI did not observe all {self.num_steps} steps and both CFG branches")
            # Each active step runs once per CFG branch. Off has no active steps.
            expected_events = 2 * len(self.step_map)
            if len(self.events) != expected_events:
                raise ValueError("Unexpected Wan CHAI capture/replacement count")
            report = {
                "mode": mode, "cache_id": config["cache_id"],
                "num_steps": self.num_steps, "block0_calls": len(self.calls),
                "source_steps": [i + 1 for i in self.source_steps],
                "target_steps": [i + 1 for i in self.target_steps],
                "captures": sum(event["action"] == "record" for event in self.events),
                "replacements": sum(event["action"] == "reuse" for event in self.events),
                "cache_bytes": sum(t.numel() * t.element_size()
                                   for pair in self.entries.values() for t in pair),
                "events": self.events,
            }
            path = Path(config["report_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return key, value
