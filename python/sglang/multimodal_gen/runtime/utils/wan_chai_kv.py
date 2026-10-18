"""Worker-local, opt-in K/V reuse for the single-GPU Wan CHAI experiment."""

import json
from pathlib import Path


class WanChaiKVCache:
    # Human steps 2, 3, 4; only block 0 owns this controller.
    STEPS = (1, 2, 3)
    NUM_STEPS = 40

    def __init__(self):
        self.entries = {}
        self.cache_id = None
        self.source_signature = None
        self.request = None
        self.calls = set()
        self.events = []

    def _begin(self, request, config):
        mode = config.get("mode")
        if mode not in ("record", "reuse", "off"):
            raise ValueError("wan_chai_kv mode must be record, reuse, or off")
        if request.num_inference_steps != self.NUM_STEPS:
            raise ValueError("Wan CHAI experiment requires exactly 40 steps")
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
        signature = (request.height, request.width, request.num_frames)
        if mode == "reuse":
            expected = {(step, neg) for step in self.STEPS for neg in (False, True)}
            if (self.cache_id != cache_id or set(self.entries) != expected
                    or self.source_signature != signature):
                raise ValueError("Missing, incomplete, or incompatible A K/V cache")
        else:
            self.entries.clear()
            self.cache_id = cache_id if mode == "record" else None
            self.source_signature = signature if mode == "record" else None
        self.request = request
        self.calls = set()
        self.events = []

    def apply(self, request, step_index, key, value):
        config = getattr(request, "wan_chai_kv", None)
        if config is None:
            # Normal requests never read or retain an experiment's cache.
            self.entries.clear()
            self.cache_id = self.source_signature = self.request = None
            return key, value
        if request is not self.request:
            self._begin(request, config)
        negative = bool(request.is_cfg_negative)
        call = (step_index, negative)
        if call in self.calls or not 0 <= step_index < self.NUM_STEPS:
            raise ValueError(f"Unexpected Wan CHAI block-0 invocation: {call}")
        self.calls.add(call)
        mode = config["mode"]
        if step_index in self.STEPS and mode != "off":
            if mode == "record":
                # K is already normalized and rotated; never rotate it again.
                self.entries[call] = (key.detach().clone(), value.detach().clone())
            else:
                cached = self.entries[call]
                for current, saved in zip((key, value), cached):
                    if (current.shape != saved.shape or current.dtype != saved.dtype
                            or current.device != saved.device):
                        raise ValueError("A/B K/V shape, dtype, or device mismatch")
                key, value = cached
            self.events.append({
                "step": step_index + 1,
                "block_index": 0,
                "branch": "negative" if negative else "positive",
                "action": mode,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "dtype": str(key.dtype),
            })
        if step_index == self.NUM_STEPS - 1 and negative:
            expected_calls = {(i, n) for i in range(self.NUM_STEPS) for n in (False, True)}
            if self.calls != expected_calls:
                raise ValueError("Wan CHAI did not observe all 40 steps and both CFG branches")
            expected_events = 0 if mode == "off" else 6
            if len(self.events) != expected_events:
                raise ValueError("Unexpected Wan CHAI capture/replacement count")
            report = {
                "mode": mode, "cache_id": config["cache_id"],
                "num_steps": self.NUM_STEPS, "block0_calls": len(self.calls),
                "captures": len(self.events) if mode == "record" else 0,
                "replacements": len(self.events) if mode == "reuse" else 0,
                "cache_bytes": sum(t.numel() * t.element_size()
                                   for pair in self.entries.values() for t in pair),
                "events": self.events,
            }
            path = Path(config["report_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return key, value
