"""Worker-local, opt-in K/V reuse for the single-GPU Wan CHAI experiment."""

import json
from pathlib import Path


class WanChaiKVCache:
    # Default B steps 2, 3, 4. A capture steps come from scheduler sigma matching.
    STEPS = (1, 2, 3)

    def __init__(self):
        self.entries = {}
        self.source_masks = {}
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
        token_indices = config.get("mask_token_indices")
        if token_indices is not None:
            if (not isinstance(token_indices, list) or not token_indices
                    or any(type(i) is not int or i < 0 for i in token_indices)
                    or len(set(token_indices)) != len(token_indices)):
                raise ValueError("mask_token_indices must contain distinct text-token positions")
            if not 0 < config.get("mask_threshold", 0.5) < 1:
                raise ValueError("mask_threshold must be between 0 and 1")
        self.calls = set()
        self.events = []

    def capture_source_mask(self, request, step_index, query, key, scale):
        """Probe positive text attention; do not change the source attention output."""
        config = request.wan_chai_kv
        token_indices = config.get("mask_token_indices")
        if (config["mode"] != "record" or token_indices is None
                or request.is_cfg_negative or step_index not in self.step_map):
            return
        import torch

        if query.shape[0] != 1 or key.shape[0] != 1:
            raise ValueError("Masked Wan CHAI supports one sample per request")
        if max(token_indices) >= key.shape[1]:
            raise ValueError("Mask text-token position exceeds the cross-attention context")
        # Match live attention, including padded text keys in the denominator.
        # Q/K arrive as [B, sequence, heads, dim]. Bound the probe's memory.
        with torch.no_grad():
            text_key = key.detach().transpose(1, 2).float().transpose(-1, -2)
            scores = []
            for chunk in query.detach().split(256, dim=1):
                logits = (chunk.transpose(1, 2).float() @ text_key) * scale
                probs = logits.softmax(dim=-1)
                scores.append(probs[..., token_indices].sum(-1).mean(1))
            relevance = torch.cat(scores, dim=1)[0]
            peak, low = relevance.max().item(), relevance.min().item()
            reason = None
            if not torch.isfinite(relevance).all().item():
                reason = "nonfinite_attention"
            elif peak <= 1e-8 or peak - low <= max(1e-8, peak * 1e-3):
                reason = "flat_attention"
            keep = relevance <= config.get("mask_threshold", 0.5) * peak
            indices = keep.nonzero(as_tuple=True)[0]
            if not reason and indices.numel() == 0:
                reason = "empty_cache"
            info = dict(
                source_step=step_index + 1, total_tokens=relevance.numel(),
                kept_tokens=indices.numel(),
                excluded_fraction=1 - indices.numel() / relevance.numel(),
                fallback_reason=reason,
            )
            self.source_masks[step_index] = (indices, info)
            path = Path(config["report_path"]).parent / f"source_mask_step_{step_index + 1}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(
                scores=relevance.cpu(), keep=keep.cpu(),
                grid_shape=config.get("mask_grid_shape"),
                token_indices=token_indices,
                threshold=config.get("mask_threshold", 0.5), **info,
            ), path)

    def record(self, request, step_index, key, value):
        """Save A's selected K/V tensors and leave A's attention unchanged."""
        if request is not self.request:
            if max(self.source_steps) >= self.num_steps:
                raise ValueError("Wan CHAI source step exceeds A's schedule")
            self.entries.clear()
            self.source_masks.clear()
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
            if request.wan_chai_kv.get("mask_cache") and set(self.source_masks) != set(self.source_steps):
                raise ValueError("Missing source attention masks; record A with mask_token_indices")
        if step_index in self.step_map:
            source_step = self.step_map[step_index]
            cached = self.entries[(source_step, bool(request.is_cfg_negative))]
            for current, saved in zip((key, value), cached):
                if (current.shape != saved.shape or current.dtype != saved.dtype
                        or current.device != saved.device):
                    raise ValueError("A/B K/V shape, dtype, or device mismatch")
            if request.wan_chai_kv.get("mask_cache"):
                indices, info = self.source_masks[source_step]
                self.reuse_info = info
                if info["fallback_reason"]:
                    return key, value
                # K already contains its original RoPE; select without re-rotating.
                return tuple(t.index_select(1, indices).contiguous() for t in cached)
            return cached
        return key, value

    def off(self, request, step_index, key, value):
        """Clear A's cache and pass through B's own K/V for the baseline."""
        if request is not self.request:
            self.entries.clear()
            self.source_masks.clear()
            self.cache_id = self.source_signature = None
            self.step_map = {}
        return key, value

    def apply(self, request, step_index, key, value):
        config = getattr(request, "wan_chai_kv", None)
        if config is None:
            # Normal requests never read or retain an experiment's cache.
            self.entries.clear()
            self.source_masks.clear()
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
        self.reuse_info = {}
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
                "action": "fallback" if self.reuse_info.get("fallback_reason") else mode,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "dtype": str(key.dtype),
                **self.reuse_info,
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
                "fallbacks": sum(event["action"] == "fallback" for event in self.events),
                "source_masks": [info for _, info in self.source_masks.values()],
                "cache_bytes": sum(t.numel() * t.element_size()
                                   for pair in self.entries.values() for t in pair),
                "events": self.events,
            }
            path = Path(config["report_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return key, value
