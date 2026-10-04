# FastWan draft -> Wan 2.1 refinement experiment

This opt-in native SGLang path exports FastWan's clean, normalized latent before
VAE decoding, then uses it to initialize base Wan 2.1. Cache-dit retains its
existing decision logic and starts with a fresh base-model cache context.

## Flags

| Flag | Meaning |
| --- | --- |
| `--wan-save-latent-path draft.pt` | Save final, gathered BCTHW model-space latents before VAE normalization/decode. Works for base Wan 2.1 and FastWan 2.1 T2V. |
| `--wan-init-latent-path draft.pt` | Load that clean latent into base Wan 2.1 T2V. |
| `--wan-refine-sigma 0.4` | Maximum actual flow noise level, not a step fraction or a video-to-video strength setting. |
| `--wan-refine-schedule rescale` | Keep all requested iterations; scale the native, already-shifted sigma schedule to the new starting sigma. Default. |
| `--wan-refine-schedule suffix` | Keep only native schedule steps at or below the requested sigma. This runs fewer iterations. |

`--num-inference-steps 50 --wan-refine-sigma 0.4 --wan-refine-schedule rescale`
runs **50 base-model refinement iterations**. Sigma spacing is the scaled
native schedule, not uniform. `suffix` uses the same 50-step native schedule
but executes only its selected tail. The log reports actual sigma and count.
Sigma=1 is capped at the native schedule maximum below 1 to avoid a UniPC
singularity; it is therefore nearly, rather than exactly, pure noise.

Initialization is `z = (1 - sigma) * draft + sigma * noise`, using the seed's
ordinary SGLang noise tensor. The native UniPC solver is reset on the selected
sigmas with `shift=1` so flow shifting is not applied twice. Cache-dit receives
the effective refinement step count for its warmup, refresh and masks.

## Run

Use your existing GPU allocation and activated SGLang environment. From the
repository root:

```bash
python python/sglang/multimodal_gen/test/unit/test_wan_refinement.py -v

FASTWAN_MODEL_PATH="$SCRATCH/models/FastWan2.1-T2V-14B-Diffusers" \
OUTPUT_DIR="$SCRATCH/sglang/outputs/week8/refinement_seed42_sigma04" \
SIGMA=0.4 SCHEDULE=rescale \
bash slurm/week8/wan_refinement_experiment.sh
```

The runner uses the existing local FastWan checkpoint/`--model-id`/DMD pipeline
recipe in this repository. It starts with one GPU and produces six cases:

1. Three-step FastWan draft, with latent export.
2. Base Wan full computation, also exported as a control draft.
3. Ordinary base Wan with cache-dit.
4. FastWan draft refinement without cache-dit.
5. Identical FastWan draft refinement with cache-dit.
6. Base-generated draft refinement with cache-dit and the same schedule/noise seed.

Use separate output directories for repeats; video, latent and perf outputs
are replaced when names repeat, while the external CSV hook may append.
Repeat over multiple prompts and seeds. Try sigma 0.2, 0.4, 0.6, then compare
`rescale` with `suffix`. Keep threshold, Fn/Bn, guidance and other acceleration
settings fixed. Keep the same resolution/frame count in both generation stages.

## Measurements and interpretation

Each case saves a video, a perf JSON, and logs. Existing `CACHE_DIT_DECISION_CSV`,
`CACHE_DIT_RUN_LABEL` and `CACHE_DIT_RUN_ID` hooks are used for decisions; they
require the instrumented cache-dit already used by the week7 scripts. An
upstream cache-dit without those hooks may not create the CSV. Do not interpret
a missing CSV as zero hits. No threshold logic is modified by this patch.

Compare cached/full calls, decision reasons and residual differences separately
for each CFG branch. Warmup/forced-refresh decisions are not threshold misses.
Use full vs cached refinement to assess approximation damage, and compare both
against ordinary Wan for motion, anatomy and prompt adherence.

For speed, include draft generation plus refinement, file I/O and any model
loading/switching costs. These are separate CLI processes: process startup and
model loading can dominate. Perf denoising time measures only the denoising
portion, not the complete two-stage service. The experiment still decodes the
draft for visual inspection. Higher hit rate over a compressed sigma range
alone does not show that FastWan supplies an intrinsically more cacheable input.

## Scope and checks

This experiment targets offline `sglang generate` with native pipelines,
one prompt and one output per invocation. It does not add an HTTP API or a
single-process two-model serving scheduler. It rejects mismatched latent shape,
frame count, resolution, VAE normalization, non-finite values, progressive
resolution, rollout mode, and refinement in the DMD pipeline. It does not
support arbitrary MP4 input, resizing, Wan 2.2, or cross-model feature-cache reuse.
Compatible VAE weights are still required; metadata verifies normalization and
layout, not a full weight checksum. Use exported latents from these pipelines.

The preparation stage runs before sequence sharding; export runs after gathering
and unpadding. Only rank zero writes the file; warmup does not load or export
drafts. Multi-GPU behavior needs GPU validation after the single-GPU experiment.

CPU tests cover schedule spacing, tail selection, invalid configurations and
(when PyTorch is installed) latent round-tripping and rejection of invalid files.
Before interpreting research results, run the tests in the cluster environment
and verify all six cases complete and save correctly. This patch has not been
validated by actual Wan generation on the Windows development machine.
