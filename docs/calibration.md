# Calibration status

This file is the honest ledger of what in the simulator is **measured** and what is
**invented**. Nothing from an uncalibrated row may appear in the thesis as a quantity.
Ranking claims ("policy A recomputes less than policy B") survive uncalibrated
constants; magnitude claims ("saves RM X per 1000 calls") do not.

## Status legend

| status | meaning |
|---|---|
| MEASURED | fitted to a run on the target hardware, with the run recorded |
| DERIVED | computed from published model/hardware specs, arithmetic shown |
| INVENTED | a plausible-looking number with no evidence behind it |

## Engine timing (`sim/config.py::EngineConfig`) -- MEASURED 2026-08-01

vLLM 0.26.0, Qwen2.5-3B-Instruct, RTX 5080, WSL2, prefix caching OFF,
`gpu_memory_utilization=0.85`. Raw samples and fits: `results/calib/timing_fit.json`.
Reproduce with `bench/serve_calib.sh` then `bench/fit_timing.py`.

| parameter | derived guess | MEASURED | fit quality |
|---|---|---|---|
| `prefill_s_per_token` | 8.3e-5 (implied) | **6.669e-5** | R^2 1.0000 (quadratic fit) |
| `prefill_s_per_token2` | **absent from the model** | **1.954e-9** | R^2 1.0000 |
| `decode_s_per_kv_token` | 3.8e-8 | **5.994e-8** (+58%) | R^2 0.972 |
| `step_overhead_s` + `decode_s_per_seq` | 6.5 ms | **9.06 ms** (+39%) | see "separated" below |
| `kv_pool_blocks` | 16000 | **12868** (-20%) | read from startup log |

**Every guess was wrong by 20-58%, and one whole term was missing.** This is the entire
justification for the "read it, do not derive it" rule.

### Why the KV pool was 20% smaller than derived

The derivation `(16 - 6.2 - 1) GiB / (36864 B/token x 16)` assumed 1 GiB of headroom on
a 16 GiB card. Two things it missed:

- the Windows desktop permanently holds ~1.3 GiB of the 15.92 GiB, so only ~14.6 GiB is
  free at startup -- vLLM 0.26's default `gpu_memory_utilization` of 0.92 cannot even be
  satisfied and the server refuses to start;
- vLLM's own activation buffers and CUDA-graph capture take more than the assumed 1 GiB.

vLLM reported `Available KV cache memory: 7.07 GiB` -> `GPU KV cache size: 205,888
tokens` -> 12868 blocks at `block_size=16`. A headless HPC node has no desktop holding
memory and will land somewhere else again, which is a concrete reason local and HPC
numbers must never appear in the same figure.

### Prefill is not linear -- fixed, not just noted

Measured prefill is superlinear: doubling the prompt multiplied the time by 1.97, 2.02,
2.16 and 2.37 as it grew from 1k to 16k, because attention is quadratic in prompt
length. A straight-line fit lands with a **negative** intercept of -50.8 ms, which is
not a physical quantity -- the sign is the tell that the model was mis-specified.

A quadratic fits exactly: `t = 9.42 ms + 6.669e-5*n + 1.954e-9*n^2`, R^2 = 1.0000.

`sim/engine.py` now charges prefill **per position** rather than per token. The token at
position `p` costs `b + 2cp`, so a chunk covering `[s, e)` costs
`b*(e-s) + c*(e^2 - s^2)`.

This is not cosmetic for this project. After a partial prefix-cache hit the recomputed
suffix begins at a high position, and each of its tokens must attend over the entire
cached prefix. The old linear model charged those tokens the same as tokens at position
zero, understating the cost of a partial hit -- which is the central quantity the whole
thesis measures. Marginal rate is 15.0k tok/s at position 0 and 7.6k tok/s at position
16k. Guarded by `tests/test_invariants.py::test_prefill_is_charged_per_position`.

### WSL-specific caveats that do not transfer to HPC

- `VLLM_WSL2_ENABLE_PIN_MEMORY=1` is required; vLLM disables pinned memory under WSL by
  default and the V1 worker then dies with a misleading "UVA is not available".
- `VLLM_USE_FLASHINFER_SAMPLER=0` is required, because FlashInfer JIT-compiles its
  sampling kernels and the torch wheel ships the CUDA runtime without `nvcc`. The native
  sampler is slower by a fixed per-step amount that lands inside the measured
  `step_overhead_s`, so an HPC node with a full toolkit will measure a smaller one.

KV bytes per token for Qwen2.5-3B (bf16, as vLLM loads it):
`2 (K,V) x 36 layers x 2 KV heads x 128 head_dim x 2 bytes = 36864 B/token`.
Cross-check: 7.07 GiB / 36864 B = 205,900 tokens, against the 205,888 reported. The
per-token arithmetic was right; the memory budget was not.

## Workload (`sim/config.py::WorkloadConfig`)

| parameter | value | status | basis |
|---|---|---|---|
| `pause_seconds_median` | 2.0 | INVENTED | -- |
| `pause_seconds_sigma` | 1.0 | INVENTED | -- |
| `tool_result_tokens_median` | 700 | INVENTED | -- |
| `tool_result_tokens_sigma` | 1.1 | INVENTED | -- |
| `output_tokens_median` | 160 | INVENTED | -- |
| `turns_min/max` | 12 / 30 | INVENTED | matches the 20-30 calls per task quoted in the proposal |
| `system_prompt_tokens` | 1200 | INVENTED | -- |
| `tool_pause_spread` | 0.9 | INVENTED | **this one sets how predictable the world is** |
| `tool_markov_self_prob` | 0.45 | INVENTED | tool-use inertia, cf. AutoTool |

The pause distribution is the single most load-bearing invented number in the project.
Every headroom figure is a function of it.

**To calibrate:** the machinery now exists. `docs/trace_schema.md` defines the six
required fields, and `bench/fit_workload.py` turns a trace into these parameters:

```
python -m bench.fit_workload --self-test          # verify the fitter first
python -m bench.fit_workload --trace runs.jsonl   # then point it at real data
```

Run one real coding agent (OpenHands, SWE-agent, or Claude Code) over a handful of
SWE-bench instances with timestamped logging. Use the trace *only* to fit distributions
-- the experiments still run on synthetic traces, so they stay reproducible and free of
API keys. This does not violate the "no real agents" rule; it is what makes the
synthetic generator defensible.

The self-test is the part that matters: it generates traces at known parameters and
checks the fitter inverts its own generator. On first run it failed 7 of 9 parameters
and exposed three separate defects -- a marginal-instead-of-within-tool sigma fit, an
un-inverted Markov transition rule, and a generator whose per-tool multipliers had a
geometric mean that was not 1, so that fit-then-generate did not round-trip. None of
those would have announced themselves on real data; the fitter returns plausible
numbers either way.

Until that is done, every result must be reported as a **sensitivity curve over
`tool_pause_spread`**, not as a single number. A predictor's value is bounded above by
how much structure the generator was handed.

## Cost (`sim/config.py::CostConfig`)

| parameter | value | status |
|---|---|---|
| T4 16GB | RM 3.06/h | MEASURED (Sunway HPC price list) |
| RTX PRO 6000 96GB, 4 core | RM 21.00/h | MEASURED |
| RTX PRO 6000 96GB, 16 core | RM 32.91/h | MEASURED |
| L4 / L40S / A10G | unknown | TODO |

Cost is charged on wall-clock, so `rm_total = makespan_s / 3600 * rm_per_hour`. Note
what this implies: **a cache win only becomes a cost win when the GPU is the
bottleneck.** Check `gpu_busy_frac` before quoting any RM figure -- below saturation,
recomputing fewer tokens shortens nothing you are paying for.

## Model deviations from a real engine

Documented in `sim/cache.py` and `sim/engine.py`. All of them apply identically to
every policy arm, so they cannot bias a policy comparison, but they do bias absolute
numbers:

- no CPU/disk KV offload tier;
- whole-prompt admission (a request is admitted only if its entire prompt fits);
- preemption always recomputes, never swaps;
- a block becomes reusable when its producer's prefill step ends, not sub-step;
- among expired blocks, eviction orders by expiry rather than last use (identical
  whenever TTLs are uniform, which covers `lru` and `const_ttl` exactly).
