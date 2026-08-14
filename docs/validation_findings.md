# Validation against vLLM — the timing model holds, the eviction model does not

**Date:** 2026-08-02 · **Data:** `results/validate/` (pressure 0.64) and
`results/validate_pressured/` (pressure 1.02) · **Harness:** `bench/validate_vs_vllm.py`

Every number in this project comes from a simulator whose timing constants were fitted
to vLLM but whose *behaviour* had never been compared to it. This closes that gap, and
the answer is split: **the timing model is validated, the eviction model is refuted.**

## Method

The same generated sessions drive both sides, turn by turn, with real `sleep`s for the
pauses so vLLM's scheduler sees the same idle gaps the simulator models. The KV pool is
read from the server's startup log (13663 blocks) and handed to the simulator, so neither
side guesses. Prompts are built as raw **token ids**, structured to mirror the
simulator's content model exactly: one shared system prefix across all sessions, then a
session-private tail that grows turn by turn.

Sessions are truncated at 30000 prompt tokens because Qwen2.5-3B stops at 32768, and the
truncation is applied to both sides, so the comparison stays exact at the cost of not
covering the longest contexts.

## Result

| pressure | metric | vLLM | simulator | difference |
|---|---|---|---|---|
| 0.64 | prefix-cache hit rate | 0.9010 | 0.9151 | **+1.4 pp** |
| 0.64 | wall clock | 652 s | 639 s | **−2%** |
| **1.02** | prefix-cache hit rate | **0.5501** | **0.8041** | **+25.4 pp (+46%)** |
| **1.02** | wall clock | 831 s | 836 s | **+1%** |

**The timing model is sound.** End-to-end makespan agrees within 2% at both pressures,
from four fitted constants and a hand-written scheduler. Every cost and latency figure in
this project rests on that, and it survives.

**The eviction model is not.** At the pressure where this project's claims actually live,
the simulator retains far more cache than vLLM does.

## Decomposition: the simulator under-evicts by 3.2x

Running the simulator on the identical trace with a pool so large that nothing can ever
be evicted gives the ceiling — the hit rate a perfect cache would achieve on this
workload:

| | hit rate | lost to eviction |
|---|---|---|
| no eviction possible | 0.9179 | — |
| simulator, real pool | 0.8041 | 11.4 pp |
| vLLM, same pool, same trace | 0.5501 | **36.8 pp** |

**vLLM loses 3.2x as much to eviction as the simulator does, at the same pool size on
the same workload.** This is not a metric-definition artefact: the block-versus-token
counting difference is worth about 1.4 pp, which is what the low-pressure run measures,
and it cannot account for 25.

## What this does to the project's claims

**Unaffected:** anything that depends on the timing model. Cost conversion, the
wall-clock versus GPU-time billing divergence, makespan, and the arithmetic bound on what
a retention policy can touch at long pauses.

**Affected, and the direction is known:** every absolute hit rate under pressure is
optimistic, and the **pressure axis is shifted**. The simulator at nominal pressure 1.02
behaves like a cache under considerably less stress than vLLM at the same nominal
pressure. Read against EXP02's curve, the simulator's peak-headroom point (pressure 0.84)
probably corresponds to a *lower* real pressure than 0.84.

**Unknown, and this is the uncomfortable part:** whether the *policy comparison* survives.
All arms run inside the same simulator, so a systematic under-eviction affects all of
them and the ranking may well hold. But EXP01 and EXP02 both show the headroom is
strongly non-monotone in pressure — it peaks near 0.85 and collapses by 1.6 — so a shift
along that axis moves the headroom, and possibly the termination share with it. **Until
the eviction model is fixed, headroom figures should be quoted in simulator pressure
units, not as predictions about a real deployment.**

## Candidate causes, as hypotheses

Not diagnosed. Listed so the next session starts from a shortlist rather than from
scratch, roughly in order of how much of the 25 pp each could plausibly explain.

1. **Whole-prompt admission.** `sim/engine.py` admits a request only if its entire prompt
   fits in the pool. That serialises admission under pressure and keeps the concurrent
   footprint smaller than vLLM's, which admits with chunked prefill and grows into the
   space. A smaller live footprint means less eviction — exactly the observed direction.
   This is already documented as a simplification in `sim/cache.py`; it may be the whole
   story.
2. **Finished-request blocks.** The simulator keeps them in an LRU pool until something
   needs the space. If vLLM frees or deprioritises them more eagerly, it would evict the
   very blocks an agent workload returns to.
3. **Reserved capacity.** vLLM may hold back blocks for `max_num_batched_tokens` or for
   CUDA-graph padding that the 13663 figure does not reflect, so its *usable* pool is
   smaller than the number handed to the simulator.
4. **Fragmentation.** The simulator's pool is exact; a real allocator is not.

## The methodological point worth keeping

The first run of this validation reported an 8.1 pp disagreement in the other direction,
and it was entirely an artefact of the harness: prompts were built as `"word " * n`, which
made all forty sessions byte-identical and mutual prefixes of one another. vLLM served
every session from every other session's cache and reported 99.6%.

What caught it was not inspection but a decomposition: **give the simulator an unlimited
pool.** The hit rate did not move, which proved the disagreement had nothing to do with
eviction and had to live in the content model or the accounting. One number cut the
search space in half and pointed straight at the harness.

**Before attributing a disagreement to the system under test, prove the measurement
apparatus is not the cause.** The same decomposition is what makes the 25 pp finding
above trustworthy, because this time the ceiling *did* move.

## What is not established here

- Two pressures, one seed, one model, one pool size. The 3.2x eviction ratio is a single
  measurement, not a curve.
- The longest contexts (30k–50k tokens) are excluded by the context-window truncation,
  and those are exactly the contexts where eviction pressure is highest.
- vLLM's `prefix_cache_queries_total` counts blocks; the simulator counts prompt tokens.
  The low-pressure run bounds that difference at about 1.4 pp, but only in a regime where
  eviction is inactive.
- No attempt was made to fix the eviction model. That is the next piece of work, and it
  should be driven by testing the hypotheses above rather than by tuning until the
  numbers match — a simulator tuned to reproduce one measurement is not validated by it.
