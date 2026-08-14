# Validation against vLLM — the timing model holds, and one comparison was invalid

**Date:** 2026-08-14, revised 2026-08-15 · **Data:** `results/validate/` (pressure 0.64),
`results/validate_pressured/` (pressure 1.02), `results/validate_diagnosis/` ·
**Harness:** `bench/validate_vs_vllm.py`, re-analysis in `bench/diagnose_validation.py`

Every number in this project comes from a simulator whose timing constants were fitted
to vLLM but whose *behaviour* had never been compared to it. This closes that gap.

> **Correction, 2026-08-15.** The first version of this document reported that the
> simulator "under-evicts by 3.2x" at pressure 1.02, based on a 25 pp hit-rate gap. That
> conclusion was wrong. It compared two different quantities: vLLM's `/metrics` counters
> are incremented **once per scheduling** and count tokens, so a preempted-and-resumed
> request is counted twice in both numerator and denominator, while the simulator's
> `token_hit_rate` counts each prompt once. The section below is the corrected analysis.
> The retraction is kept rather than removed because the check that caught it is the
> reusable part.

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
| 0.64 | wall clock | 652 s | 639 s | **−2%** |
| 1.02 | wall clock | 831 s | 836 s | **+1%** |
| 0.64 | prefix-cache hit rate | 0.9010 | 0.9151 | **+1.4 pp** |
| 1.02 | prefix-cache hit rate | *not measured* | 0.8041 | — |

**The timing model is validated.** End-to-end makespan agrees within 2% at both
pressures, from four fitted constants and a hand-written scheduler. Every cost and
latency figure in this project rests on that, and it survives at both pressures —
including the one where the cache comparison does not.

**The hit-rate model is validated at pressure 0.64 only.** There, vLLM scheduled every
prompt exactly once (9,854,329 queried tokens against 9,854,329 prompt tokens sent, a
ratio of 1.000), so its counters *are* a per-request hit rate and the comparison is
exact. 1.4 pp is a good agreement for a model that was never fitted to it.

## Why pressure 1.02 is not a measurement

At pressure 1.02, vLLM queried 26,013,279 tokens against 15,360,566 prompt tokens sent —
**1.694x**. Every prompt was scheduled an average of 1.69 times, because requests were
preempted and re-queried the cache on resumption. The counters do not distinguish a first
query from a re-query, and both hits and queries are inflated.

That this is inflation and not a real hit rate is forced by arithmetic, not assumed. With
an infinite pool the simulator reaches 0.9179 on this trace, so at most
15,360,566 × 0.9179 = **14,099,072** tokens could ever be hit under a per-request
accounting. vLLM reported **14,310,464** hits — 1.01x more than the ceiling. A ratio whose
numerator exceeds the maximum possible is not the quantity it looks like.

Decomposing it, with `β` the true per-request hit rate and `α` the rate at which resumed
requests re-hit:

| assumed α (resumed-request hit rate) | implied β (true per-request hit rate) |
|---|---|
| 0.00 | 0.9316 |
| 0.50 | 0.5849 |
| 0.9179 (the ceiling) | 0.2951 |
| 1.00 | 0.2381 |

`β` is somewhere in **[0.238, 0.932]** and the data cannot narrow it further. The
simulator's 0.8041 sits inside that interval. The honest statement is that **this run does
not measure whether the simulator's eviction model is right at pressure 1.02**, in either
direction. The physically plausible end is the low one — blocks freed by preemption stay
cached in vLLM's free queue and a promptly resumed request should re-hit most of them —
which would mean the simulator is optimistic, but that is an argument, not a measurement.

## What *is* refuted: the admission and preemption model

One quantity in that run needs no denominator at all, and it disagrees completely:

| | preemptions |
|---|---|
| simulator, 1122 calls at pressure 1.02 | **0** |
| vLLM, same trace, same pool | ~69% of prompt tokens re-scheduled |

The simulator never preempted once. This follows directly from a documented
simplification: `sim/engine.py` admits a request **only if its whole prompt fits**, so it
can never over-commit and never has to take memory back. vLLM admits optimistically and
grows into the space with chunked prefill, then preempts when it runs out.

So the simulator runs a smaller, more conservative live footprint than vLLM does at the
same nominal pressure. Two consequences, one reassuring and one not:

- **Reassuring:** makespan still agreed within 1%, so vLLM's preemptions are cheap. That
  fits the mechanism — a preempted request's blocks go back to the free queue still
  cached, and resumption largely re-hits them rather than recomputing.
- **Not:** "pressure" does not mean the same thing on the two systems. A simulator run at
  nominal pressure 1.02 is doing something a real server would only do at some lower
  pressure, because the real server has admitted more work into the same pool.

## What this does to the project's claims

**Unaffected:** everything resting on the timing model — cost conversion, the wall-clock
versus GPU-time billing divergence, makespan, and the arithmetic bound on what a retention
policy can touch at long pauses. Validated at both pressures.

**Unaffected:** hit rate at moderate pressure, validated to 1.4 pp.

**Open:** absolute hit rates at pressure near and above 1.0, and therefore the position of
EXP01/EXP02's peak-headroom point on the pressure axis. The admission difference gives the
direction — the simulator's pressure axis is *softer* than a real server's — but not the
size. Headroom figures should be quoted in **simulator pressure units** until this is
closed.

**Probably unaffected, and worth saying why:** the policy *ranking*. All arms run inside
the same admission model, and the mechanism that differs is policy-independent — whole-
prompt admission and preemption are identical code for `lru`, `belady` and `predict`
alike. That is an argument for the ranking being robust, not evidence of it.

## How to close it

The fix is one more local run, not a redesign. vLLM's OpenAI-compatible responses carry
`usage.prompt_tokens_details.cached_tokens` — a **per-request** count, exactly the
simulator's definition, immune to the re-query inflation. The harness now records it,
reports it as the headline comparison, and prints a warning whenever the `/metrics`
inflation exceeds 1.0. Re-running the pressure-1.02 case is enough to turn the row above
from *not measured* into a number.

The second thing worth measuring at the same time is preemption: vLLM's
`vllm:num_preemptions_total` against the simulator's `n_preemptions`, which is now in the
summary. That comparison needs no denominator either.

## Candidate causes, if the corrected comparison still disagrees

Ordered by how much of a gap each could plausibly explain, with the leading one now
promoted from hypothesis to confirmed mechanism.

1. **Whole-prompt admission — confirmed to differ, size unknown.** See above.
2. **Finished-request blocks.** The simulator keeps them in an LRU pool until something
   needs the space. If vLLM frees or deprioritises them more eagerly, it would evict the
   very blocks an agent workload returns to.
3. **Reserved capacity.** vLLM may hold back blocks for `max_num_batched_tokens` or for
   CUDA-graph padding that the 13663 figure does not reflect, so its *usable* pool is
   smaller than the number handed to the simulator.
4. **Fragmentation.** The simulator's pool is exact; a real allocator is not.

## The methodological point worth keeping

This validation produced two false findings before producing a true one, and both were
caught the same way — by asking what the measurement apparatus would report if the system
under test were perfect.

1. The first run reported an 8.1 pp disagreement caused entirely by the harness: prompts
   were built as `"word " * n`, which made all forty sessions byte-identical and mutual
   prefixes of one another, so vLLM served every session from every other session's cache
   and reported 99.6%. **Giving the simulator an unlimited pool** did not move its hit
   rate, which proved eviction was not involved and pointed straight at the harness.
2. The second run reported a 25 pp disagreement and a 3.2x eviction ratio, and I wrote it
   up as a refutation of the eviction model. It was an accounting difference. **Comparing
   the query count to the number of prompt tokens sent** — a ratio that must be 1.000 if
   the counters mean what they appear to mean — settles it in one line, and the same
   unlimited-pool ceiling shows the reported hit count exceeding the maximum possible.

Both checks are cheap, both are denominator-level sanity rather than domain insight, and
in both cases the wrong answer was more interesting than the right one, which is exactly
why it got written up. `tests/test_invariants.py::test_query_accounting_matches_vllms_counters`
now pins the relationship between the two definitions so the second mistake cannot recur
silently.

## What is not established here

- Two pressures, one seed, one model, one pool size.
- The hit-rate comparison at pressure 1.02 has not been made at all; only its bound is
  known.
- The longest contexts (30k–50k tokens) are excluded by the context-window truncation,
  and those are exactly the contexts where eviction pressure is highest.
- No attempt was made to change the simulator to match vLLM. The admission difference is
  now documented and its direction is known; closing it should be driven by the corrected
  measurement, not by tuning until the numbers agree — a simulator tuned to reproduce one
  measurement is not validated by it.
