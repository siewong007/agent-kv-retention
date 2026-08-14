# Validation against vLLM — timing and eviction both hold; admission does not

**Date:** 2026-08-14, revised 2026-08-15 · **Data:** `results/validate/` (pressure 0.64),
`results/validate_pressured/` (pressure 1.02, invalid — see below),
`results/validate_matched_admission/` (pressure 1.08), `results/validate_diagnosis/` ·
**Harness:** `bench/validate_vs_vllm.py`, re-analysis in `bench/diagnose_validation.py`

Every number in this project comes from a simulator whose timing constants were fitted
to vLLM but whose *behaviour* had never been compared to it. This closes that gap.

> **Correction, 2026-08-15.** The first version of this document reported that the
> simulator "under-evicts by 3.2x" at pressure 1.02, based on a 25 pp hit-rate gap. That
> conclusion was wrong. It compared two different quantities: vLLM's `/metrics` counters
> are incremented **once per scheduling** and count tokens, so a preempted-and-resumed
> request is counted twice in both numerator and denominator, while the simulator's
> `token_hit_rate` counts each prompt once. The measurement has since been redone with
> the two admission models matched, which removes the preemption and makes the counters
> comparable again: **the eviction model agrees to 1.4 pp above pressure 1.0.** The
> retraction is kept rather than removed because the check that caught it is the
> reusable part.

## Method

The same generated sessions drive both sides, turn by turn, with real `sleep`s for the
pauses so vLLM's scheduler sees the same idle gaps the simulator models. The KV pool is
read from the server's startup log for each run (13663 blocks at default settings,
12865 with `--max-num-seqs 8`) and handed to the simulator, so neither side guesses. Prompts are built as raw **token ids**, structured to mirror the
simulator's content model exactly: one shared system prefix across all sessions, then a
session-private tail that grows turn by turn.

Sessions are truncated at 30000 prompt tokens because Qwen2.5-3B stops at 32768, and the
truncation is applied to both sides, so the comparison stays exact at the cost of not
covering the longest contexts.

## Result

| pressure | admission | metric | vLLM | simulator | difference |
|---|---|---|---|---|---|
| 0.64 | default | wall clock | 652 s | 639 s | **−2%** |
| 1.02 | default | wall clock | 831 s | 836 s | **+1%** |
| 1.08 | matched | wall clock | 958 s | 972 s | **+1.5%** |
| 0.64 | default | prefix-cache hit rate | 0.9010 | 0.9151 | **+1.4 pp** |
| 1.02 | default | prefix-cache hit rate | *not measurable* | 0.8041 | — |
| **1.08** | **matched** | **prefix-cache hit rate** | **0.7517** | **0.7382** | **−1.4 pp** |

**The timing model is validated.** End-to-end makespan agrees within 2% in all three
runs, from four fitted constants and a hand-written scheduler. Every cost and latency
figure in this project rests on that, and it survives everywhere tested — including the
run where the cache comparison does not work.

**The hit-rate model is validated at both pressures where the comparison is valid.** At
0.64 and at 1.08 vLLM scheduled every prompt exactly once (inflation 1.0000, preemptions
0), so its counters *are* a per-request hit rate and the comparison is exact. The
agreement is 1.4 pp in both cases and **the sign flips** — the simulator is slightly
optimistic at 0.64 and slightly pessimistic at 1.08 — which is what a small definitional
difference looks like, not a bias. For a model whose eviction policy was never fitted to
anything, that is a good result, and it is the one that matters: pressure 1.08 is above
where this project's claims live.

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

## The measurement that closes it: match the admission models

The ambiguity above exists only because vLLM preempted. The fix is to remove the
preemption rather than to correct for it. vLLM 0.26 does not populate
`usage.prompt_tokens_details.cached_tokens` on `/v1/completions` (it comes back `None`),
so the per-request number is not available from a stock server; but capping
`--max-num-seqs` on **both** sides makes vLLM queue instead of over-committing, which is
exactly what the simulator's whole-prompt admission already does.

At `--max-num-seqs 8` on the server and `engine.max_num_seqs=8` in the simulator, 60
sessions at concurrency 16 against the same pool:

| | vLLM | simulator |
|---|---|---|
| query inflation | **1.0000** | **1.0000** |
| preemptions | **0** | **0** |
| prefix-cache hit rate | **0.7517** | **0.7382** |
| wall clock | 958 s | 972 s |

Inflation 1.0000 on both sides is the precondition, not a nicety: it says every prompt was
scheduled exactly once, so the two ratios are the same quantity. The eviction model agrees
to **1.4 pp at pressure 1.08**, and the simulator is on the *pessimistic* side this time.
Capping `max_num_seqs` also shrinks the KV pool, because more activation memory is
reserved (13663 to 12865 blocks), which is why this run sits at 1.08 rather than 1.02; the
pool was re-read from the server rather than carried over.

This isolates a mechanism. It is **not** vLLM as deployed -- a production server would use
its default admission and would preempt. What it establishes is narrower and sufficient:
given the same admission behaviour, the simulator's block pool, prefix lookup and LRU
eviction reproduce vLLM's to about a point and a half, above pressure 1.0.

## What remains refuted: the admission and preemption model

At default settings the two systems still differ, and one quantity in the pressure-1.02
run needs no denominator at all:

| | preemptions |
|---|---|
| simulator, 1122 calls at pressure 1.02 | **0** |
| vLLM, same trace, same pool, default admission | 69% of prompt tokens re-scheduled |

The simulator never preempted once. This follows directly from a documented
simplification: `sim/engine.py` admits a request **only if its whole prompt fits**, so it
can never over-commit and never has to take memory back. vLLM admits optimistically and
grows into the space with chunked prefill, then preempts when it runs out.

So at default settings the simulator runs a smaller, more conservative live footprint than
vLLM does at the same nominal pressure. Two consequences, one reassuring and one not:

- **Reassuring:** makespan still agreed within 1%, so vLLM's preemptions are cheap. That
  fits the mechanism -- a preempted request's blocks go back to the free queue still
  cached, and resumption largely re-hits them rather than recomputing.
- **Not:** "pressure" does not mean quite the same thing on the two systems at default
  settings. A real server admits more work into the same pool, so the mapping from
  simulator pressure to server pressure is not the identity. Its size is unmeasured.

## What this does to the project's claims

**Validated:** the timing model, in all three runs (2%, 1%, 1.5% on makespan). Cost
conversion, the wall-clock versus GPU-time billing divergence, and the arithmetic bound on
what a retention policy can touch at long pauses all rest on this.

**Validated:** the eviction model, to 1.4 pp at pressure 0.64 and 1.4 pp at pressure 1.08,
with the sign flipping between them. This is the one that was in doubt for a day, and it
holds. Absolute hit rates in EXP01-EXP05 are not systematically optimistic.

**Open, and now the only open item from this work:** the pressure axis itself. Matching
`max_num_seqs` validated the cache under matched admission; it did not measure how a
server's *default* admission shifts the mapping. EXP01/EXP02 headroom peaks near pressure
0.85 and collapses by 1.6, so where that peak sits on a real server's axis is worth
knowing. Quote headroom in **simulator pressure units**, and say so.

**Argued, not measured:** the policy *ranking*. All arms run inside the same admission
model, and the mechanism that differs is policy-independent -- whole-prompt admission and
preemption are identical code for `lru`, `belady` and `predict` alike. That is a reason to
expect the ranking to be robust, not evidence that it is.

## How to close the remaining item

Two local runs, no HPC time:

1. Sweep `--max-num-seqs` on both sides (say 6, 8, 12, default) at fixed workload and read
   off how the hit rate moves. That maps simulator pressure to server pressure directly,
   without needing the per-request counter.
2. Compare `vllm:num_preemptions_total` to the simulator's `n_preemptions` across that
   sweep. Both are denominator-free, so the admission difference becomes a curve instead
   of a single 0-versus-69% data point.

Giving `sim/engine.py` the opposite policy -- optimistic admission with chunked prefill and
recompute preemption -- is the larger fix, and it should only be attempted if step 1 shows
the mapping matters for the headroom claims.

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
   The redone measurement put the real disagreement at 1.4 pp, in the other direction.

Both checks are cheap, both are denominator-level sanity rather than domain insight, and
in both cases the wrong answer was more interesting than the right one, which is exactly
why it got written up. `tests/test_invariants.py::test_query_accounting_matches_vllms_counters`
now pins the relationship between the two definitions so the second mistake cannot recur
silently.

## What is not established here

- Three runs, one seed each, one model, one workload generator. The 1.4 pp agreements
  are single measurements without intervals; nothing here says how much they would
  move on another seed.
- The pressure-1.08 agreement holds under *matched* admission. Under vLLM's default
  admission the hit rate at that pressure is still not measured, and the pressure-1.02
  attempt bounds it only to [0.238, 0.932].
- The longest contexts (30k–50k tokens) are excluded by the context-window truncation,
  and those are exactly the contexts where eviction pressure is highest.
- No attempt was made to change the simulator to match vLLM. The admission difference is
  now documented and its direction is known; closing it should be driven by the corrected
  measurement, not by tuning until the numbers agree — a simulator tuned to reproduce one
  measurement is not validated by it.
