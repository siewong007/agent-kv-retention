# EXP03 findings — was the pause sweep confounded, and which billing model was it?

**Date:** 2026-08-01 (rerun on measured constants) · **Platform:** CPU simulation
**Data:** `results/v2_exp03/` — 540 runs, 15 seeds, 6 pause points x 2 arrival regimes x
3 policy arms, plus `calibration.json` recording how each open-loop arrival rate was
chosen. The pre-calibration run is in `results/exp03/`; do not quote it.

**Caveat on where this sits.** The closed-loop arm is pinned at concurrency 16, which
with the measured pool of 12868 blocks is pressure 1.35 -- above the headroom peak that
[EXP01](exp01_findings.md) and [EXP02](exp02_findings.md) put at 0.84. Absolute headroom
here is therefore smaller than at the peak. The comparison this experiment is *for* --
wall-clock versus GPU-time billing, and closed versus open loop -- is unaffected, because
both sides of each comparison sit at the same pressure.


> **Correction (2026-08-01, from [EXP04](exp04_findings.md)):** the `belady` arm is
> **not** an upper bound on retention policies, and wording to that effect elsewhere in
> this document is wrong. It ranks by *next* use only, which is myopic, and Belady's
> optimality assumes a fixed offline reference stream -- here eviction changes when work
> is recomputed, so the stream is not fixed. A simpler oracle that ranks by pause length
> alone (`belady_pause`) beats it by 6-42%. Everything below that compares an arm
> against **LRU** stands unchanged; only the phrase "upper bound" has to become "a
> strong oracle reference", and headroom shares can legitimately exceed 100%.

## The question

EXP01's pause sweep ran closed-loop at fixed concurrency, where a longer tool pause also
means a lower arrival rate. GPU utilisation fell from 100% to 46% across it. Finding 4 —
"a cache win stops being a cost win when the GPU idles" — is exactly the claim that
confound could have manufactured.

Two things had to be fixed before the question could even be asked.

**Utilisation is not a valid control variable.** A cache miss creates GPU work, so
pinning "GPU busy fraction" partly pins the waste being measured: a thrashing
configuration reaches 99% utilisation *because* it is recomputing. The first version of
this experiment calibrated the arrival rate to a target utilisation, which would have
built the answer into the design. It now calibrates to a target *live session count*,
which sets memory pressure directly and is not inflated by misses.

**One cost number was not enough.** Under open-loop arrivals the makespan is pinned by
the arrival schedule, so a wall-clock cost metric cannot move no matter what the policy
does. The simulator now reports two:

| metric | charges | corresponds to |
|---|---|---|
| `rm_per_1k_calls` | wall clock | a reserved box — how the Sunway HPC session is billed |
| `rm_gputime_per_1k_calls` | seconds the GPU actually worked | shared or autoscaled capacity |

## Result 1: Finding 4 was half billing model, half real

Closed loop, 15 seeds, concurrency 16 (pressure 1.27-1.36 throughout, so pressure is
*not* what varies here):

| median pause | GPU busy | LRU hit | tokens headroom | RM wall | RM gpu-time | conversion |
|---|---|---|---|---|---|---|
| 0.5 s | 99.9% | 0.184 | 1.1% | 0.5% | 0.5% | 45% |
| 1 s | 99.8% | 0.221 | 2.6% | 1.7% | 1.7% | 65% |
| 2 s | 99.4% | 0.297 | 5.7% | 3.6% | 3.6% | 63% |
| 5 s | 95.9% | 0.484 | 19.8% | 9.6% | 10.4% | 53% |
| 10 s | 83.5% | 0.647 | 27.9% | 5.2% | 8.8% | 32% |
| 30 s | 45.7% | 0.742 | 35.2% | **0.9%** | **7.2%** | 20% |

**The dramatic version of Finding 4 was a property of wall-clock billing.** At a 30 s
median pause, tool execution is most of a session's wall clock and no retention policy
can touch it, so the reserved-box cost saving goes to 0.9%. That is arithmetic, not a
measurement, and no experimental design removes it. Under utilisation billing the same
runs save 7.2% -- eight times more, from identical simulator runs. The billing model is
not a presentation choice; it changes the conclusion.

**The robust core survives in both.** Conversion efficiency — how much of a token saving
becomes a cost saving — falls from about 60% to 20% as pauses lengthen, under *either*
billing model. A 35% cut in recomputed tokens is worth 7% of cost. That gap is the
finding, and it is not a billing artefact: when the cache is already warm, recomputation
is a small share of GPU seconds and cutting it further has little leverage.

The practical statement for the thesis is therefore conditional, and should be written
that way: *on a reserved instance with long tool pauses, KV retention is nearly
worthless as a cost lever; on shared capacity it is worth single-digit percent; in
neither case does it deliver the reduction that a hit-rate figure implies.*

## Result 2: open-loop agent workloads at high load are metastable

The open-loop arm was supposed to reproduce the closed-loop pressure with Poisson
arrivals. It could not, and the reason is worth more than the comparison would have been.

At the calibrated rate, the seeds split into two populations:

| median pause | median live | range across seeds | seeds that ran away |
|---|---|---|---|
| 0.5 s | 51.3 | 4.6 – 89.3 | 67% |
| 1 s | 26.1 | 3.9 – 81.5 | 47% |
| 2 s | 57.0 | 5.6 – 82.1 | 73% |
| 5 s | 43.9 | 6.3 – 62.1 | 60% |
| 10 s | 37.3 | 8.6 – 50.0 | 53% |
| 30 s | 22.0 | 13.0 – 29.6 | **0%** |

At the same offered load, one arrival realisation settles at ~5 live sessions while
another runs away to ~89. There is no stable operating point in between: during
calibration, raising the arrival rate by 25% moved the mean live count from 7.3 to 94.6.

The mechanism is a positive feedback loop specific to cached agent workloads. More live
sessions means more memory pressure, which means a lower hit rate, which means more
recomputation, which means slower service, which means sessions stay live longer. Above
a threshold it does not recover.

**Consequences.**

- The closed loop is not a neutral simplification. By replacing a finished session
  immediately it caps the feedback, which is why every closed-loop row here has a
  0% runaway rate and a live-session range of ±1. EXP01's numbers are all from the
  stable side of a system that has an unstable side.
- Only the 30 s pause point is a valid open-vs-closed comparison, and there the two
  agree: tokens headroom 34.3% open vs 35.2% closed, and GPU-time cost headroom 9.9%
  open vs 7.2% closed. That is the one piece of evidence that the EXP01 conclusions are
  not closed-loop artefacts, and it is a single point.
- A "cache hit rate under load" figure for an open-loop agent workload near saturation
  is reporting a mixture of two regimes. Its mean is not a central tendency of anything.

## What is not established

- The single-seed calibration is invalid near saturation, and no amount of refinement
  fixes it, because the quantity being calibrated has no stable value. Rates in
  `calibration.json` should be read as "a rate near the stability boundary", not as
  "the rate that gives 16 live sessions".
- Whether the metastability is a property of agent workloads or of this simulator's
  admission and preemption rules is untested. Recompute-preemption in particular
  amplifies the feedback, and a swap-based preemption might not.
- The open-vs-closed agreement rests on one pause point.
- Bootstrap intervals were not computed for this experiment.
