# EXP01 findings — is predictive KV retention worth building?

**Date:** 2026-08-01 (rerun on measured constants) · **Platform:** CPU simulation
**Data:** `results/v2_exp01_seeds15/` — 15 seeds, concurrency 8–18. Intervals are 95%
paired bootstrap over seeds (5000 resamples, seeded, reproducible). All arms run on
byte-identical workloads, so every comparison is paired.

**Everything below was rerun after calibration** against vLLM 0.26 + Qwen2.5-3B on the
RTX 5080. The KV pool turned out to be 12868 blocks, not the 16000 that was derived, so
the same concurrency now sits at 24% higher memory pressure and the grid moved down to
8–18 (pressure 0.67–1.51). The pre-calibration numbers are in `results/exp01_seeds15/`
and should not be quoted. Timing constants are now MEASURED — see
[calibration.md](calibration.md) — but the workload distributions are still INVENTED,
so ringgit magnitudes remain provisional.


> **Correction (2026-08-01, from [EXP04](exp04_findings.md)):** the `belady` arm is
> **not** an upper bound on retention policies, and wording to that effect elsewhere in
> this document is wrong. It ranks by *next* use only, which is myopic, and Belady's
> optimality assumes a fixed offline reference stream -- here eviction changes when work
> is recomputed, so the stream is not fixed. A simpler oracle that ranks by pause length
> alone (`belady_pause`) beats it by 6-42%. Everything below that compares an arm
> against **LRU** stands unchanged; only the phrase "upper bound" has to become "a
> strong oracle reference", and headroom shares can legitimately exceed 100%.

---

## The decision rule, fixed before the numbers were seen

| outcome | action |
|---|---|
| headroom < 10% | drop the direction |
| a tuned constant TTL captures > 70% of headroom | prediction is not the interesting variable |
| headroom > 30% and constant captures < 50% | premise survives, build a predictor |

## Verdict: the premise survives, but not for the reason the proposal assumed

Peak headroom (the `belady` oracle reference vs LRU, on prompt tokens recomputed) is **18.3%, 95% CI
[15.9, 21.0]**, at concurrency 10 — which is memory pressure 0.84, exactly where
[EXP02](exp02_findings.md) independently puts the peak (18.6% there, at 15 seeds). A tuned constant TTL captures
**exactly 0%** of it, at every TTL value, bit-for-bit. So prediction is the only route
to the headroom — but a large share of it is not the quantity the proposal planned to
predict.

Calibration moved this number: the pre-calibration estimate was 22.4% at concurrency 14.
The peak is real and in the same place on the *pressure* axis; what was wrong was the
mapping from pressure to concurrency, because the pool size was wrong.

---

## Four findings

### 1. A uniform TTL is *identical* to LRU. Not similar — identical.

Bit-identical output at every TTL from 1 s to 1e9 s, at every concurrency
(`tests/test_invariants.py::test_uniform_ttl_is_exactly_lru`).

The reason is structural: in a demand-evicted pool, a uniform TTL gives every block the
same protection window, so expiry order equals release order equals LRU order, and the
eviction *sequence* never changes. Freeing a block earlier than the moment something
needs it changes nothing.

**Consequence: all of a TTL policy's value comes from its variance across sessions.** A
constant TTL is not a baseline that competes with prediction; it is LRU wearing a
parameter. This closes off the cheapest way the project could have died.

### 2. Perfect information through the wrong mechanism is *worse than nothing*.

Using the true pause duration as a TTL — the obvious reading of "add a TTL to KV cache" —
loses to plain LRU across the whole pressured band:

| concurrency | 8 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|
| pressure | 0.67 | 0.84 | 1.01 | 1.17 | 1.34 | 1.51 |
| LRU, M tokens recomputed | 7.01 | 12.73 | 24.03 | 38.83 | 51.79 | 60.30 |
| true pause as TTL | *6.52* | **13.67** | **27.87** | **43.69** | **56.39** | **62.51** |
| Belady (same information, right mechanism) | 6.29 | 10.40 | 20.62 | 35.16 | 48.84 | 58.78 |

Worse than the incumbent everywhere at pressure >= 0.84, by up to 16% (concurrency 12).

It protects longest exactly the sessions that return latest. Under scarcity the correct
use of a pause estimate is as an eviction *priority* (evict furthest-future first), not
as a protection *duration*. Information and mechanism are separate axes and the
literature's TTL framing picks the wrong one.

**One qualification the recalibrated run added:** at concurrency 8 (pressure 0.67) the
TTL mechanism is *better* than LRU, not worse — 6.52M against 7.01M. Below saturation
there is room to spare, so protecting anything at all helps and protecting the
long-pause sessions costs nothing, because nothing else needs the space. The mechanism
is harmful only under scarcity. That is where the interesting regime is, but the claim
has to carry the boundary rather than be stated as a universal.

### 3. Termination is most of the prize at low pressure and a minority at high pressure.

An oracle that knows *only* whether a session is over, and otherwise falls back to LRU.
15 seeds, 95% bootstrap CI on the ratio of means:

| concurrency | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|
| pressure | 0.67 | 0.84 | 1.01 | 1.17 | 1.34 |
| total headroom % | 10.3 [8.1, 12.4] | **18.3 [15.9, 21.0]** | 14.2 [12.1, 16.4] | 9.4 [7.3, 11.9] | 5.7 [4.4, 7.1] |
| share from termination % | **87.9 [62.6, 105]** | **70.9 [55.1, 88.1]** | 53.1 [36.0, 68.3] | 35.9 [25.1, 45.0] | 22.9 [-1.4, 43.7] |

The decline is monotone and the endpoints do not overlap, so this is a real trend, not
noise. It also has a mechanical explanation: at low pressure the only thing worth doing
is evicting dead sessions, and there are just enough of them to matter; at high pressure
every live session is competing and the ordering *among* them is what decides the
outcome, which is precisely what termination information cannot tell you.

**Consequence for week 2: predict termination first.** It is binary, strongly signalled
(the model stops emitting tool calls), wrong at most once per session, and it is worth
71% of the available gain at the headroom peak, which is where a 16 GB card running
this workload actually sits.
Pause-length regression is the harder half and pays only under heavier pressure.

Note the estimator: this is a ratio of means with a bootstrap interval over seeds, not
the mean of per-seed ratios. The latter has an unstable denominator, and on an earlier
3-seed run it reported 91% where the correct estimate was 82%. The two are not
interchangeable and the per-seed version overstated its own precision.

### 4. A cache win is only a cost win when the GPU is saturated.

> **The pause sweep that used to live here has moved to [EXP03](exp03_findings.md)**,
> which redid it at 15 seeds and split it by billing model. The dramatic version of this
> finding -- cost saving collapsing to ~0 at long pauses -- turned out to be specific to
> wall-clock billing, and to be arithmetic rather than a measurement: at a 30 s median
> pause, tool execution is most of the wall clock and no retention policy can touch it.
> What survives, in both billing models, is the conversion loss below.

Belady's advantage on cost is roughly half its advantage on tokens, everywhere:

| concurrency | 8 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|
| headroom on tokens % | 10.3 | 18.3 | 14.2 | 9.4 | 5.7 | 2.5 |
| headroom on RM/1k calls % | 1.6 | 5.7 | 6.8 | 5.7 | 3.6 | 1.6 |

Conversion never exceeds about half, and at the token peak (concurrency 10) an 18.3% cut
in recomputed tokens is worth 5.7% of cost. Recomputation is not the only thing the GPU
is doing. [EXP03](exp03_findings.md) takes this apart by billing model.

### 5. TTFT responds only in a narrow window, and reporting it needs care.

15 seeds, headroom on TTFT p95:

| concurrency | 8 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|
| headroom % | 22.9 [17.1, 28.1] | 31.2 [22.7, 40.4] | 4.0 [0.2, 7.6] | 2.9 [-0.2, 5.9] | -0.9 [-3.8, 1.8] | -0.8 [-2.1, 0.3] |

TTFT is where the effect is largest — 31% at the peak — and it collapses to a null two
points later. The window is narrower than for tokens. Any "TTFT explodes" figure must
state where on this axis it was measured.

---

## The regime where any of this matters is narrow

Retention policy is irrelevant outside roughly pressure 0.7-1.3, which for this
pool/workload pair is 8-16 concurrent sessions:

- below that, the pool holds everything and every arm ties;
- above it, nothing survives a pause under any policy and every arm ties again
  (headroom 2.5% at concurrency 18; EXP02 measures 0.5% at pressure 2.0).

The real independent variable is **working set / pool capacity**, not concurrency and
not pause length — those are two handles on the same ratio. Every figure must report
that ratio, or the result does not transfer to another GPU. Reporting "TTFT explodes at
concurrency 32" without it would be unreproducible on any other card.

---

## What this changes about the two-week plan

- **Day 8's falsification came first and was worth it.** Two of the three planned arms
  turned out to be the same policy, and the oracle as originally specified was
  backwards. Both would have been discovered in week 2, on GPU time.
- **Reframe the predictor.** Target session termination first, pause length second, and
  report the split. At concurrency 12–14 — the regime a 16 GB card sits in — termination
  alone is worth 67–82% of the whole prize.
- **Lead with cost, not hit rate.** Finding 4 is the one no existing paper states.
- **Add the working-set ratio to every axis** before running anything on GPU.

## What is not yet established

- Every magnitude depends on uncalibrated timing constants and an invented pause
  distribution. Finding 1 is structural and immune; findings 2–4 are directional but
  their numbers are not quotable yet.
- Findings 1–3 and 5 rest on 15 seeds with bootstrap intervals. Finding 4's pause
  sweep here is 3 seeds; [EXP03](exp03_findings.md) redid it at 15 and split it by
  billing model.
- Every number in this document is closed-loop. [EXP03](exp03_findings.md) shows the
  same workload has an unstable open-loop regime that the closed loop cannot reach,
  so these results are all from the stable side of a bistable system.
- Results are reported against concurrency, which is a property of this GPU and model.
  [EXP02](exp02_findings.md) re-states them on the transferable axis.
- Only closed-loop arrivals were run. The open-loop (Poisson) mode exists but is
  untested, and it is the mode where latency and cost decouple most sharply.
- No GPU validation of any kind. Nothing here says the simulator resembles vLLM.
