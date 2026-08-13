# EXP02 findings — does the result belong to the hardware or to the pressure?

**Date:** 2026-08-01 (rerun on measured constants) · **Platform:** CPU simulation
**Data:** `results/v2_exp02_seeds15/` — 1080 runs, **15 seeds**, 8 target pressures x 3
conditions x 3 policy arms. Intervals are 95% paired bootstrap over seeds. The 10-seed
run is in `results/v2_exp02/` and the pre-calibration run in `results/exp02/`; neither
should be quoted. Session count scaled to 10 per concurrency slot so that every condition
runs the same number of waves and transients cannot differ between them.


> **Correction (2026-08-01, from [EXP04](exp04_findings.md)):** the `belady` arm is
> **not** an upper bound on retention policies, and wording to that effect elsewhere in
> this document is wrong. It ranks by *next* use only, which is myopic, and Belady's
> optimality assumes a fixed offline reference stream -- here eviction changes when work
> is recomputed, so the stream is not fixed. A simpler oracle that ranks by pause length
> alone (`belady_pause`) beats it by 6-42%. Everything below that compares an arm
> against **LRU** stands unchanged; only the phrase "upper bound" has to become "a
> strong oracle reference", and headroom shares can legitimately exceed 100%.

## The question

EXP01 said "the cache collapses above about 16 concurrent sessions". That sentence is
unreproducible: 16 is a property of one card running one model. The transferable claim
would be that the phenomenon depends on

```
pressure = live sessions x mean context blocks / KV pool blocks
```

with concurrency and pool size as two handles on the same quantity. This experiment
reaches each target pressure three different ways and measures whether they agree:

| condition | pool | sessions |
|---|---|---|
| `vary_concurrency` | fixed at 16000 blocks (RTX 5080 16 GB) | scaled |
| `vary_pool` | scaled | fixed at 16 |
| `bigger_gpu` | fixed at 40000 blocks (2.5x the KV capacity) | scaled |

## Verdict: pressure is necessary but not sufficient — session count is a second axis

Disagreement between the three conditions at equal pressure:

At 10 seeds the disagreement between conditions looked like scatter that more seeds
would resolve. It is not. At 15 seeds the intervals barely narrowed — the upper bound at
pressure 0.85 moved only from 11.5 pp to 9.5 pp — and breaking the spread down by
condition shows why: **it is not scatter, it is one condition sitting systematically
below the other two.**

| pressure | `bigger_gpu` (26–48 sessions) | `vary_concurrency` (10–19) | `vary_pool` (16) |
|---|---|---|---|
| 0.70 | **4.4 [3.3, 5.7]** | 9.6 [5.9, 13.2] | 9.2 [6.5, 12.1] |
| 0.85 | **15.4 [12.6, 18.4]** | 20.1 [17.9, 22.5] | 20.4 [17.2, 23.7] |
| 1.00 | 13.4 [11.1, 15.8] | 11.7 [9.1, 14.7] | 15.7 [12.7, 19.0] |
| 1.15 | **6.9 [5.2, 9.0]** | 9.2 [7.5, 11.3] | 8.9 [7.1, 10.9] |
| 1.30 | **2.1 [1.5, 2.8]** | 5.4 [4.0, 6.8] | 6.2 [4.9, 7.8] |

Two readings fall out immediately:

**Pressure transfers across pool size.** `vary_concurrency` (7–30 sessions, 16k-block
pool) and `vary_pool` (16 sessions, pool scaled from 8.6k to 34.5k) agree to within
about 1 pp at every pressure — 9.6 vs 9.2, 20.1 vs 20.4, 9.2 vs 8.9, 5.4 vs 6.2. A
four-fold change in pool size, holding the session count near 16, changes nothing once
pressure is matched. That is the part of the original claim that is solid.

**Pressure does not transfer across session count.** `bigger_gpu` reaches the same
pressure with 26–48 sessions on a 40k-block pool, and its headroom is systematically
lower, with intervals disjoint from the other two at pressures 0.70, 0.85 and 1.30.
At pressure 1.30 it is 2.1% against 5.4–6.2%: less than half.

The mechanism is the same one the hit-rate spread already showed, and it turns out to
apply to the headroom too, at *every* pressure rather than only above saturation:
**splitting the same deficit across more sessions behaves like higher pressure.** With
48 sessions competing, even a perfect eviction order cannot keep enough of any one
context intact, so the configuration is already close to the regime where every policy
ties — while 19 sessions at the same nominal pressure still has room for ordering to
matter.

**So the pressure ratio is necessary but not sufficient.** The transferable statement is
narrower than the earlier drafts of this document claimed, and needs both terms:

> at a given memory pressure *and* a comparable number of concurrent sessions, the
> headroom transfers across hardware.

A result measured at 16 sessions does not carry to a machine running 48 sessions at the
same pressure, even though the pressure ratio was constructed precisely to make those
two comparable.

## The second useful result: the peak is at working set ≈ pool

Headroom is not monotone in pressure. Averaged across the three conditions it is near
zero at 0.5, peaks at **18.6%** at pressure 0.85, and decays back to 0.9% by pressure
1.6. That average is over conditions that differ systematically — at pressure 0.85 the
individual values are 15.4%, 20.1% and 20.4% — so it is a summary of the shape, not a
figure to quote on its own. The per-condition numbers are the ones with meaning.

That shape has a clean reading. Below saturation there is nothing to evict, so no policy
can beat any other. Far above it, nothing survives a pause under any policy, so again no
policy can beat any other. Predictive retention only pays in the band around working
set ≈ pool capacity, and **that band is where a 16 GB card running a 3B model with
agent-length contexts actually sits** — with the measured pool of 12868 blocks,
concurrency 8–16 maps to pressure 0.67–1.34.

This is worth stating explicitly in the thesis because it also bounds the claim: a
deployment far from that band gains nothing from any of this work, and saying so is
what makes the positive result credible.

## What this changes

- Every EXP01 figure should be re-plotted against pressure, not concurrency. The
  concurrency axis is fine as a secondary tick label.
- Any future GPU measurement must report `kv_pool_blocks` (read from the vLLM startup
  log) and the mean context size, or the number cannot be placed on this axis.
- Claims made above pressure ~1.1 must also state the concurrency, because at those
  pressures the two are not interchangeable.

## What is not established

- Going from 10 to 15 seeds narrowed the peak interval by only about 2 pp, which is the
  evidence that the residual disagreement is a real effect rather than noise. More seeds
  will keep shrinking it as sqrt(n) and will not make the `bigger_gpu` condition line up.
- The session-count effect is established as a direction, not as a functional form. How
  headroom depends on session count at fixed pressure — whether it is a second
  multiplicative term, a saturating one, or something else — was not measured, and would
  need a grid over both axes rather than three points on one.
- GPU utilisation is not held constant across the three conditions (0.856 to 0.998 at
  the same pressure), because more sessions means more offered work. That does not
  affect the token metrics used for the collapse test, but it does mean the cost and
  TTFT columns of this experiment are not a clean comparison.
- `bigger_gpu` models a larger KV pool but keeps every timing constant unchanged. A real
  96 GB card is also faster, so this condition isolates capacity, not hardware. It is
  also the condition that runs the most sessions, so "bigger pool" and "more sessions"
  are confounded within it — the two could be separated with a fourth condition holding
  the pool at 40k while capping sessions at 16, and that was not run.
