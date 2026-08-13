# EXP02 findings — does the result belong to the hardware or to the pressure?

**Date:** 2026-08-01 (rerun on measured constants) · **Platform:** CPU simulation
**Data:** `results/v2_exp02/` — 720 runs, 10 seeds, 8 target pressures x 3 conditions x
3 policy arms. The pre-calibration run is in `results/exp02/`; do not quote it. Session count scaled to 10 per concurrency slot so that every condition
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

## Verdict: the axis holds below saturation and breaks above it

Disagreement between the three conditions at equal pressure:

Disagreement between the three conditions, with 95% paired-bootstrap intervals over
seeds (whole seeds resampled, the max-minus-min spread recomputed on each draw — the
statistic has no closed-form standard error):

| target pressure | headroom mean | headroom spread [95% CI] | hit-rate spread [95% CI] |
|---|---|---|---|
| 0.50 | 1.2% | 0.7 pp [0.3, 1.1] | 0.00 [0.00, 0.00] |
| 0.70 | 9.0% | 5.9 pp [4.9, 7.2] | 0.00 [0.00, 0.00] |
| 0.85 | **17.4%** | 5.3 pp **[1.7, 11.5]** | 0.07 [0.07, 0.07] |
| 1.00 | 15.3% | 1.4 pp **[0.5, 7.3]** | 0.06 [0.04, 0.06] |
| 1.15 | 8.8% | 2.5 pp [1.0, 5.3] | 0.13 [0.07, 0.15] |
| 1.30 | 5.0% | 2.7 pp [2.1, 4.4] | 0.15 [0.12, 0.17] |
| 1.60 | 1.4% | 2.6 pp [1.7, 3.4] | 0.06 [0.03, 0.10] |
| 2.00 | 0.5% | 1.0 pp [0.5, 1.6] | 0.01 [0.00, 0.03] |

**The intervals change the reading, and they change it exactly where it matters.** An
earlier draft of this document said "the spread never exceeds 5.9 pp, and is 1.4-2.7 pp
through the whole pressured band", concluding that headroom transfers everywhere. That
was a claim about point estimates. At pressure 0.85 and 1.00 — the two points where the
headroom peaks and where the transfer claim is worth the most — the intervals reach
**11.5 pp and 7.3 pp**. With 10 seeds the data is compatible with disagreement roughly
twice as large as the point estimate suggests.

What survives:

**Headroom transfers, but not tightly enough to quote at the peak.** At pressures away
from the peak the upper bounds are 1.1-5.3 pp, which is small against headrooms of
5-9%. At the peak the upper bound is comparable to a third of the effect itself.
"Belady beats LRU by about 17% at pressure 0.85, on any of three hardware
configurations" is supportable; "and the three agree to within 5 pp" is not, at this
seed count.

**Hit rate does not transfer above pressure 1.0**, and this part is firm. The spread
grows to 0.13 [0.07, 0.15] and 0.15 [0.12, 0.17] at pressures 1.15 and 1.30, intervals
well clear of the sub-0.01 spreads below saturation. The direction is systematic: at
equal pressure, *more concurrent sessions collapse harder*. Thrashing depends on how
many ways the deficit is split, not only on how big the deficit is.

The asymmetry is still convenient for the project. The quantity actually claimed — how
much a better policy is worth — rides the pressure axis well enough to state with a
number and an interval. The quantity reported only as context — absolute hit rate —
needs the concurrency stated alongside it once pressure exceeds 1.

## The second useful result: the peak is at working set ≈ pool

Headroom is not monotone in pressure. It is near zero at 0.5, peaks at **17.4%** at
pressure 0.85, and decays back to 1.4% by pressure 1.6.

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

- 10 seeds. The collapse test now carries bootstrap intervals, and they are wide at the
  peak: the tightness of the agreement at pressures 0.85 and 1.00 is not resolved at this
  seed count. Raising to 15+ seeds is the cheap fix and has not been done.
- GPU utilisation is not held constant across the three conditions (0.856 to 0.998 at
  the same pressure), because more sessions means more offered work. That does not
  affect the token metrics used for the collapse test, but it does mean the cost and
  TTFT columns of this experiment are not a clean comparison.
- `bigger_gpu` models a larger KV pool but keeps every timing constant unchanged. A real
  96 GB card is also faster, so this condition isolates capacity, not hardware.
