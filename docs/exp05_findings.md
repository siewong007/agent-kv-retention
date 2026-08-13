# EXP05 findings — the decision threshold is the policy

**Date:** 2026-08-01 · **Platform:** CPU simulation
**Data:** `results/exp05_seeds15/` — 270 runs, **15 seeds**, 2 signal strengths x 6
thresholds, concurrency 10 (pressure 0.84). Intervals are 95% paired bootstrap over
seeds (5000 resamples, seeded). Reference arms (`lru`, `oracle_terminal`, `belady`) do
not depend on the threshold and are run once per (signal, seed).
A 10-seed run is kept in `results/exp05/`; where the two disagree the 15-seed numbers
are the ones to use, and the disagreement is itself discussed below.

## The question

[EXP04](exp04_findings.md) found that `predict_terminal` loses to LRU below roughly 0.7
precision. It ran at a decision threshold of 0.5 — the default for a *balanced* problem,
and this problem is not balanced:

| error | what it does | cost |
|---|---|---|
| false positive | marks a live session dead, evicts its cache first, it recomputes its whole context on return | the full cost of a miss |
| false negative | leaves a dead session in LRU order, where it would have been anyway | close to free |

With costs that asymmetric, 0.5 is close to the worst defensible choice. This experiment
sweeps the threshold.

## Result 1: false positives are the entire story, and they are roughly linear

Weak signal (`termination_signal_strength = 0.75`), where oracle_terminal captures 65.0%
of the LRU-to-belady gap:

| threshold | precision | recall | FPs / seed | gain vs LRU % [95% CI] | share of gap |
|---|---|---|---|---|---|
| 0.30 | 0.471 | 0.494 | **112** | **−9.08 [−14.31, −4.76]** | −55.4% |
| 0.50 | 0.644 | 0.256 | 29 | −1.91 [−4.37, +0.42] | −11.7% |
| 0.70 | 0.838 | 0.100 | 4 | −0.24 [−2.40, +2.23] | −1.4% |
| 0.85 | 0.962 | 0.056 | 1 | −0.48 [−2.56, +1.42] | −2.9% |
| 0.95 | 0.996 | 0.051 | 0 | +1.18 [−0.74, +2.94] | +7.2% |
| 0.99 | 0.933 | 0.016 | 0 | +0.74 [−0.69, +2.06] | +4.5% |

The damage tracks the false-positive count: 112 FPs costs 9.1% against LRU, 29 costs
1.9%, 4 costs 0.2%. **Roughly 0.08 percentage points of throughput per false positive
per seed**, and recall barely enters. Precision is the whole lever.

**Only one row is statistically distinguishable from LRU, and it is a loss.** Every
threshold from 0.50 up has an interval spanning zero. The best point estimate (+1.18% at
threshold 0.95) is not a win; it is a number whose interval includes doing nothing.

## Result 2: at weak signal, the threshold can only avoid harm — it cannot produce a win

At 10 seeds this looked like a rescue: −9.5% at threshold 0.5 became +4.4% at 0.95, and
the reading was "the floor is an operating point, not a property of the method". With 15
seeds and intervals, that reading is too generous. **No threshold at weak signal beats
LRU by a measurable amount.** The intervals from 0.50 upward all contain zero; only the
reckless setting (0.30) is distinguishable, and it is a 9.1% loss.

**The corrected statement: at weak signal the decision threshold controls how much you
lose, not whether you win.** Setting it well takes the policy from harmful to neutral.
There is nothing above neutral to reach.

The gap between what is *available* and what is *extractable* is the reason.
`oracle_terminal` captures 65.0% of the headroom at this signal strength, so the
information is there — the oracle proves it. The features do not carry it, and no
decision rule applied to a weak score can manufacture information that the score does
not contain.

## Result 3: the optimal threshold moves with classifier quality, and reverses direction

Strong signal (`termination_signal_strength = 3.0`), oracle_terminal captures 70.8%:

| threshold | precision | recall | FPs / seed | gain vs LRU % [95% CI] | share of gap |
|---|---|---|---|---|---|
| 0.30 | 0.832 | 0.931 | 38 | +6.61 [+1.97, +11.68] | 44.0% |
| **0.50** | 0.889 | 0.887 | 22 | **+7.44 [+2.27, +12.62]** | **49.6%** |
| 0.70 | 0.923 | 0.832 | 14 | +6.93 [+2.50, +11.52] | 46.1% |
| 0.85 | 0.952 | 0.735 | 7 | +5.94 [+1.37, +10.82] | 39.5% |
| 0.95 | 0.984 | 0.514 | 2 | +4.26 [+0.43, +8.28] | 28.3% |
| 0.99 | 0.994 | 0.280 | 0 | +2.11 [−0.63, +4.52] | 14.1% |

Five of the six thresholds beat LRU with intervals that exclude zero. Only the most
conservative setting (0.99, which flags almost nothing) fails to.

Here 0.5 is near-optimal and raising the threshold *hurts*, monotonically, because recall
starts being worth something once the classifier is good enough that recall does not cost
many false positives.

**The rule is therefore conditional on the classifier, not fixed:**

- **weak classifier** → be conservative. Act rarely, at high threshold. The best you can
  do is avoid losing.
- **strong classifier** → use the recall. Around 0.5, where the marginal true positive
  still outweighs the marginal false positive.

A deployment cannot pick this from theory; it has to measure its own precision/recall
curve and locate itself. That is a real operational requirement and worth stating as
one, not a footnote.

## The headline number, and a correction to how it was first reported

At strong signal and the right threshold, the learned classifier captures **49.6% of the
LRU-to-belady gap, against `oracle_terminal`'s 70.8% — 70% of what perfect termination
knowledge delivers**, for a gain over LRU of **+7.4% [+2.3, +12.6]**.

The 10-seed version of this run reported 57.9% and 64.9%, i.e. 89% of the oracle. Both
numbers moved, in opposite directions, and the ratio fell from 89% to 70%. Nothing about
the method changed — only the seed count. **That is the argument for never quoting a
point estimate from this project without its interval**, and it applies to the 89% figure
that appeared in an earlier draft of the EXP04 findings and the README.

The prediction problem is not the bottleneck once the signal is there. Finding out
whether real agents carry that signal is.

## What is not established

- **Two signal strengths only**, and both are invented. Where a real agent sits on this
  axis decides everything, and `docs/trace_schema.md` plus `bench/fit_workload.py` exist
  to answer that from a recorded trace.
- The weak-signal rows are shown to be indistinguishable from LRU, which is not the same
  as showing they are equal to it. With 15 seeds the intervals are roughly +/-2.5
  percentage points wide, so a real gain smaller than that would not be detected.
- One concurrency point (10, pressure 0.84). EXP01 shows termination information is worth
  less at higher pressure, so the threshold optimum may move with it.
- The threshold was swept on a fixed grid, not optimised. The true optimum at strong
  signal lies somewhere in 0.3–0.7 and was not located.
- The classifier was not recalibrated (e.g. Platt scaling) before thresholding, so
  "threshold 0.5" is 0.5 on an uncalibrated score, not on a probability.
