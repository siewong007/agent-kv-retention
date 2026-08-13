# EXP05 findings — the decision threshold is the policy

**Date:** 2026-08-01 · **Platform:** CPU simulation
**Data:** `results/exp05/` — 180 runs, 10 seeds, 2 signal strengths x 6 thresholds,
concurrency 10 (pressure 0.84). Reference arms (`lru`, `oracle_terminal`, `belady`) do
not depend on the threshold and are run once per (signal, seed).

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

Weak signal (`termination_signal_strength = 0.75`), where oracle_terminal captures 57.1%
of the LRU-to-belady gap:

| threshold | precision | recall | false positives / seed | share of gap | vs LRU |
|---|---|---|---|---|---|
| 0.30 | 0.467 | 0.494 | **114** | **−46.2%** | −8.2% |
| 0.50 | 0.635 | 0.262 | 30 | −9.5% | −1.7% |
| 0.70 | 0.845 | 0.105 | 4 | +3.3% | +0.6% |
| 0.85 | 0.965 | 0.057 | 1 | −5.0% | −0.9% |
| 0.95 | 0.995 | 0.053 | 0 | **+4.4%** | +0.8% |
| 0.99 | 1.000 | 0.016 | 0 | +2.9% | +0.5% |

The share tracks the false-positive count almost linearly: 114 → 30 FPs moves it 36.7
points, and 30 → 4 moves it 12.8 points. **Each false positive costs about 0.45% of the
total available headroom.** Recall barely matters; precision is the whole lever.

## Result 2: the EXP04 precision floor was an operating point, not a property of the method — but the corrected answer is zero, not a win

Raising the threshold does flip the weak-signal case from −9.5% to positive. It does not
make it *good*: the best point is +4.4% of the gap, which is +0.8% against LRU, and the
curve is not monotone (0.85 sits at −5.0% between two positive points). At 10 seeds that
non-monotonicity says the whole weak-signal band is inside the noise.

**So the honest correction to EXP04 is: a badly chosen threshold turns a neutral policy
into a harmful one. Fixing the threshold removes the loss; it does not create a gain.**
The floor is on the operating point, not on the method.

Note the gap between what is *available* and what is *extractable* here: at this signal
strength `oracle_terminal` captures 57.1% of the headroom, and the best classifier
captures 4.4%. The information exists — the oracle proves it — and the features do not
carry it.

## Result 3: the optimal threshold moves with classifier quality, and reverses direction

Strong signal (`termination_signal_strength = 3.0`), oracle_terminal captures 64.9%:

| threshold | precision | recall | false positives / seed | share of gap | vs LRU |
|---|---|---|---|---|---|
| 0.30 | 0.829 | 0.926 | 38 | 41.9% | +6.8% |
| **0.50** | 0.890 | 0.882 | 22 | **57.9%** | **+9.4%** |
| 0.70 | 0.924 | 0.827 | 14 | 53.0% | +8.6% |
| 0.85 | 0.954 | 0.730 | 7 | 47.0% | +7.6% |
| 0.95 | 0.981 | 0.507 | 2 | 26.8% | +4.3% |
| 0.99 | 0.994 | 0.279 | 0 | 25.3% | +4.1% |

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

## The headline number

At strong signal and the right threshold, the learned classifier captures **57.9% of the
LRU-to-belady gap against `oracle_terminal`'s 64.9% — 89% of what perfect termination
knowledge delivers.** The prediction problem is not the bottleneck once the signal is
there; finding out whether real agents carry that signal is.

## What is not established

- **Two signal strengths only**, and both are invented. Where a real agent sits on this
  axis decides everything, and `docs/trace_schema.md` plus `bench/fit_workload.py` exist
  to answer that from a recorded trace.
- No bootstrap intervals. The weak-signal rows are explicitly described as noise here on
  the strength of their non-monotonicity, which is an argument, not an interval.
- One concurrency point (10, pressure 0.84). EXP01 shows termination information is worth
  less at higher pressure, so the threshold optimum may move with it.
- The threshold was swept on a fixed grid, not optimised. The true optimum at strong
  signal lies somewhere in 0.3–0.7 and was not located.
- The classifier was not recalibrated (e.g. Platt scaling) before thresholding, so
  "threshold 0.5" is 0.5 on an uncalibrated score, not on a probability.
