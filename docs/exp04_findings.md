# EXP04 findings — what a real predictor captures, and where it starts losing

**Date:** 2026-08-01 · **Platform:** CPU simulation
**Data:** `results/exp04_seeds15/` — 420 runs, **15 seeds**, 4 termination-signal
strengths x 7 arms, at concurrency 10 (pressure 0.84, the headroom peak). Intervals are
95% paired bootstrap over seeds. Predictors: gradient boosting, trained on seeds
9000–9002 and evaluated on seeds 0–14, so train and test sessions are disjoint by
construction. The 10-seed run is kept in `results/exp04/`; its point estimates run high
and should not be quoted.

## Why this experiment had to be rebuilt before it could be run

The first attempt measured nothing. The generator drew `n_turns = rng.randint(12, 30)`
independently of every other variable, so `P(this turn is the last | features)` was a
pure function of `turn_index` and the gradient-boosted classifier was already at the
Bayes limit at recall 0.10. Reporting a headroom share from that would have been a
statement about `randint`.

`WorkloadConfig.termination_signal_strength` now supplies a signal, indirectly: near the
end of a session outputs shorten and the tool mix shifts toward a designated finishing
tool. Neither reveals the remaining turn count. **The default is 0, which reproduces the
old generator byte-for-byte**, so the v2 results of EXP01–EXP03 remain valid
(`tests/test_invariants.py::test_termination_signal_off_by_default`), and a second test
checks the signal is neither absent nor a giveaway.

## The results

Share of the `belady - lru` gap captured, 15 seeds, 95% bootstrap intervals. `*` marks
an interval that excludes zero:

| signal | gap vs LRU % | oracle_terminal % | **predict_terminal %** | belady_pause | predict_guarded | predict | precision | recall |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 13.1 [10.3, 16.1]* | 85.3 [72.8, 101.7]* | **−11.3 [−42.2, 10.7]** | 118.9% | −946.6% | −1648.4% | 0.684 | 0.093 |
| 0.75 | 16.4 [12.2, 21.1]* | 65.0 [51.6, 82.3]* | **−11.7 [−31.8, 2.2]** | 97.9% | −763.9% | −1322.5% | 0.644 | 0.256 |
| 1.50 | 15.1 [9.7, 20.6]* | 78.1 [62.4, 94.6]* | **18.4 [−13.9, 36.0]** | 114.9% | −560.4% | −1011.8% | 0.742 | 0.594 |
| 3.00 | 15.0 [10.5, 20.1]* | 70.8 [49.5, 86.2]* | **49.6 [20.9, 67.2]\*** | 112.0% | −189.3% | −495.0% | 0.889 | 0.887 |

Two things are worth reading off this table before the findings below.

**The gap itself barely moves with signal strength** — 13.1% to 16.4%, intervals all
overlapping. The termination signal changes how *extractable* the headroom is, not how
much of it exists. `oracle_terminal` confirms that from the other side: it captures
65–85% at every signal strength, including 0.00 where nothing can be learned.

**Only the strongest signal produces a measurable win for the deployable arm.** Three of
the four `predict_terminal` intervals contain zero. The +49.6% at signal 3.00 matches
[EXP05](exp05_findings.md)'s independent 15-seed measurement of the same configuration
to the decimal, which is a useful cross-check between two experiments that share only
the simulator.

## Finding 1: `belady` is not an upper bound

`belady_pause` — the same oracle information ranked by pause length alone, with an LRU
tie-break — beats `belady` at three of the four points, by up to 15%.

Belady's optimality assumes a **fixed offline reference stream**. Neither condition holds
here. The arm is myopic (it sees the next use, not the whole future), and eviction
changes when work is recomputed, so the stream responds to the policy being evaluated.

Every "upper bound" claim about this arm elsewhere in the project has been corrected to
"strong oracle reference". Comparisons **against LRU** are unaffected; what changes is
that headroom shares may legitimately exceed 100%, and the genuinely available gain is
larger than reported, not smaller.

## Finding 2: a noisy ordering is worse than no ordering, because LRU is not naive

`predict` reaches **−1534%** of the gap. `predict_guarded`, built specifically to fail
safely by ranking on the predicted pause alone so that uninformative predictions tie and
fall back to LRU, still reaches −892%.

The guard did not work, and why it did not is the finding. The regressor's output is not
constant — it is *varied and wrong* (log-RMSE 1.33 against a true pause spread of 1.20 in
log units, so the prediction carries less signal than the quantity it orders by).
Ordering by a wrong pause is close to random eviction, and random eviction is far worse
than LRU.

**LRU is not a naive baseline on this workload. Age is a termination signal**: a block
that has sat unused belongs to a session that probably ended. Any policy that overrides
LRU ordering discards that signal and has to pay for it with something better. A pause
regressor at this accuracy is not better.

This is the concrete form of the warning for the literature's framing. "Predict the next
use, evict furthest-future" has no safe failure mode, and the failure is not graceful
degradation to the incumbent — it is a large loss.

## Finding 3: termination prediction wins only at the strongest signal tested

> **Refined by [EXP05](exp05_findings.md), at 15 seeds with intervals.** This experiment
> fixed the decision threshold at 0.5, which is close to the worst defensible choice for
> a decision whose two error types cost wildly different amounts. Correcting the
> threshold changes the weak-signal reading from "loses" to "indistinguishable from LRU
> at every threshold above 0.5" — the threshold controls how much you lose, not whether
> you win. At strong signal 0.5 turns out to be near-optimal, and the optimum reverses
> direction with classifier quality, so it has to be measured rather than assumed.
> The numbers on this page are 10-seed point estimates and run high; the 15-seed run
> puts the strong-signal share at 49.6% rather than 57.9%.

**This finding has now been corrected twice, in the same direction both times, and each
correction came from adding seeds.** A 3-seed smoke run said `predict_terminal` is never
negative. A 10-seed run said it is negative below precision ~0.74 and positive above.
The 15-seed run with intervals says something narrower than either: **at precision 0.68,
0.64 and 0.74 the arm is indistinguishable from LRU, and only at precision 0.89 does it
measurably win.** The negatives were not significant, and neither was the +32.8% at
signal 1.50 that the 10-seed run reported as a success — it is 18.4% [−13.9, 36.0].

What survives is the asymmetry that drives it.

The mechanism is the asymmetry of the two error types:

- a **false positive** marks a live session dead and evicts a cache that is about to be
  needed — the full cost of a miss;
- a **false negative** simply leaves the session in LRU order, where it would have been
  anyway — nearly free.

So the operating point matters more than the accuracy. At precision 0.64, roughly one in
three "this session is finished" calls destroys a cache that LRU would have kept, which
is enough to cancel the correct calls — not enough, at 15 seeds, to prove a loss.

**Practical consequence: the classifier must be tuned for precision, not accuracy, and a
deployment must be able to measure its precision before trusting it.**

[EXP05](exp05_findings.md) swept the threshold and the guess above was half right. It
does remove the loss at weak signal, but it does not "move the crossover left" into a
gain -- the corrected weak-signal answer is approximately zero. And at strong signal the
0.5 used here turns out to be near-optimal, with higher thresholds hurting. The optimum
reverses direction with classifier quality.

## What this means for the project

The chain now reads: the gap exists (EXP01), no constant TTL reaches any of it (EXP01),
most of it at the operating point comes from termination rather than pause length
(EXP01), and a real classifier captures **49.6% [20.9, 67.2]** of it -- 70% of what a
perfect termination oracle delivers, for a gain of +7.4% [+2.3, +12.6] over LRU -- but
only at the strongest termination signal tested. At weaker signals it is
indistinguishable from doing nothing.

That is a defensible thesis result, and it is conditional in a way that is worth stating
plainly: predictive KV retention pays only when the workload carries a termination
signal AND the classifier is operated at the right point on its own ROC curve. The naive
furthest-in-future framing is actively harmful regardless.

## What is not established

- **The signal strength is invented.** The dial is calibrated to nothing. A real agent's
  termination signal could sit anywhere on this curve, and where it sits decides whether
  the method pays at all. `docs/trace_schema.md` and `bench/fit_workload.py` exist to
  answer that from a real trace; until then, every number here is "at synthetic signal
  strength X".
- Only one of four signal strengths gives a measurable win, so the *shape* of the curve
  between signal 1.5 and 3.0 is unresolved: where the crossover actually sits, and
  whether it is sharp or gradual, would need points in between and more seeds than 15.
- The `predict_terminal` intervals are 30-60 percentage points wide. Establishing that a
  configuration is *neutral* rather than merely unresolved would need far more seeds
  than establishing that it wins.
- ~~The classifier threshold was not tuned (fixed 0.5)~~ — done in
  [EXP05](exp05_findings.md), which changes the reading of Finding 3.
- The pause regressor was not given the features that would most plausibly help it in
  reality — the tool's own historical runtime distribution, or a per-tool prior. It gets
  `tool_id` and has to learn that mapping from scratch.
- One concurrency point (10). Finding 3's threshold may move with pressure, since EXP01
  shows termination information is worth less at higher pressure.
