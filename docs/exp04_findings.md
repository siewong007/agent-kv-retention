# EXP04 findings — what a real predictor captures, and where it starts losing

**Date:** 2026-08-01 · **Platform:** CPU simulation
**Data:** `results/exp04/` — 280 runs, 10 seeds, 4 termination-signal strengths x 7 arms,
at concurrency 10 (pressure 0.84, the headroom peak).
Predictors: gradient boosting, trained on seeds 9000–9002 and evaluated on seeds 0–9, so
train and test sessions are disjoint by construction.

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

Share of the `belady - lru` gap captured, 10 seeds:

| signal | gap vs LRU | oracle_terminal | belady_pause | **predict_terminal** | predict_guarded | predict | precision | recall |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 13.7% | 80.9% | 111.1% | **−16.9%** | −891.9% | −1534.5% | 0.689 | 0.096 |
| 0.75 | 17.7% | 57.1% | 93.4% | **−9.5%** | −670.9% | −1183.5% | 0.635 | 0.262 |
| 1.50 | 17.3% | 83.5% | 114.8% | **+32.8%** | −461.2% | −868.4% | 0.740 | 0.586 |
| 3.00 | 16.2% | 64.9% | 113.2% | **+57.9%** | −168.9% | −466.3% | 0.890 | 0.882 |

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

## Finding 3: termination prediction has a precision floor, below which it also loses

**This corrects a claim made from a 3-seed smoke run.** `predict_terminal` is *not*
unconditionally safe. At 10 seeds it is **negative at the two lowest signal strengths**:
−16.9% at precision 0.689, −9.5% at precision 0.635. It turns positive only once
precision reaches ≈0.74 with recall ≈0.59, and reaches +57.9% at precision 0.89.

The mechanism is the asymmetry of the two error types:

- a **false positive** marks a live session dead and evicts a cache that is about to be
  needed — the full cost of a miss;
- a **false negative** simply leaves the session in LRU order, where it would have been
  anyway — nearly free.

So the operating point matters more than the accuracy. At precision 0.635, roughly one
in three "this session is finished" calls destroys a cache that LRU would have kept, and
that outweighs the correct calls.

**Practical consequence: the classifier must be tuned for precision, not accuracy, and a
deployment must be able to measure its precision before trusting it.** The threshold was
fixed at 0.5 here, which is the wrong default for an asymmetric-cost decision; sweeping
it is the obvious next step and would likely move the crossover left.

## What this means for the project

The chain now reads: the gap exists (EXP01), no constant TTL reaches any of it (EXP01),
most of it at the operating point comes from termination rather than pause length
(EXP01), and a real classifier captures 33–58% of it **once it is accurate enough** and
loses money below that (here).

That is a defensible thesis result, and it is conditional in a way that is worth stating
plainly: predictive KV retention pays only above a measurable prediction-quality
threshold, and the naive furthest-in-future framing is actively harmful below it.

## What is not established

- **The signal strength is invented.** The dial is calibrated to nothing. A real agent's
  termination signal could sit anywhere on this curve, and where it sits decides whether
  the method pays at all. `docs/trace_schema.md` and `bench/fit_workload.py` exist to
  answer that from a real trace; until then, every number here is "at synthetic signal
  strength X".
- No bootstrap intervals. 10 seeds, point estimates only. The negative `predict_terminal`
  values at low signal are large enough not to be noise, but the crossover point between
  0.75 and 1.50 is not resolved.
- The classifier threshold was not tuned (fixed 0.5), and the cost asymmetry says it
  should be.
- The pause regressor was not given the features that would most plausibly help it in
  reality — the tool's own historical runtime distribution, or a per-tool prior. It gets
  `tool_id` and has to learn that mapping from scratch.
- One concurrency point (10). Finding 3's threshold may move with pressure, since EXP01
  shows termination information is worth less at higher pressure.
