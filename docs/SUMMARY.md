# Summary of findings — the numbers as they currently stand

Written to be the single place a report draws from. Every number here carries its 95%
paired-bootstrap interval over seeds, and every one is traceable to a results directory
that contains the config and seed that produced it.

**Read the caveats section before quoting anything.** The engine is calibrated against
real hardware; the workload is not.

> **2026-08-16, from [validation_findings.md](validation_findings.md):** the simulator
> has been compared to vLLM end-to-end across seven runs, and the result is a **range of
> validity rather than a verdict**. Up to about pressure 1.1 it tracks the real server:
> makespan within 2% at pressures 0.64, 1.02 and 1.08, hit rate within 1.4 pp at 0.64 and
> 1.08 with the sign flipping. At pressure 1.27, on comparisons that are equally exact,
> it is **4.8–10.9 pp pessimistic on hit rate and 11–19% slow on makespan**. Peak headroom
> (pressure 0.84) and every per-experiment run sit inside the validated range; EXP02's
> high-pressure tail does not, and the collapse it shows is probably exaggerated. Two
> earlier readings of these runs were retracted along the way — a 3.2x eviction claim that
> was an accounting artefact, and a preemption claim that was an inference from it — both
> documented in place.

---

## The question and the answer in one paragraph

An agent task issues 20–30 LLM calls sharing a growing prefix, separated by pauses while
tools run. Current engines keep prefix-cache blocks until something else needs the space,
so a session that pauses long enough returns to a cold cache. The question was whether
predicting agent behaviour and feeding that into cache retention is worth building. The
answer is a qualified yes: there is a real gap between LRU and an oracle, no tuned
constant reaches any of it, most of it comes from knowing a session has *ended* rather
than how long it pauses, and a learned classifier captures about half of it — but only
when the workload carries a strong termination signal and the classifier is operated at
the right point on its ROC curve. Below that, the deployable policy is indistinguishable
from doing nothing, and the naive "predict the next use, evict furthest-future" framing
is actively harmful at any accuracy.

---

## Headline numbers

All at concurrency 8–18 on a KV pool of 12868 blocks (measured, RTX 5080 + Qwen2.5-3B).

| quantity | value [95% CI] | seeds | source |
|---|---|---|---|
| peak headroom, oracle vs LRU (tokens recomputed) | **13.7% [12.5, 15.0]** at pressure 0.84 | 100 | `results/exp01_share_seeds100` |
| share of it from knowing the session ended | **73.6% [66.0, 81.2]** at the same point | 100 | same |
| share captured by *any* tuned constant TTL | **exactly 0%**, bit-for-bit, at every TTL | — | structural, test-guarded |
| learned classifier, strong signal, threshold 0.5 | **49.6% [20.9, 67.2]** of the gap | 15 | `results/exp04_seeds15` |
| the same, as a gain over LRU | **+7.4% [+2.3, +12.6]** | 15 | `results/exp05_seeds15` |
| cost conversion, wall-clock billing | 4.1% cost for a 13.7% token cut at the peak | 100 | `results/exp01_share_seeds100` |
| cost conversion, GPU-time billing | 20–69% of the token saving, falling as pauses lengthen | 15 | `results/v3_exp03` |

## The six findings, in the order they were established

### 1. A uniform TTL is *identical* to LRU — not similar, identical

Bit-identical output at every TTL from 1 s to 1e9 s. In a demand-evicted pool a uniform
TTL gives every block the same protection window, so expiry order equals release order
equals LRU order and the eviction sequence never changes.

**Consequence:** all of a TTL policy's value comes from its *variance across sessions*.
A constant TTL is not a competing baseline; it is LRU wearing a parameter. This closed
off the cheapest way the project could have died.

Guarded by `tests/test_invariants.py::test_uniform_ttl_is_exactly_lru`.

### 2. Perfect information through the wrong mechanism is worse than nothing — under scarcity

Using the true pause as a TTL loses to plain LRU at every pressure ≥ 0.84, by up to 14%.
Below saturation (pressure 0.67) it *helps*, by 2.9%. Both directions have intervals
excluding zero at 100 seeds.

It protects longest exactly the sessions that return latest. The correct use of a pause
estimate under scarcity is as an eviction *priority*, not a protection *duration*.
Information and mechanism are separate axes, and the literature's TTL framing picks the
wrong one.

### 3. Termination is most of the prize, and its share falls with pressure

| pressure | 0.67 | 0.84 | 1.01 | 1.17 | 1.34 |
|---|---|---|---|---|---|
| headroom % | 10.1 [9.1, 11.1] | **13.7 [12.5, 15.0]** | 12.4 [11.3, 13.4] | 8.8 [8.0, 9.7] | 5.2 [4.7, 5.8] |
| share from termination % | 86.1 [78.0, 94.5] | **73.6 [66.0, 81.2]** | 55.0 [48.4, 61.7] | 45.5 [39.6, 51.5] | 31.8 [24.3, 39.5] |

The decline is monotone with barely-overlapping neighbouring intervals, and no point
includes zero. At low pressure the only useful act is evicting dead sessions; at high
pressure every live session competes and the ordering *among* them decides the outcome,
which is exactly what termination information cannot tell you.

### 4. `belady` is not an upper bound

`belady_pause` — the same oracle information ranked by pause length with an LRU
tie-break — beats it at three of four points, by up to 15%. Belady's optimality assumes
a fixed offline reference stream; here eviction changes when work is recomputed, so the
stream responds to the policy. The arm is also myopic: next use only, not the whole
future.

Everything stated as "headroom vs LRU" is unaffected. What changes is that shares can
legitimately exceed 100%, and the genuinely available gain is *larger* than reported.

### 5. A noisy ordering is worse than no ordering, because LRU is not naive

Ranking by a predicted pause reaches **−1648%** of the gap. A version built to fail
safely, ranking on the predicted pause alone so uninformative predictions tie back to
LRU, still reaches −947%. The regressor's output is not constant, it is varied and wrong
(log-RMSE 1.33 against a true spread of 1.20), and ordering by a wrong pause approaches
random eviction.

**Age is a termination signal.** A block that has sat unused belongs to a session that
probably ended, so LRU is already exploiting real information. Any policy overriding its
ordering must beat that, and a pause regressor at this accuracy does not.

### 6. A cache win is not a cost win, and the billing model changes the answer 8.6-fold

Closed loop, concurrency 16, sweeping the median tool pause. Cost conversion falls as
pauses lengthen under either billing model, and the two models diverge sharply at the
long end:

- **wall-clock billing** (a reserved box, how the Sunway HPC session is charged): at a
  30 s median pause, tool execution is most of the wall clock and no retention policy can
  touch it. The saving goes to under 1%. This is arithmetic, not a measurement.
- **GPU-time billing** (shared or autoscaled): the same runs hold 7.1% — **8.6x more,
  from identical simulator runs.** The billing model is not a presentation choice.

Numbers refreshed from `results/v3_exp03`; see [exp03_findings.md](exp03_findings.md).

---

## What is NOT established

Ordered by how much damage each would do to the report.

1. **The workload distributions are invented.** Pause length, tool-result size, and above
   all the *termination signal strength* are calibrated to nothing. The one configuration
   where the learned predictor measurably wins corresponds to a classifier at 0.89
   precision, and whether a real agent affords that is unknown. Machinery to answer it
   from a recorded trace exists and is tested ([trace_schema.md](trace_schema.md),
   `bench/fit_workload.py --self-test`), but no real trace has been fitted.
2. **Pressure is necessary but not sufficient.** It transfers across a 4x change in pool
   size and *not* across a 2.5x change in session count: at equal pressure, 48 sessions
   show less than half the headroom of 19. Any transfer claim needs both terms.
3. **Open-loop operation is metastable near saturation** and its aggregates there are
   bimodal mixtures. More seeds estimate the mixture better, not an operating point —
   the fitted interval-shrinkage exponent is 0.10 against an ideal 0.5.
4. **One model, one GPU, one context-length regime.** Qwen2.5-3B on an RTX 5080 under
   WSL2, and as of 2026-08-19 that is the final platform by decision, not by default — an
   HPC round was scripted and deliberately not run. So nothing here is checked against a
   second model or a headless server, and two WSL2 effects are baked into the constants
   rather than averaged away: `step_overhead_s` carries a native-sampler penalty from the
   missing CUDA toolkit, and the KV pool is what a Windows desktop leaves behind. Report
   the constants as belonging to this setup rather than to the card.
5. ~~The simulator has never been validated against vLLM end-to-end.~~ **Done, with a
   boundary.** It tracks vLLM to about pressure 1.1 (2% on makespan, 1.4 pp on hit rate)
   and comes apart above it (11–19%, 4.8–10.9 pp at pressure 1.27). Everything quoted in
   this document is from inside the validated range except EXP02's high-pressure tail.
   The boundary is located only to somewhere in (1.08, 1.27), and every point on both
   sides of it is a single seed. See [validation_findings.md](validation_findings.md).
6. **`predict_terminal`'s neutrality at weak signal is unresolved, not proved.** Showing
   an arm is neutral needs far more seeds than showing it wins; those intervals are
   30–60 pp wide.

---

## Methodological points worth a section of their own in the report

These cost real time to learn and are the part a reader is least likely to already know.

**Never quote a point estimate without its interval.** Every headline in this project
that was once quoted bare later turned out to be wrong: the oracle share moved 89% → 70%
between 10 and 15 seeds; the predictor's apparent win at a middling signal became
18.4% [−13.9, 36.0]; the peak headroom moved 18.3% → 13.7% when a generator change was
finally propagated.

**Proving "A beats B" is cheap; estimating "by how much" is expensive.** Interval width
falls as 1/sqrt(seeds), so tightening from ±16 pp to ±5 pp costs about ten times the
compute. `experiments/seed_sufficiency.py` fits the actual decay per claim and projects
the cost, which is how the 100-seed run was scoped rather than guessed.

**A shallow shrinkage exponent means seeds will not help.** Fitting width against seed
count separates "needs more samples" from "estimating the wrong thing". The open-loop
runaway rows decay as n^-0.10; the projection of 5295 seeds is a proof that the quantity
has no single value, not a plan.

**Validate the quantity you are deciding about, not its upstream input.** The generator's
tool multipliers were normalised, and the decision not to rerun EXP01 was justified by
showing the change was unbiased for the *pause median*. Headroom is a nonlinear function
of the pause distribution, and it moved 28%. Stale numbers then sat in the findings docs
until they were noticed by accident. `tests/test_invariants.py::test_no_result_directory_is_stale`
now fails the build instead.

**Superseded runs are kept, not deleted.** Every findings document that cites an
old number says which run replaced it and why. The corrections are the most credible part
of the record.

---

## Where each number lives

| results directory | what it is | status |
|---|---|---|
| `exp01_share_seeds100` | EXP01 headroom + termination share, 100 seeds, 4 arms | **current** |
| `v2_exp02_seeds15` | EXP02 pressure axis, 15 seeds | **current** |
| `v3_exp03` | EXP03 pause isolation, 15 seeds | **current** |
| `exp04_seeds15` | EXP04 predictor, 15 seeds | **current** |
| `exp05_seeds15` | EXP05 threshold sweep, 15 seeds | **current** |
| `calib` | vLLM timing and batch fits | **current** |
| `exp01`, `exp01_seeds15`, `v2_exp01_seeds15` | earlier EXP01 runs | superseded |
| `exp02`, `v2_exp02` | earlier EXP02 runs | superseded |
| `exp03`, `v2_exp03` | earlier EXP03 runs | superseded |
| `exp04`, `exp05` | 10-seed predictor runs | superseded |

Superseded directories are kept deliberately: the movement between them is the evidence
for the methodological points above.
