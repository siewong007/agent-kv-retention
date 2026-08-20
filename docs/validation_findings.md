# Validation against vLLM — validated to about pressure 1.1, not beyond

**Date:** 2026-08-14, revised 2026-08-19 · **Data:** `results/validate/` (pressure 0.64),
`results/validate_pressured/` (pressure 1.02, invalid — see below),
`results/validate_matched_admission/` (pressure 1.08),
`results/sweep_admission/` (pressure 1.27, four admission widths),
`results/seeds_1p27/` (pressure 1.27, paired seeds at two widths),
`results/validate_diagnosis/` · **Harness:** `bench/validate_vs_vllm.py`,
`bench/sweep_admission.sh`, re-analysis in `bench/diagnose_validation.py` and
`bench/analyze_admission_sweep.py`

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
>
> **Second correction, 2026-08-16.** That revision then explained the inflation as
> preemption, and concluded that the simulator never preempts where vLLM constantly
> does. A sweep that measures `vllm:num_preemptions_total` directly says both halves
> are wrong. At pressure 1.27 vLLM inflates its query count 4.2x while preempting
> **three times**, which accounts for 0.18% of the excess; and where both systems do
> preempt, **the simulator preempts more, not less** (10 vs 3, 19 vs 4). The inflation
> counts scheduling *attempts*, not preemptions. The sweep also found the real limit
> of the validation, which is neither of those things: see the last section.
>
> **Third correction, 2026-08-19.** The sweep was one seed per point, and one of the
> conclusions drawn from it does not survive four. It reported gaps of 4.8 pp and
> 10.9 pp at two admission widths at the SAME pressure, and used that 2.3x difference
> to rule out a uniform effective-pressure offset as the explanation. Over paired
> seeds the width effect is **−1.2 pp, 95% [−6.1, +1.4]** — it spans zero. The 2.3x
> was seed noise and the dull explanation is back in contention. What the seeds DO
> establish is the gap itself, which was previously only measured: see below.

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
| ~1.27 | matched, cap 6 | wall clock | — | — | **+8.8%** (4 seeds) |
| ~1.27 | matched, cap 8 | wall clock | — | — | **+6.7%** (3 seeds) |
| **~1.27** | **matched, cap 6** | **prefix-cache hit rate** | — | — | **−6.0 pp, 95% [−9.4, −2.5]** |
| **~1.27** | **matched, cap 8** | **prefix-cache hit rate** | — | — | **−4.1 pp, 95% [−5.3, −2.1]** |

**The timing model is validated up to about pressure 1.1.** Makespan agrees within 2%
at 0.64, 1.02 and 1.08, from four fitted constants and a hand-written scheduler. At
1.27 it does not: the simulator runs 7–9% slow, averaged over paired seeds. (The
single-seed figures were 11% and 19%; seed 0 was the extreme one.) Cost and latency
figures inherit that boundary.

**The hit-rate model is validated on the same range and no further.** At 0.64 and at
1.08 vLLM scheduled every prompt exactly once (inflation 1.0000, preemptions 0), so its
counters *are* a per-request hit rate and the comparison is exact: 1.4 pp both times,
with the sign flipping, which is what a small definitional difference looks like rather
than a bias. Both of those are single seeds.

At pressure 1.27, over paired seeds on comparisons that are equally exact, the
simulator is **4.1 pp [2.1, 5.3] pessimistic at cap 8 and 6.0 pp [2.5, 9.4] at cap 6**.
Both intervals exclude zero, so unlike everything else in this document the
high-pressure disagreement is now established rather than merely observed.

The agreement therefore does not hold everywhere; it holds up to roughly pressure 1.1
and degrades sharply above it. **EXP01/EXP02's peak headroom sits at pressure 0.84,
inside the validated range.** The high-pressure tail of EXP02, where headroom collapses
by 1.6, is outside it and is not backed by any measurement.

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

## The admission models differ — but not the way this document first said

`sim/engine.py` admits a request **only if its whole prompt fits**, so it cannot
over-commit at admission time. vLLM admits optimistically and grows into the space with
chunked prefill. That difference is real and is visible in the pressure-1.02 run, where
vLLM issued 1.694 scheduling attempts per prompt and the simulator issued 1.000.

What this document previously concluded from that -- that vLLM therefore preempts
constantly while the simulator never does -- was an inference, not a measurement, and the
sweep in the next section shows it is wrong in both directions. Whole-prompt admission
stops the simulator over-committing on *admit*; it does not stop `_preempt_one` firing
when a decode crosses a block boundary with no free block, and under real pressure it
fires more often than vLLM's preemption does. Read the scheduling-attempt count as what it
is -- a count of scheduling attempts -- and take preemption from
`vllm:num_preemptions_total`, which the harness now records.

## The admission sweep, and what it corrected

`bench/sweep_admission.sh` runs the same workload at four admission widths with the KV
pool **pinned** by `--kv-cache-memory` (10922 blocks at every point, pressure 1.272), so
admission width is the only thing varying. Changing `max_num_seqs` moves the pool on its
own -- it went 13663 to 12865 between the default and 8 -- and a sweep that let that
happen would confound admission width with memory pressure, which is the one thing it
exists to separate. The script stops rather than produce a mixed curve.

| max_num_seqs | inflation vLLM | inflation sim | preempt vLLM | preempt sim | hit vLLM | hit sim | delta |
|---|---|---|---|---|---|---|---|
| 6 | 1.000 | 1.000 | 0 | 0 | 0.5733 | 0.4644 | **−10.9 pp** |
| 8 | 1.000 | 1.000 | 0 | 0 | 0.5668 | 0.5193 | **−4.8 pp** |
| 12 | 4.230 | 1.010 | **3** | **10** | — | 0.4732 | invalid |
| 16 | 5.546 | 1.029 | **4** | **19** | — | 0.5273 | invalid |

Three things fall out, and two of them contradict what this document said yesterday.

**The inflation is not preemption.** At `max_num_seqs=12` the query count is 4.23x the
prompt tokens sent -- about 49.6 million extra queried tokens -- while vLLM preempted
**three times**. Three preemptions of at most 30000 prompt tokens each is 90000 tokens,
**0.18%** of the excess. Whatever the counters are counting, it is scheduling *attempts*:
a request that waits for KV blocks appears to be re-queried on each scheduler pass. The
earlier reading of the pressure-1.02 run -- "1.694x, because 69% of prompt tokens were
preempted and re-scheduled" -- was an inference from the same mistaken premise, and that
run never measured preemption at all, so it should be read as "1.694 scheduling attempts
per prompt" and nothing more.

**The simulator preempts more than vLLM, not less.** 10 against 3, and 19 against 4. The
claim that whole-prompt admission means the simulator "never preempts" was only ever true
of admission: `_preempt_one` still fires when a decode crosses a block boundary with no
free block, and under real pressure it fires more often than vLLM's does. The admission
model differs, but not in the direction or the magnitude previously claimed here.

**The real limit is pressure, not admission width.** The two rows where the comparison is
exact disagree by 4.8 and 10.9 pp, against 1.4 pp at pressure 1.08 with the same
`max_num_seqs=8`. Holding admission fixed and raising pressure from 1.08 to 1.27 took the
disagreement from 1.4 pp to 4.8 pp, and the makespan error from 1.5% to 11%. That is the
mapping the sweep was run to find, and the answer is a boundary rather than a conversion
factor: **the simulator tracks vLLM up to about pressure 1.1 and comes apart above it.**

The direction is consistent across both quantities -- the simulator holds less cache and
takes longer -- and whole-prompt admission remains the natural explanation, now with the
sign the data actually shows. Admitting a prompt whole requires enough free blocks at one
instant, so it evicts a larger contiguous slice of other sessions' cache than vLLM's
chunked prefill does, and it serialises admission while it waits for that slice. Lower hit
rate and longer makespan follow from the same mechanism. This is an explanation that fits,
not one that has been tested.

## The paired seeds, and the claim they withdrew

`bench/seeds_at_pressure.sh` repeats the pressure-1.27 comparison on seeds 1–3 at both
admission widths, under conditions held identical to the sweep so its seed 0 is a fourth
data point rather than a near-miss: the same pinned pool (10922 blocks), the same 60
sessions at concurrency 16, `max_num_seqs` passed to both sides. The server is restarted
between seeds, so it never begins warm against a simulator that begins cold.

| cap | seed | vLLM | simulator | gap | makespan sim/vLLM |
|---|---|---|---|---|---|
| 6 | 0 | 0.5733 | 0.4644 | −10.9 pp | 1.187 |
| 6 | 1 | 0.6730 | 0.6336 | −4.0 pp | 1.078 |
| 6 | 2 | 0.6250 | 0.6147 | −1.0 pp | 1.006 |
| 6 | 3 | 0.4781 | 0.3983 | −8.0 pp | 1.082 |
| 8 | 0 | 0.5668 | 0.5193 | −4.8 pp | 1.107 |
| 8 | 1 | 0.6857 | 0.6324 | −5.3 pp | 1.075 |
| 8 | 2 | 0.6515 | 0.6307 | −2.1 pp | 1.019 |
| 8 | 3 | *dropped* | | | |

Seed 3 at cap 8 is excluded because vLLM's query inflation was 1.281 there: it preempted,
its counters stopped being a per-request hit rate, and averaging it in would reproduce the
25 pp phantom this document already retracted once. It is dropped with its reason rather
than silently.

**What the seeds establish.** The gap is real:

| | gap | 95% interval | n |
|---|---|---|---|
| cap 6 | **−6.0 pp** | [−9.4, −2.5] | 4 |
| cap 8 | **−4.1 pp** | [−5.3, −2.1] | 3 |

Both intervals exclude zero. The simulator is genuinely pessimistic about cache retention
at pressure 1.27, by roughly four to six points, and roughly 7–9% slow on makespan. That
is the first disagreement in this document that has an interval attached to it.

**What the seeds withdrew.** The previous section used the *difference between the two
widths* — 4.8 pp against 10.9 pp at identical pressure, a 2.3x ratio — to rule out a
uniform effective-pressure offset as the explanation, on the grounds that one offset
cannot produce two different gaps at one pressure. Paired on the three shared seeds, that
width effect is **−1.2 pp, 95% [−6.1, +1.4]**. It spans zero. The 2.3x was one seed's
noise; seed 0 happened to be the extreme member of the cap-6 group, whose other seeds gave
−4.0, −1.0 and −8.0 pp.

So the dull explanation is back in contention. Dividing each gap by the local slope
(`bench/pressure_sensitivity.py`) gives implied offsets of **+0.041** at cap 8 and
**+0.060** at cap 6, which agree to within 0.02 of pressure. On the multi-seed evidence,
the simulator behaves like a server at slightly higher pressure than its nominal one, by
about +0.05, and the "breakdown above 1.1" may be that same small error read off a much
steeper part of the curve — the slope is 17x larger at 1.08 than at 0.64.

That is not confirmed. It is no longer excluded, which is a weaker and more accurate thing
to say. The two lower-pressure points that would test it across the curve are single seeds,
and at pressure 0.64 the slope is −0.061, small enough that dividing by it turns any noise
in one run into a meaningless offset.

## What this does to the project's claims

**Validated below about pressure 1.1:** timing (2% on makespan at 0.64, 1.02 and 1.08)
and hit rate (1.4 pp at 0.64 and 1.08, sign flipping). Cost conversion, the wall-clock
versus GPU-time billing divergence, and the arithmetic bound on what a retention policy
can touch at long pauses all rest on the timing model and inherit this range.

**EXP01/EXP02's headline sits inside it.** Peak headroom is at pressure 0.84, and the
per-experiment runs are at or below that. Those numbers are backed by measurement.

**Not validated above it, and now quantified.** At pressure 1.27 the simulator is
4.1 pp [2.1, 5.3] pessimistic at cap 8 and 6.0 pp [2.5, 9.4] at cap 6, and 7-9% slow on
makespan. EXP02's high-pressure tail -- the collapse of headroom by pressure 1.6 -- is in
that region. It should be described as a property of the simulator, not a prediction about
a server, and the *shape* of the collapse is the part to distrust: a simulator holding
less cache than vLLM under pressure exaggerates exactly that collapse.

**Argued, not measured:** the policy *ranking*. All arms run inside the same admission
model, and the mechanism that differs is policy-independent -- whole-prompt admission and
preemption are identical code for `lru`, `belady` and `predict` alike. That is a reason to
expect the ranking to be robust, not evidence that it is.

## How to close the remaining item

Step 2 below is done and is what produced the section above. What is left is step 1, now
sharper for having a real effect size to test against:

1. **Paired seeds at pressures 1.08 and 0.64.** Both are single runs, and they are the
   points that would decide between "one uniform pressure offset" and "the model degrades
   above 1.1". If the offset implied at 1.08 comes out near +0.05 with an interval, the
   uniform-offset reading is confirmed and the "boundary at 1.1" framing should go; if it
   comes out near zero, the model really does change behaviour and the admission
   difference is the place to look. About 40 minutes per seed at 1.08.
2. ~~Repeat the pressure-1.27 point on more seeds.~~ Done: `results/seeds_1p27/`.

Giving `sim/engine.py` optimistic admission with chunked prefill is still the larger fix,
and it is now *less* urgent rather than more: the effect it was invoked to explain -- a
gap that depends on admission width -- turned out not to be there. Do step 1 before
rebuilding anything.

## The methodological point worth keeping

This validation produced three false findings before producing a true one. The first two
were caught the same way — by asking what the measurement apparatus would report if the
system under test were perfect. The third needed a different question, and a cheaper one.

1. The first run reported an 8.1 pp disagreement caused entirely by the harness: prompts
   were built as `"word " * n`, which made all forty sessions byte-identical and mutual
   prefixes of one another, so vLLM served every session from every other session's cache
   and reported 99.6%. **Giving the simulator an unlimited pool** did not move its hit
   rate, which proved eviction was not involved and pointed straight at the harness.
2. The second run reported a 25 pp disagreement and a 3.2x eviction ratio, and it was written
   up as a refutation of the eviction model. It was an accounting difference. **Comparing
   the query count to the number of prompt tokens sent** — a ratio that must be 1.000 if
   the counters mean what they appear to mean — settles it in one line, and the same
   unlimited-pool ceiling shows the reported hit count exceeding the maximum possible.
   The redone measurement put the real disagreement at 1.4 pp, in the other direction.
3. The third finding was a gap that depended on admission width — 4.8 pp against 10.9 pp
   at the same pressure — which no single pressure offset can produce, so it was written
   up as evidence that the admission model is what differs. It came from one seed per
   point and did not survive four: the width effect is −1.2 pp, 95% [−6.1, +1.4].
   **Run the same thing again with a different seed before explaining it.** The
   explanation was constructed, was mechanistically plausible, and described noise.

The first two checks are denominator-level sanity rather than domain insight, and the
third is not even that — it is just repetition. All three are cheap. In all three cases
the wrong answer was more interesting than the right one, which is exactly why it got
written up, and why the habit that catches these has to be mechanical rather than a matter
of being suspicious enough on the day.

`tests/test_invariants.py::test_query_accounting_matches_vllms_counters` pins the
relationship between the two hit-rate definitions so the second mistake cannot recur
silently. There is no equivalent guard against the third, because no test can tell you
that a number came from one seed — only the discipline of not explaining an effect until
it has an interval.

## What is not established here

- Thirteen runs, one model, one workload generator. The pressure-1.27 comparison has
  four paired seeds at cap 6 and three at cap 8; **every other point in this document
  is still a single seed without an interval**, including both of the agreements that
  the validation rests on.
- Three pressures, and the boundary between the two regimes is located only to
  somewhere in (1.08, 1.27).
- All comparisons above pressure 1.0 use *matched* admission. Under vLLM's default
  admission the hit rate is still not measured at any pressure above 1.0, and the
  pressure-1.02 attempt bounds it only to [0.238, 0.932].
- What vLLM's prefix-cache counters increment on is inferred from their behaviour, not
  read from its source. "Scheduling attempts" fits the evidence; it has not been
  confirmed against the implementation.
- The longest contexts (30k–50k tokens) are excluded by the context-window truncation,
  and those are exactly the contexts where eviction pressure is highest.
- No attempt was made to change the simulator to match vLLM. The admission difference is
  now documented and its direction is known; closing it should be driven by the corrected
  measurement, not by tuning until the numbers agree — a simulator tuned to reproduce one
  measurement is not validated by it.
