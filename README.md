# KV cache retention for agent workloads

Experiment code for an MSc thesis on inference serving for LLM agent workflows:
how KV cache should be retained across the pauses in an agent loop, and what that is
actually worth in ringgit rather than in hit rate.

## The question

An agent task issues 20-30 LLM calls that share a growing prefix, separated by pauses
while tools run. Current engines keep prefix-cache blocks only until something else
needs the space, so a session that pauses long enough comes back to a cold cache and
recomputes its whole context.

Predicting agent behaviour is a live topic (AutoTool, PASTE); managing KV cache is a
live topic (Continuum, PBKV, KVFlow). PBKV explicitly leaves the predictor pluggable
and does not build a good one. This repo is an attempt to find out whether wiring the
two together is worth doing -- **starting with the experiment most likely to prove it
is not.**

## Where this is

**Start here: [docs/SUMMARY.md](docs/SUMMARY.md)** — every current number with its
interval, what is not established, and where each result lives.

Week 0. A CPU simulator, five experiments, and one calibration pass against real vLLM.
All experiment numbers below were **rerun after calibration**; the pre-calibration runs
are kept in `results/exp0*` and should not be quoted.

| experiment | question | verdict |
|---|---|---|
| [EXP01](docs/exp01_findings.md) | is predictive retention worth anything a tuned constant cannot get? | yes — a constant TTL gets exactly 0% of it. Peak headroom **13.7% [12.5, 15.0]**, of which **73.6% [66.0, 81.2]** comes from knowing the session ended (100 seeds) |
| [EXP02](docs/exp02_findings.md) | does the result belong to the hardware or to the memory pressure? | pressure is necessary but **not sufficient**: it transfers across a 4x change in pool size, but not across a 2.5x change in session count |
| [EXP03](docs/exp03_findings.md) | was EXP01's pause sweep confounded by falling load? | half of it was the billing model; open-loop turns out to be metastable |
| [EXP04](docs/exp04_findings.md) | how much headroom does a real predictor capture? | 49.6% [20.9, 67.2] at the strongest signal tested, and indistinguishable from LRU at the other three. Ranking by predicted *pause length* is catastrophic at any accuracy |
| [EXP05](docs/exp05_findings.md) | where should the classifier's decision threshold sit? | it *is* the policy. False positives are the whole cost; at weak signal no threshold beats LRU, at strong signal five of six do |
| [calibration](docs/calibration.md) | were the derived constants any good? | no — all wrong by 20–58%, and one whole term was missing |
| [validation](docs/validation_findings.md) | does the simulator behave like vLLM? | **up to about pressure 1.1, yes** — 2% on makespan, 1.4 pp on hit rate. At 1.27 it is 4.8–10.9 pp pessimistic and 11–19% slow |

## Layout

```
sim/          the simulator. stdlib only, no GPU, deterministic under a seed
  config.py     every knob; a result file carries the config that produced it
  workload.py   synthetic agent trace generator, independent of any policy
  cache.py      block pool + prefix cache + the two eviction mechanisms
  engine.py     continuous-batching step loop, chunked prefill, preemption
  run.py        single-run CLI
experiments/  sweeps that answer one question each, plus their figures
bench/        GPU-side calibration: env check, vLLM launch, timing and batch fits
hpc/          an HPC round, written then NOT used -- see the decision in hpc/README.md
tests/        invariants the simulator must hold for its numbers to mean anything
docs/         findings, and the calibration ledger of measured vs invented constants
results/      run outputs, each with full config + seed + environment
```

## Run it

```bash
python -m tests.test_invariants
```

```bash
python -m sim.run --config configs/base.json --set policy.kind=belady
```

```bash
python -m experiments.exp01_ttl_falsify --sessions 200 --seeds 100 --concurrency 8,10,12,14,16,18 --pause "" --arms lru,ttl_oracle,oracle_terminal,belady --out results/exp01_share_seeds100
```

```bash
python -m experiments.exp02_pressure_axis --sessions 200 --seeds 15 --out results/v2_exp02_seeds15
```

```bash
python -m experiments.exp03_pause_isolation --sessions 200 --seeds 15 --out results/v3_exp03
```

```bash
python -m experiments.exp04_predictor --sessions 200 --seeds 15 --out results/exp04_seeds15
```

```bash
python -m experiments.exp05_threshold --sessions 200 --seeds 15 --out results/exp05_seeds15
```

Add `--reanalyze` to any of them to redo the analysis over an existing `runs.csv`
without simulating anything, then `python -m experiments.plot_exp0N --results <dir>`
for the figure.

Before buying more seeds, check whether they would help:

```bash
python -m experiments.seed_sufficiency --results results/v2_exp01_seeds15 --exp exp01
```

It fits how each interval actually shrinks under subsampling and projects the seeds
needed for a target precision. A decay much shallower than n^-0.5 means the width is
held up by something seeds cannot remove.

To recalibrate against real hardware (needs the GPU; releases it when done):

```bash
bash bench/serve_calib.sh && python bench/check_env.py && python bench/read_server_config.py ~/vllm_calib_server.log && python -m bench.fit_timing && python -m bench.fit_batch
```

To re-validate the simulator's behaviour against vLLM (needs the GPU; ~90 min, releases it
when done). The sweep pins the KV pool so admission width is the only thing varying:

```bash
bash bench/sweep_admission.sh && python -m bench.analyze_admission_sweep
```

## Policy arms

Two mechanisms, so that "better information" and "better mechanism" never get confused
with each other. Only the oracle arms may look at the future.

| arm | mechanism | information used | role |
|---|---|---|---|
| `lru` | TTL = 0 | none | the incumbent: what vLLM does today |
| `const_ttl` | uniform TTL | none | Continuum-shaped tuned constant |
| `lru_priority` | priority | none | validation: must equal `lru` exactly |
| `ttl_oracle` | TTL = true pause | oracle | ablation: right information, wrong mechanism |
| `oracle_terminal` | priority | oracle, session-ended only | isolates termination prediction |
| `belady` | priority | oracle, full | strong oracle reference. **Not** an upper bound: it is myopic and the reference stream is not fixed, see [EXP04](docs/exp04_findings.md) |
| `predict` | priority | learned | the deployable version of `belady` |
| `predict_terminal` | priority | learned, session-ended only | the deployable version of `oracle_terminal`; the only predict arm that ever measurably beats LRU, and only at ~0.89 precision |
| `predict_guarded` | priority | learned | ranks by predicted pause alone, ties break LRU |
| `belady_pause` | priority | oracle, pause only | oracle counterpart of `predict_guarded`; beats `belady` |

## Reading the numbers

Seven rules, all load-bearing:

1. **A cache win is not a cost win.** Conversion runs about 20–60%: a 35% cut in
   recomputed tokens is worth roughly 7% of cost. Always check `gpu_busy_frac` before
   quoting a ringgit figure.
2. **Say which billing model.** `rm_per_1k_calls` charges wall clock (a reserved box,
   which is how the Sunway HPC session is billed). `rm_gputime_per_1k_calls` charges
   only seconds the GPU worked (shared or autoscaled). They disagree, and under
   open-loop arrivals the wall-clock one cannot move at all because the makespan is
   pinned by the arrival schedule.
3. **Report pressure AND session count.** `pressure = live sessions x context blocks /
   pool blocks` transfers across pool size, but not across session count: at equal
   pressure, 48 sessions show less than half the headroom of 19. Splitting the same
   deficit more ways behaves like higher pressure. See
   [EXP02](docs/exp02_findings.md).
4. **Never quote a point estimate without its interval.** Every experiment reports 95%
   paired-bootstrap intervals over seeds. Every headline in this project that was once
   quoted without one later turned out to be wrong — see the corrections in
   [EXP02](docs/exp02_findings.md), [EXP04](docs/exp04_findings.md) and
   [EXP05](docs/exp05_findings.md). Proving that one arm *beats* another is cheap;
   estimating *by how much* is expensive, and some quantities here would need 300+ seeds
   to pin down. `experiments/seed_sufficiency.py` says which is which.
5. **LRU is not a naive baseline.** Age correlates with termination on this workload,
   so LRU is already exploiting a real signal. Any arm that overrides its ordering must
   beat that signal, and a noisy predictor does not — see [EXP04](docs/exp04_findings.md).
   The corollary is that a predictor's *decision threshold* is not a hyperparameter, it
   is the policy: see [EXP05](docs/exp05_findings.md).
6. **The simulator is validated to about pressure 1.1, not beyond.** Against real vLLM it
   holds makespan to 2% and hit rate to 1.4 pp at pressures 0.64 and 1.08; at 1.27 it runs
   4.8–10.9 pp pessimistic on hit rate and 11–19% slow. Peak headroom (pressure 0.84) is
   inside that range. EXP02's high-pressure tail is not, and the collapse it shows is
   probably exaggerated — quote it as a property of the simulator. See
   [validation](docs/validation_findings.md).
7. **Engine constants are measured; workload constants are not.** See
   [docs/calibration.md](docs/calibration.md) for the row-by-row ledger. The pause and
   tool-result distributions are still invented, and they are what every headroom figure
   is a function of.

## Platform

Every number in this repository comes from one machine: an **RTX 5080 (Blackwell, sm_120)
under WSL2**, with Qwen2.5-3B-Instruct on vLLM 0.26. That is a deliberate choice made on
2026-08-19 rather than a limitation nobody got around to fixing — an HPC round was
scripted (`hpc/`) and then not run, because one platform satisfies the
never-mix-two-platforms rule more simply than two do.

What it costs is that two WSL2-specific effects are properties of the reported numbers
rather than things a second platform would have averaged away:

- `step_overhead_s = 9.554 ms` includes a native-sampler penalty, because the torch wheel
  ships the CUDA runtime without `nvcc` and FlashInfer cannot JIT its kernels. A machine
  with a full toolkit would measure less.
- The KV pool is what remains after a Windows desktop holds ~1.3 GiB.

Both are in [docs/calibration.md](docs/calibration.md). Quote the constants as belonging
to this setup, not to the card.

## License

MIT — see [LICENSE](LICENSE).
