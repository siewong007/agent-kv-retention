# HPC calibration round

Every engine constant in `sim/config.py` was fitted on a local RTX 5080 under WSL2, and
[docs/calibration.md](../docs/calibration.md) records that some of them are WSL-specific
and do not transfer — no `nvcc` in the container inflates `step_overhead_s`, and a Windows
desktop holding ~1.3 GiB changes the KV pool size. The project's first rule is that every
number in the paper comes from one platform and one round.

So there is a decision to make before the report is written, not after: **does the thesis
report 5080/WSL2 numbers, or HPC numbers?** If HPC, this round has to run and the five
experiments have to be re-run on its constants. The experiments themselves are CPU-only,
so only this round needs the GPU.

## What to fill in before submitting

Four values, none of which the scripts will guess:

| variable | what it is | how to find it |
|---|---|---|
| `PROJECT` | checkout path on the HPC filesystem | wherever you cloned it |
| `VENV` | virtualenv holding torch + vllm | build it once on a login node |
| `EXPECT_CAPABILITY` | the node's compute capability | `7.5` for T4 (Turing), `8.9` for L4/L40S (Ada) |
| `RM_PER_HOUR` | list price of that node | `3.06` for T4; see the cost table in [calibration.md](../docs/calibration.md) |

Plus, inside `calibrate.sbatch`, the two commented `##SBATCH` lines for partition and
account. Check `sinfo` and `sacctmgr show assoc user=$USER` rather than guessing —
a wrong partition name is rejected at submit time and costs nothing, a wrong account
charges someone else's budget.

`EXPECT_CAPABILITY` is asserted, not recorded. If the node is not the architecture you
said, the job aborts before measuring anything, because a constant fitted on an unexpected
architecture is worse than no constant.

## Submitting

```bash
sbatch --export=ALL,PROJECT=$HOME/agent-kv-retention,VENV=$HOME/venv-vllm,EXPECT_CAPABILITY=7.5,RM_PER_HOUR=3.06 hpc/calibrate.sbatch
```

On a PBS or LSF site, replace the header in `calibrate.sbatch` and call the same body;
`hpc/run_calibration.sh` knows nothing about any scheduler.

## What it costs

Billing is on wall clock, not GPU utilisation, so the only thing that matters is that the
job stops the moment it stops measuring. It is a single batch job with no interactive
step, and `run_calibration.sh` kills vLLM from an `EXIT` trap on every path including
crashes — the expensive failure is a job that dies and leaves the server holding the node.

| step | roughly |
|---|---|
| environment check | 1 min |
| calibration server + `fit_timing` + `fit_batch` | ~1 h |
| validation at pressure ~0.64 | ~15 min |
| validation above pressure 1.0, matched admission | ~25 min |
| **total** | **~2.5 h ≈ RM 8 on a T4** |

`--time=03:00:00` is a ceiling, not a booking. Asking for too little is the expensive
mistake: a job killed mid-validation has spent the money and produced nothing.

The measured elapsed time and its cost are printed at the end and written into
`results/hpc/manifest.json`, so the round's cost is on record rather than remembered.

## What it produces

```
results/hpc/env.json            the platform, asserted rather than assumed
results/hpc/calib/              timing and batch fits
results/hpc/validate/           behaviour check at pressure ~0.64
results/hpc/validate_matched/   behaviour check above pressure 1.0
results/hpc/manifest.json       platform, job, git commit, elapsed, cost
```

Everything lands under `results/hpc/` so that a stray path cannot mix the two rounds.

## Afterwards — deliberately not automatic

The constants do not take effect by existing.

1. Diff `results/hpc/calib/` against the constants in `sim/config.py`.
2. If adopting them, edit `sim/config.py` **in its own commit that says so**, then re-run
   all five experiments (CPU, no GPU time).
3. Never plot the two rounds together.

`manifest.json` carries `adopted_into_sim_config: false`, hardcoded, so that a
half-finished adoption is visible rather than silent.

## Why validation is in this round and not left for later

The calibration fits four constants; the validation checks whether the simulator *behaves*
like the server those constants came from. On the local box those two answers came apart —
the constants fitted fine while the behaviour agreed only up to about pressure 1.1 (see
[validation_findings.md](../docs/validation_findings.md)). Whether that boundary is a
property of the model or of the 5080/WSL2 setup is unknown, and it is one server start
away from being answered on a second platform. Running it in the same job costs ~40
minutes and one queue wait instead of two.
