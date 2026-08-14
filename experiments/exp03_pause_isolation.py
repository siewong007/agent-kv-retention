"""EXP03 -- separating "the pause got longer" from "the load got lighter".

EXP01's pause sweep ran closed-loop at fixed concurrency. In that setup a longer tool
pause also means a lower arrival rate, so GPU utilisation fell from 100% to 45% across
the sweep. Every latency and cost number in that figure is a mix of two effects pushing
in opposite directions: the cache gets colder (worse) while the machine gets emptier
(better). The conclusion drawn from it -- that a cache win stops being a cost win when
the GPU idles -- is exactly the kind of claim that confound could have manufactured.

Two things have to be said about what can and cannot be controlled here.

**Utilisation is not a valid control variable in this system.** A cache miss creates
GPU work, so pinning "GPU busy fraction" partly pins the very waste being measured: a
thrashing configuration reaches 99% utilisation *because* it is recomputing. Calibrating
the arrival rate to a target utilisation would therefore build the answer into the
experiment. That was this file's first design and it was wrong.

**Pause length and utilisation are coupled by arithmetic, not by an artefact.** With C
live sessions,

    pressure    ~ C x context_blocks / pool                  (independent of pause)
    utilisation ~ C x service_per_turn / (service + pause)   (falls as pause grows)

At a 30 s median pause, tool time is ~97% of a session's wall clock. Cost is billed on
wall clock, so *no* retention policy can touch 97% of the bill. The near-zero cost
headroom at long pauses is a bound, not a measurement, and no experimental design
removes it. It should be reported as arithmetic.

What is left worth testing is whether the closed loop's own feedback -- a finished
session instantly replaced by a new one, and a hard cap on concurrency -- shaped the
EXP01 conclusions. So:

    closed          fixed concurrency 16 (what EXP01 did)
    open_matched    Poisson arrivals, rate calibrated per pause so that the *mean live
                    session count* matches 16. Same pressure, but sessions arrive in
                    bursts, concurrency is unbounded, and nothing is replaced on
                    completion.

If the findings hold in both, they are not artefacts of closed-loop scheduling.

    python -m experiments.exp03_pause_isolation --seeds 15 --out results/v3_exp03
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    bootstrap_ratio, default_workers, read_results, run_grid, run_job, write_results,
)
from sim.config import Config  # noqa: E402

PAUSE_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
ARMS = ["lru", "oracle_terminal", "belady"]
CLOSED_CONCURRENCY = 16
# The open-loop arm is calibrated to a mean live-session count, not to a utilisation.
# Live sessions set memory pressure directly and are not inflated by cache misses, so
# unlike utilisation they are a control variable rather than a partial outcome.
TARGET_LIVE = float(CLOSED_CONCURRENCY)
LIVE_TOLERANCE = 0.4
# A run whose live-session count exceeds this multiple of the target did not settle at
# the offered load; it ran away. Used only as a diagnostic, never to drop data.
RUNAWAY_MULTIPLE = 2.0
CALIBRATION_SEED = 0
# Calibration must use the SAME session count as the real runs. Mean live sessions
# includes the ramp-up and drain, whose share depends on how long the run is, so
# calibrating on a shorter workload lands on a different rate than it reports.
MAX_CALIBRATION_REFINEMENTS = 5


def calibrate_rate(base: Config, pause: float, n_sessions: int) -> dict:
    """Bisect the Poisson arrival rate until the mean live-session count hits TARGET_LIVE.

    Uses one fixed seed and a shorter workload: the rate is an experiment *setting*, not
    a result, so it must be reproducible but need not be precise to three decimals. The
    calibrated value is written into the metadata so the whole sweep can be rebuilt.
    """
    cfg = base.replace(**{
        "policy.kind": "lru",
        "arrival.mode": "poisson",
        "workload.pause_seconds_median": pause,
        "workload.n_sessions": n_sessions,
        "seed": CALIBRATION_SEED,
    })

    def live_at(rate: float) -> float:
        row = run_job({"config": cfg.replace(**{"arrival.rate_per_s": rate}).to_dict()})
        return row["mean_live_sessions"]

    # Starting point from Little's law rather than a blind search. A closed-loop run at
    # the target concurrency completes sessions at exactly the rate that sustains that
    # many in flight, so its completion rate IS the arrival rate we want. Bisecting from
    # an arbitrary bracket instead spends most of its runs at absurdly high rates, where
    # the simulator thrashes and each run costs minutes.
    seed_row = run_job({"config": cfg.replace(**{
        "arrival.mode": "closed",
        "arrival.concurrency": int(TARGET_LIVE),
    }).to_dict()})
    rate = seed_row["n_sessions"] / seed_row["makespan_s"]

    # Refine multiplicatively: live sessions are near-proportional to the arrival rate,
    # so target/live is a good step. Bounded, and never explores upward blindly.
    trace = []
    best = (float("inf"), rate, None)
    for _ in range(MAX_CALIBRATION_REFINEMENTS):
        live = live_at(rate)
        trace.append({"rate": rate, "live": live})
        if abs(live - TARGET_LIVE) < best[0]:
            best = (abs(live - TARGET_LIVE), rate, live)
        if abs(live - TARGET_LIVE) <= LIVE_TOLERANCE:
            break
        # Clamp the correction so one bad sample cannot send the next run into thrash.
        rate *= min(1.6, max(0.5, TARGET_LIVE / live)) if live > 0 else 1.6

    _, rate, live = best
    matched = abs(live - TARGET_LIVE) <= LIVE_TOLERANCE
    note = "" if matched else "   NOT MATCHED: target unreachable in open loop"
    print(f"  pause {pause:>5.1f}s -> rate {rate:.5f}/s  "
          f"(mean live {live:.2f}, target {TARGET_LIVE:.0f}){note}", flush=True)
    return {"pause_median_s": pause, "rate_per_s": rate, "calibrated_live": live,
            "target_live": TARGET_LIVE, "matched": matched, "trace": trace}


def build_jobs(base: Config, seeds: list[int], n_sessions: int,
               rates: dict[float, float]) -> list[dict]:
    jobs = []
    for pause in PAUSE_GRID:
        for seed in seeds:
            for arm in ARMS:
                closed = base.replace(seed=seed, **{
                    "policy.kind": arm,
                    "arrival.mode": "closed",
                    "arrival.concurrency": CLOSED_CONCURRENCY,
                    "workload.pause_seconds_median": pause,
                    "workload.n_sessions": n_sessions,
                })
                jobs.append({"config": closed.to_dict(),
                             "labels": {"loop": "closed"}})

                openl = base.replace(seed=seed, **{
                    "policy.kind": arm,
                    "arrival.mode": "poisson",
                    "arrival.rate_per_s": rates[pause],
                    "workload.pause_seconds_median": pause,
                    "workload.n_sessions": n_sessions,
                })
                jobs.append({"config": openl.to_dict(),
                             "labels": {"loop": "open_matched"}})
    return jobs


def analyze(rows: list[dict], out_dir: str) -> dict:
    points: dict = {}
    for r in rows:
        points.setdefault((r["loop"], r["pause_median_s"]), []).append(r)

    out = []
    for (loop, pause), group in sorted(points.items()):
        def mean_of(policy, metric):
            vals = [r[metric] for r in group if r["policy"] == policy]
            return st.fmean(vals) if vals else None

        # Near saturation an open-loop run is metastable: the same offered load either
        # settles or runs away depending on the arrival realisation, so the seeds form a
        # bimodal mixture and their mean is not a central tendency of anything. Flag it
        # rather than average through it.
        lru_live = [r["mean_live_sessions"] for r in group if r["policy"] == "lru"]
        runaway = [v for v in lru_live if v > RUNAWAY_MULTIPLE * TARGET_LIVE]
        entry = {"loop": loop, "pause_median_s": pause,
                 "n_seeds": len({r["seed"] for r in group}),
                 "gpu_busy_frac": mean_of("lru", "gpu_busy_frac"),
                 "pressure_measured": st.fmean([r["pressure_measured"] for r in group]),
                 "mean_live_sessions": st.fmean([r["mean_live_sessions"] for r in group]),
                 "median_live_sessions": st.median(lru_live) if lru_live else None,
                 "runaway_frac": len(runaway) / len(lru_live) if lru_live else 0.0,
                 "live_min": min(lru_live) if lru_live else None,
                 "live_max": max(lru_live) if lru_live else None,
                 "hit_rate_lru": mean_of("lru", "token_hit_rate")}
        def by_seed(policy, metric):
            return {r["seed"]: r[metric] for r in group if r["policy"] == policy}

        for metric in ("prefill_tokens_computed", "rm_per_1k_calls",
                       "rm_gputime_per_1k_calls", "ttft_p95"):
            lru = mean_of("lru", metric)
            bel = mean_of("belady", metric)
            term = mean_of("oracle_terminal", metric)

            # Paired over seeds. In the open-loop arm the seeds are a bimodal mixture of
            # runs that settled and runs that ran away, so the interval there is wide by
            # construction -- and that width is the honest report of what a mixture
            # supports, not a defect of the estimator.
            lru_s, bel_s = by_seed("lru", metric), by_seed("belady", metric)
            paired = sorted(set(lru_s) & set(bel_s))
            headroom_ci = bootstrap_ratio(
                [(lru_s[sd], bel_s[sd]) for sd in paired],
                [(lru_s[sd], 0.0) for sd in paired]) if paired else None

            entry[metric] = {
                "lru": lru, "belady": bel, "oracle_terminal": term,
                "headroom_frac": (lru - bel) / lru if lru else 0.0,
                "headroom_ci": headroom_ci,
                "terminal_share": ((lru - term) / (lru - bel))
                                  if lru and lru > bel else None,
            }
        out.append(entry)

    report = {"points": out}
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def print_report(report: dict) -> None:
    print()
    print("=" * 106)
    print("EXP03  the same pause sweep with utilisation free (closed) and pinned (open)")
    print("=" * 106)
    for loop, note in (
            ("closed", f"fixed concurrency {CLOSED_CONCURRENCY}, sessions replaced on completion"),
            ("open_matched", f"Poisson arrivals calibrated to {TARGET_LIVE:.0f} mean live "
                             f"sessions: same pressure, bursty, no replacement")):
        pts = sorted([p for p in report["points"] if p["loop"] == loop],
                     key=lambda p: p["pause_median_s"])
        if not pts:
            continue
        print(f"\n  {loop} loop -- {note}")
        print(f"  {'pause':>6} {'busy':>6} {'press':>6} {'live':>6} {'runaway':>8} "
              f"{'hit':>6} | {'tokens head% [95% CI]':>24} "
              f"{'RM wall%':>10} {'RM gpu%':>9} {'TTFT%':>8}")
        print("  " + "-" * 108)
        for p in pts:
            tok = p["prefill_tokens_computed"]
            ci = tok.get("headroom_ci") or {}
            if ci.get("lo") is None:
                tok_txt = f"{100*tok['headroom_frac']:>10.1f}%"
            else:
                sig = " " if (ci["lo"] <= 0 <= ci["hi"]) else "*"
                tok_txt = (f"{100*tok['headroom_frac']:>7.1f}% "
                           f"[{100*ci['lo']:>5.1f},{100*ci['hi']:>5.1f}]{sig}")
            print(f"  {p['pause_median_s']:>6.1f} {p['gpu_busy_frac']:>6.3f} "
                  f"{p['pressure_measured']:>6.2f} {p['median_live_sessions']:>6.1f} "
                  f"{100*p['runaway_frac']:>7.0f}% "
                  f"{p['hit_rate_lru']:>6.3f} | {tok_txt:>24} "
                  f"{100*p['rm_per_1k_calls']['headroom_frac']:>9.1f}% "
                  f"{100*p['rm_gputime_per_1k_calls']['headroom_frac']:>8.1f}% "
                  f"{100*p['ttft_p95']['headroom_frac']:>7.1f}%")
    print("\n'live' is the median over seeds. Where runaway is non-zero the seeds are a "
          "bimodal\nmixture -- some settled at the offered load and some did not -- so "
          "no single number\nsummarises that row and it must not be compared against "
          "the closed-loop block. The\nwide intervals on those rows are the honest "
          "consequence, not an estimator defect.")
    print("\n'*' marks a token headroom whose 95% interval excludes zero. Intervals are "
          "paired\nbootstrap over seeds, 5000 resamples, seeded.")
    print("\n'RM wall%' bills wall clock (reserved box). 'RM gpu%' bills only seconds "
          "the GPU worked\n(shared/autoscaled). Where they diverge, the conclusion "
          "depends on the billing model,\nnot on the cache.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--out", default="results/exp03")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--reanalyze", action="store_true")
    args = ap.parse_args(argv)

    if args.reanalyze:
        print_report(analyze(read_results(args.out), args.out))
        return 0

    base = Config.load(args.config)
    os.makedirs(args.out, exist_ok=True)

    print(f"calibrating open-loop arrival rates to {TARGET_LIVE:.0f} mean live sessions "
          f"(seed {CALIBRATION_SEED}, {args.sessions} sessions)")
    calibration = [calibrate_rate(base, pause, args.sessions) for pause in PAUSE_GRID]
    rates = {c["pause_median_s"]: c["rate_per_s"] for c in calibration}
    with open(os.path.join(args.out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    seeds = list(range(args.seeds))
    rows = run_grid(build_jobs(base, seeds, args.sessions, rates), args.workers, "EXP03")
    csv_path = write_results(args.out, rows, base, {
        "experiment": "exp03_pause_isolation",
        "seeds": seeds,
        "pause_grid": PAUSE_GRID,
        "arms": ARMS,
        "closed_concurrency": CLOSED_CONCURRENCY,
        "target_live": TARGET_LIVE,
        "calibrated_rates": rates,
        "calibration": calibration,
        "calibration_seed": CALIBRATION_SEED,
        "max_calibration_refinements": MAX_CALIBRATION_REFINEMENTS,
    })
    print_report(analyze(rows, args.out))
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
