"""EXP01 -- does perfect knowledge of agent pauses buy anything a tuned constant cannot?

This experiment exists to kill the project early if the answer is no. It runs, on
byte-identical workloads:

    lru          the incumbent (what vLLM does today)
    const_ttl    a tuned constant TTL, swept over a grid (Continuum-shaped)
    ttl_oracle   the true pause used as a TTL -- perfect information, naive mechanism
    belady       the true next-use time used as an eviction priority. A strong
                 oracle reference, NOT an upper bound: see docs/exp04_findings.md --
                 it is myopic and the reference stream is not fixed

and reports the headroom `belady - lru` together with the fraction of it that the best
constant TTL already captures.

Decision rule, fixed before looking at the numbers:
    headroom < 10%  -> prediction cannot pay for itself; drop the direction.
    best const captures > 70% of headroom -> prediction is not the interesting variable;
                                             the story is mechanism, not prediction.
    headroom > 30% and const captures < 50% -> the premise survives; build a predictor.

Usage:
    python -m experiments.exp01_ttl_falsify --sessions 200 --seeds 100 --concurrency 8,10,12,14,16,18 --pause "" --arms lru,ttl_oracle,oracle_terminal,belady --out results/exp01_share_seeds100
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import bootstrap_ratio as _bootstrap_ratio  # noqa: E402
from sim.config import Config  # noqa: E402
from sim.engine import run_config  # noqa: E402
from sim.run import build_metadata  # noqa: E402
from sim.workload import generate_sessions, workload_summary  # noqa: E402


CONST_TTL_GRID = [1.0, 4.0, 16.0, 64.0, 256.0, 1e9]
CONCURRENCY_GRID = [4, 8, 12, 14, 16, 18, 20, 24, 32]
PAUSE_GRID = [0.5, 2.0, 10.0, 30.0]
# The pause sweep is run inside the pressured regime; at low concurrency every policy
# ties on a ceiling and the sweep says nothing.
PAUSE_SWEEP_CONCURRENCY = 16

# Below this, the oracle is indistinguishable from the incumbent and any "share of the
# headroom" is a ratio of two noise terms. Reported as missing rather than as a number.
MIN_RELIABLE_HEADROOM = 0.02

# Reported per run. Everything else lives in the per-run summary.json.
METRIC_KEYS = [
    "token_hit_rate", "prefill_tokens_computed", "prompt_tokens_total",
    "ttft_p50", "ttft_p95", "e2e_p50", "e2e_p95", "queue_p95",
    "makespan_s", "gpu_busy_s", "gpu_busy_frac",
    "n_preemptions", "n_evictions", "n_protected_evictions",
    "rm_total", "rm_per_1k_calls", "n_calls",
]


def arms(only: list[str] | None = None) -> list[dict]:
    """All arms, or a named subset.

    A subset is worth having because `const_ttl` is bit-identical to `lru` at every TTL
    (proved, and guarded by tests/test_invariants.py). Re-running six TTL values at a
    high seed count buys nothing, and the arms that are actually expensive to resolve --
    the termination share needs hundreds of seeds -- are only three of the ten.
    """
    out = [{"policy.kind": "lru"}]
    out += [{"policy.kind": "const_ttl", "policy.const_ttl_s": t} for t in CONST_TTL_GRID]
    out += [{"policy.kind": "ttl_oracle"},
            {"policy.kind": "oracle_terminal"},
            {"policy.kind": "belady"}]
    if only:
        out = [a for a in out if a["policy.kind"] in only]
    return out


def _run_one(job: dict) -> dict:
    cfg = Config.from_dict(job["config"])
    sessions = generate_sessions(cfg.workload, cfg.seed)
    summary = run_config(cfg, sessions).summary
    row = {
        "sweep": job["sweep"],
        "seed": cfg.seed,
        "policy": cfg.policy.kind,
        "const_ttl_s": cfg.policy.const_ttl_s,
        "concurrency": cfg.arrival.concurrency,
        "pause_median_s": cfg.workload.pause_seconds_median,
        "pool_blocks": cfg.engine.kv_pool_blocks,
    }
    row.update({k: summary[k] for k in METRIC_KEYS})
    return row


def build_jobs(base: Config, seeds: list[int],
               conc_grid: list[int], pause_grid: list[float],
               only_arms: list[str] | None = None) -> list[dict]:
    jobs = []
    for seed, arm in itertools.product(seeds, arms(only_arms)):
        for conc in conc_grid:
            cfg = base.replace(seed=seed, **{**arm, "arrival.concurrency": conc})
            jobs.append({"sweep": "concurrency", "config": cfg.to_dict()})
        for pause in pause_grid:
            cfg = base.replace(seed=seed, **{
                **arm,
                "workload.pause_seconds_median": pause,
                "arrival.concurrency": PAUSE_SWEEP_CONCURRENCY,
            })
            jobs.append({"sweep": "pause", "config": cfg.to_dict()})
    return jobs


def analyze(rows: list[dict], out_dir: str) -> dict:
    """Headroom per sweep point, on the two metrics that matter."""
    import statistics as st

    def group_key(r):
        return (r["sweep"], r["concurrency"], r["pause_median_s"])

    points: dict = {}
    for r in rows:
        points.setdefault(group_key(r), []).append(r)

    verdicts = []
    for key, group in sorted(points.items()):
        sweep, conc, pause = key

        def by_seed(policy, metric, ttl=None):
            return {r["seed"]: r[metric] for r in group
                    if r["policy"] == policy and (ttl is None or r["const_ttl_s"] == ttl)}

        def mean_of(policy, metric, ttl=None):
            vals = list(by_seed(policy, metric, ttl).values())
            return st.fmean(vals) if vals else None

        seeds = sorted({r["seed"] for r in group})
        entry = {"sweep": sweep, "concurrency": conc, "pause_median_s": pause,
                 "n_seeds": len(seeds)}
        for metric in ("prefill_tokens_computed", "rm_per_1k_calls", "ttft_p95",
                       "gpu_busy_frac", "token_hit_rate"):
            lru = mean_of("lru", metric)
            bel = mean_of("belady", metric)
            ttl_or = mean_of("ttl_oracle", metric)
            term = mean_of("oracle_terminal", metric)
            consts = {t: mean_of("const_ttl", metric, t) for t in CONST_TTL_GRID}
            consts = {t: v for t, v in consts.items() if v is not None}
            best_ttl = min(consts, key=consts.get) if consts else None
            best_const = consts[best_ttl] if best_ttl is not None else None

            def frac(v):
                return (lru - v) / lru if lru and v is not None else 0.0

            headroom = frac(bel)
            # A share of a headroom that is itself within noise is not a number, it is
            # a division by zero wearing a percent sign. Report it as missing.
            reliable = headroom >= MIN_RELIABLE_HEADROOM
            share = lambda g: (g / headroom) if reliable else float("nan")  # noqa: E731
            entry[metric] = {
                "lru": lru,
                "belady": bel,
                "headroom_reliable": reliable,
                "ttl_oracle": ttl_or,
                "oracle_terminal": term,
                "best_const_ttl_s": best_ttl,
                "best_const": best_const,
                "headroom_frac": headroom,
                "const_gain_frac": frac(best_const),
                "const_captured_frac_of_headroom": share(frac(best_const)),
                "terminal_gain_frac": frac(term),
                "terminal_captured_frac_of_headroom": share(frac(term)),
                "ttl_oracle_gain_frac": frac(ttl_or),
                "const_ttl_curve": consts,
            }

            # Per-seed spread. The point estimates above are ratios of small
            # differences; without this block there is no way to tell a real effect
            # from three lucky seeds, and the seed-to-seed spread of the terminal
            # share turned out to be wider than the effect itself at 3 seeds.
            lru_s = by_seed("lru", metric)
            bel_s = by_seed("belady", metric)
            term_s = by_seed("oracle_terminal", metric)
            head_seeds, share_seeds = [], []
            for sd in seeds:
                if sd not in lru_s or sd not in bel_s or not lru_s[sd]:
                    continue
                gap = lru_s[sd] - bel_s[sd]
                head_seeds.append(gap / lru_s[sd])
                if sd in term_s and gap / lru_s[sd] >= MIN_RELIABLE_HEADROOM:
                    share_seeds.append((lru_s[sd] - term_s[sd]) / gap)

            def spread(values):
                if not values:
                    return None
                return {
                    "n": len(values),
                    "mean": st.fmean(values),
                    "std": st.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "values": values,
                }

            entry[metric]["headroom_by_seed"] = spread(head_seeds)
            entry[metric]["terminal_share_by_seed"] = spread(share_seeds)

            paired = [sd for sd in seeds if sd in lru_s and sd in bel_s and sd in term_s]
            lru_bel = [(lru_s[sd], bel_s[sd]) for sd in paired]
            lru_term = [(lru_s[sd], term_s[sd]) for sd in paired]
            lru_zero = [(lru_s[sd], 0.0) for sd in paired]
            entry[metric]["headroom_ci"] = _bootstrap_ratio(lru_bel, lru_zero)
            entry[metric]["terminal_share_ci"] = _bootstrap_ratio(lru_term, lru_bel)
        verdicts.append(entry)

    report = {"points": verdicts}
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def print_report(report: dict) -> None:
    header = ("headroom = how much less than lru the belady upper bound needs.  "
              "const%/term% = the share of that headroom already captured by a tuned "
              "constant TTL / by knowing only that a session ended.")
    for metric, unit, scale in (("prefill_tokens_computed", "M tokens recomputed", 1e6),
                                ("rm_per_1k_calls", "RM per 1k calls", 1.0),
                                ("ttft_p95", "TTFT p95 (s)", 1.0)):
        print()
        print("=" * 96)
        print(f"EXP01  metric: {unit}")
        if metric == "prefill_tokens_computed":
            print(header)
        print("=" * 96)
        for sweep in ("concurrency", "pause"):
            pts = [p for p in report["points"] if p["sweep"] == sweep]
            if not pts:
                continue
            axis = "conc" if sweep == "concurrency" else "pause_s"
            print(f"\n  sweep: {sweep}")
            print(f"  {axis:>7} {'lru':>9} {'const*':>9} {'ttl_orac':>9} {'term':>9} "
                  f"{'belady':>9} | {'headroom % [95% CI]':^26} {'terminal share % [95% CI]':^26}")
            print("  " + "-" * 118)
            key = "concurrency" if sweep == "concurrency" else "pause_median_s"
            for p in sorted(pts, key=lambda x: x[key]):
                m = p[metric]
                best_ttl = m["best_const_ttl_s"]
                ttl_label = ("--" if best_ttl is None
                             else "inf" if best_ttl >= 1e8 else f"{best_ttl:g}")
                def ci(block):
                    if not block:
                        return " " * 26
                    if block.get("lo") is None:
                        return f"{100*block['point']:>8.1f}  (no interval)"
                    return (f"{100*block['point']:>8.1f}  "
                            f"[{100*block['lo']:>6.1f},{100*block['hi']:>6.1f}]")

                # A share of a headroom whose own interval straddles zero is a ratio
                # with a sign-indeterminate denominator. Suppress it rather than print
                # a number that will be read as if it meant something.
                hci = m.get("headroom_ci")
                # A reduced arm set (--arms) can leave no lru/belady pair to form a
                # headroom from. Print the row rather than crashing on it: the runs are
                # still valid data and the CSV is already written by this point.
                share_txt = ci(m.get("terminal_share_ci"))
                if hci and hci.get("lo") is not None and hci["lo"] <= 0 <= hci["hi"]:
                    share_txt = f"{'-- (headroom CI spans 0)':>26}"

                def col(v):
                    return f"{v/scale:>9.3f}" if v is not None else f"{'--':>9}"

                print(f"  {p[key]:>7g} {col(m['lru'])} {col(m['best_const'])} "
                      f"{col(m['ttl_oracle'])} {col(m['oracle_terminal'])} "
                      f"{col(m['belady'])} |{ci(hci)} "
                      f"{share_txt}   n={(m.get('headroom_by_seed') or {}).get('n', 0)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/exp01")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--reanalyze", action="store_true",
                    help="re-run the analysis over an existing runs.csv, simulating nothing")
    ap.add_argument("--concurrency", default=",".join(str(c) for c in CONCURRENCY_GRID),
                    help="comma-separated concurrency grid; empty string skips the sweep")
    ap.add_argument("--pause", default=",".join(f"{p:g}" for p in PAUSE_GRID),
                    help="comma-separated pause-median grid; empty string skips the sweep")
    ap.add_argument("--arms", default="",
                    help="comma-separated subset of policy arms; empty runs all of them")
    args = ap.parse_args(argv)

    only_arms = [a.strip() for a in args.arms.split(",") if a.strip()] or None

    conc_grid = [int(v) for v in args.concurrency.split(",") if v.strip()]
    pause_grid = [float(v) for v in args.pause.split(",") if v.strip()]

    if args.reanalyze:
        with open(os.path.join(args.out, "runs.csv"), newline="", encoding="utf-8") as f:
            rows = []
            for raw in csv.DictReader(f):
                rows.append({k: (v if k in ("sweep", "policy") else float(v))
                             for k, v in raw.items()})
        print_report(analyze(rows, args.out))
        return 0

    os.makedirs(args.out, exist_ok=True)
    base = Config.load(args.config).replace(**{"workload.n_sessions": args.sessions})
    seeds = list(range(args.seeds))
    jobs = build_jobs(base, seeds, conc_grid, pause_grid, only_arms)
    print(f"EXP01: {len(jobs)} runs on {args.workers} workers "
          f"({args.sessions} sessions x {args.seeds} seeds)")

    with Pool(args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(_run_one, jobs, chunksize=1), 1):
            rows.append(row)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    csv_path = os.path.join(args.out, "runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = build_metadata(base, {
        "experiment": "exp01_ttl_falsify",
        "seeds": seeds,
        "const_ttl_grid": CONST_TTL_GRID,
        "arms_run": only_arms or "all",
        "concurrency_grid": conc_grid,
        "pause_grid": pause_grid,
        "pause_sweep_concurrency": PAUSE_SWEEP_CONCURRENCY,
        "min_reliable_headroom": MIN_RELIABLE_HEADROOM,
        "n_runs": len(rows),
        "workload_summary": workload_summary(generate_sessions(base.workload, seeds[0])),
    })
    with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print_report(analyze(rows, args.out))
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
