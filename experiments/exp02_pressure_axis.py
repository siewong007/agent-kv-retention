"""EXP02 -- is working-set / pool-capacity the right axis?

EXP01 reported everything against concurrency. That number is a property of one GPU
running one model: "the cache collapses above 16 concurrent sessions" is unreproducible
on any other card. The claim worth making is that the phenomenon depends on

    pressure = (live sessions x mean context blocks) / KV pool blocks

and that concurrency and pool size are merely two handles on it.

That claim is falsifiable, and this experiment falsifies or supports it: reach the same
pressure three different ways and see whether the curves land on top of each other.

    vary_concurrency   fix the pool at a 16 GB card's capacity, scale sessions
    vary_pool          fix sessions at 16, scale the pool
    bigger_gpu         fix the pool at 2.5x, scale sessions to match

If the three collapse onto one curve, pressure is the axis and every EXP01 result
transfers to other hardware by rescaling. If they do not, the axis is wrong and the
results are specific to the configuration that produced them.

    python -m experiments.exp02_pressure_axis --seeds 10 --out results/exp02
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    bootstrap_ratio, bootstrap_stat, default_workers, read_results, run_grid,
    write_results,
)
from sim.config import Config  # noqa: E402
from sim.workload import generate_sessions, mean_context_blocks  # noqa: E402

TARGET_PRESSURES = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0]
ARMS = ["lru", "oracle_terminal", "belady"]

POOL_16GB = 16000   # RTX 5080 16 GB with Qwen2.5-3B, see docs/calibration.md
POOL_BIG = 40000    # a card with ~2.5x the KV capacity
FIXED_CONCURRENCY = 16

# Raised well above anything the grid reaches so that the sequence cap is never the
# binding constraint. If it bound, this experiment would be measuring the cap and not
# memory pressure.
MAX_NUM_SEQS = 256


def conditions(ctx_blocks: float) -> list[dict]:
    """(condition name, pressure) -> the config knobs that produce it."""
    out = []
    for p in TARGET_PRESSURES:
        conc = max(1, round(p * POOL_16GB / ctx_blocks))
        out.append({"condition": "vary_concurrency", "target_pressure": p,
                    "engine.kv_pool_blocks": POOL_16GB, "arrival.concurrency": conc})

        pool = max(1000, round(FIXED_CONCURRENCY * ctx_blocks / p))
        out.append({"condition": "vary_pool", "target_pressure": p,
                    "engine.kv_pool_blocks": pool,
                    "arrival.concurrency": FIXED_CONCURRENCY})

        conc_big = max(1, round(p * POOL_BIG / ctx_blocks))
        out.append({"condition": "bigger_gpu", "target_pressure": p,
                    "engine.kv_pool_blocks": POOL_BIG, "arrival.concurrency": conc_big})
    return out


# Sessions per concurrency slot. A run that only fits two or three waves of sessions is
# mostly ramp-up and drain, and the transient differs between conditions -- which is
# exactly the comparison this experiment is trying to make. Scaling the session count
# with concurrency keeps every condition the same number of waves deep.
SESSIONS_PER_SLOT = 10


def build_jobs(base: Config, seeds: list[int], ctx_blocks: float,
               min_sessions: int) -> list[dict]:
    jobs = []
    for cond in conditions(ctx_blocks):
        knobs = {k: v for k, v in cond.items()
                 if k not in ("condition", "target_pressure")}
        n_sessions = max(min_sessions,
                         SESSIONS_PER_SLOT * int(knobs["arrival.concurrency"]))
        knobs["workload.n_sessions"] = n_sessions
        for seed in seeds:
            for arm in ARMS:
                cfg = base.replace(seed=seed, **{**knobs, "policy.kind": arm})
                jobs.append({
                    "config": cfg.to_dict(),
                    "labels": {"condition": cond["condition"],
                               "target_pressure": cond["target_pressure"]},
                })
    return jobs


def analyze(rows: list[dict], out_dir: str) -> dict:
    points = {}
    for r in rows:
        points.setdefault((r["condition"], r["target_pressure"]), []).append(r)

    out = []
    for (cond, target), group in sorted(points.items()):
        def by_seed(policy, metric="prefill_tokens_computed"):
            return {r["seed"]: r[metric] for r in group if r["policy"] == policy}

        def mean_of(policy, metric="prefill_tokens_computed"):
            vals = list(by_seed(policy, metric).values())
            return st.fmean(vals) if vals else None

        seeds = sorted({r["seed"] for r in group})
        lru = mean_of("lru")
        bel = mean_of("belady")
        term = mean_of("oracle_terminal")
        headroom = (lru - bel) / lru if lru else 0.0

        lru_s = by_seed("lru")
        bel_s = by_seed("belady")
        paired = [sd for sd in seeds if sd in lru_s and sd in bel_s]
        headroom_ci = bootstrap_ratio([(lru_s[sd], bel_s[sd]) for sd in paired],
                                      [(lru_s[sd], 0.0) for sd in paired])

        out.append({
            "condition": cond,
            "target_pressure": target,
            "pressure_measured": st.fmean([r["pressure_measured"] for r in group]),
            "concurrency": group[0]["concurrency"],
            "pool_blocks": group[0]["pool_blocks"],
            "hit_rate_lru": mean_of("lru", "token_hit_rate"),
            "hit_rate_belady": mean_of("belady", "token_hit_rate"),
            "headroom_frac": headroom,
            "headroom_ci": headroom_ci,
            "terminal_share": ((lru - term) / (lru - bel)) if lru and lru > bel else None,
            "gpu_busy_frac": mean_of("lru", "gpu_busy_frac"),
            "rm_per_1k_calls_lru": mean_of("lru", "rm_per_1k_calls"),
            "n_sessions": group[0].get("n_sessions"),
            "n_seeds": len(seeds),
        })

    # Collapse test. The statistic is a max-minus-min across three conditions, which has
    # no closed-form standard error, so whole seeds are resampled and the spread is
    # recomputed from scratch on each draw. Without an interval, a small spread cannot be
    # distinguished from three conditions that merely happened to land close together.
    spread = []
    for target in TARGET_PRESSURES:
        at = [p for p in out if p["target_pressure"] == target]
        if len(at) < 2:
            continue
        conds = [p["condition"] for p in at]
        per_seed = []
        seeds = sorted({r["seed"] for r in rows if r["target_pressure"] == target})
        for sd in seeds:
            row = {}
            for cond in conds:
                g = [r for r in rows
                     if r["target_pressure"] == target and r["condition"] == cond
                     and r["seed"] == sd]
                lru = next((r["prefill_tokens_computed"] for r in g
                            if r["policy"] == "lru"), None)
                bel = next((r["prefill_tokens_computed"] for r in g
                            if r["policy"] == "belady"), None)
                hit = next((r["token_hit_rate"] for r in g if r["policy"] == "lru"), None)
                if lru is None or bel is None:
                    continue
                row[cond] = {"lru": lru, "belady": bel, "hit": hit}
            if len(row) == len(conds):
                per_seed.append(row)

        def head_range(sample):
            if not sample:
                return None
            vals = []
            for cond in conds:
                lru = st.fmean([s[cond]["lru"] for s in sample])
                bel = st.fmean([s[cond]["belady"] for s in sample])
                vals.append((lru - bel) / lru if lru else 0.0)
            return max(vals) - min(vals)

        def hit_range(sample):
            if not sample:
                return None
            vals = [st.fmean([s[cond]["hit"] for s in sample]) for cond in conds]
            return max(vals) - min(vals)

        spread.append({
            "target_pressure": target,
            "hit_rate_range": max(p["hit_rate_lru"] for p in at)
                              - min(p["hit_rate_lru"] for p in at),
            "hit_rate_mean": st.fmean([p["hit_rate_lru"] for p in at]),
            "headroom_range": max(p["headroom_frac"] for p in at)
                              - min(p["headroom_frac"] for p in at),
            "headroom_mean": st.fmean([p["headroom_frac"] for p in at]),
            "headroom_range_ci": bootstrap_stat(per_seed, head_range),
            "hit_range_ci": bootstrap_stat(per_seed, hit_range),
            "n_seeds": len(per_seed),
        })

    report = {"points": out, "collapse": spread}
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def print_report(report: dict) -> None:
    print()
    print("=" * 104)
    print("EXP02  the same pressure reached three ways. If pressure is the right axis, "
          "the rows within a block agree.")
    print("=" * 104)
    print(f"{'target p':>9} {'condition':<18} {'conc':>5} {'pool':>7} {'p_meas':>7} "
          f"{'hit(lru)':>9} {'hit(bel)':>9} {'headroom':>9} {'term%':>7} {'busy':>6}")
    print("-" * 104)
    for target in sorted({p["target_pressure"] for p in report["points"]}):
        for p in sorted([x for x in report["points"] if x["target_pressure"] == target],
                        key=lambda x: x["condition"]):
            term = f"{100*p['terminal_share']:>6.1f}%" if p["terminal_share"] is not None else "     --"
            print(f"{target:>9.2f} {p['condition']:<18} {p['concurrency']:>5.0f} "
                  f"{p['pool_blocks']:>7.0f} {p['pressure_measured']:>7.2f} "
                  f"{p['hit_rate_lru']:>9.3f} {p['hit_rate_belady']:>9.3f} "
                  f"{100*p['headroom_frac']:>8.1f}% {term:>7} {p['gpu_busy_frac']:>6.3f}")
        print()

    print("Collapse test -- disagreement between the three conditions at equal pressure,")
    print("with 95% bootstrap intervals over seeds (whole seeds resampled, spread "
          "recomputed):")
    print(f"{'target p':>9} {'headroom mean':>14} {'headroom range [95% CI]':>30} "
          f"{'hit range [95% CI]':>26}")
    print("-" * 84)
    for c in report["collapse"]:
        hr = c.get("headroom_range_ci") or {}
        hc = c.get("hit_range_ci") or {}

        def fmt(ci, val, scale, unit):
            if not ci or ci.get("lo") is None:
                return f"{val*scale:>10.1f}{unit}"
            return (f"{val*scale:>8.1f}{unit} [{ci['lo']*scale:>5.1f},"
                    f"{ci['hi']*scale:>5.1f}]")

        print(f"{c['target_pressure']:>9.2f} {100*c['headroom_mean']:>13.1f}% "
              f"{fmt(hr, c['headroom_range'], 100, '%'):>30} "
              f"{fmt(hc, c['hit_rate_range'], 1, ''):>26}")
    print("\nA small range means the three ways of reaching a pressure agree, i.e. the "
          "axis transfers.\nRead the upper bound of the interval, not the point "
          "estimate: it is the largest\ndisagreement the data is compatible with, and "
          "that is what a transfer claim has to survive.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/exp02")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--reanalyze", action="store_true")
    args = ap.parse_args(argv)

    if args.reanalyze:
        print_report(analyze(read_results(args.out), args.out))
        return 0

    base = Config.load(args.config).replace(**{"engine.max_num_seqs": MAX_NUM_SEQS})
    seeds = list(range(args.seeds))
    ctx_blocks = mean_context_blocks(generate_sessions(base.workload, seeds[0]),
                                     base.engine.block_size)
    print(f"mean context per live session: {ctx_blocks:.1f} blocks")

    jobs = build_jobs(base, seeds, ctx_blocks, args.sessions)
    rows = run_grid(jobs, args.workers, "EXP02")
    os.makedirs(args.out, exist_ok=True)
    csv_path = write_results(args.out, rows, base, {
        "experiment": "exp02_pressure_axis",
        "seeds": seeds,
        "target_pressures": TARGET_PRESSURES,
        "arms": ARMS,
        "mean_context_blocks": ctx_blocks,
        "pool_16gb": POOL_16GB,
        "pool_big": POOL_BIG,
        "fixed_concurrency": FIXED_CONCURRENCY,
        "max_num_seqs": MAX_NUM_SEQS,
        "sessions_per_slot": SESSIONS_PER_SLOT,
        "min_sessions": args.sessions,
    })
    print_report(analyze(rows, args.out))
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
