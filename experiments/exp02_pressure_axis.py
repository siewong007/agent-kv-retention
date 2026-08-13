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
    default_workers, read_results, run_grid, write_results,
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
        def mean_of(policy, metric):
            vals = [r[metric] for r in group if r["policy"] == policy]
            return st.fmean(vals) if vals else None

        lru = mean_of("lru", "prefill_tokens_computed")
        bel = mean_of("belady", "prefill_tokens_computed")
        term = mean_of("oracle_terminal", "prefill_tokens_computed")
        headroom = (lru - bel) / lru if lru else 0.0
        out.append({
            "condition": cond,
            "target_pressure": target,
            "pressure_measured": st.fmean([r["pressure_measured"] for r in group]),
            "concurrency": group[0]["concurrency"],
            "pool_blocks": group[0]["pool_blocks"],
            "hit_rate_lru": mean_of("lru", "token_hit_rate"),
            "hit_rate_belady": mean_of("belady", "token_hit_rate"),
            "headroom_frac": headroom,
            "terminal_share": ((lru - term) / (lru - bel)) if lru and lru > bel else None,
            "gpu_busy_frac": mean_of("lru", "gpu_busy_frac"),
            "rm_per_1k_calls_lru": mean_of("lru", "rm_per_1k_calls"),
            "n_sessions": group[0].get("n_sessions"),
            "n_seeds": len({r["seed"] for r in group}),
        })

    # Collapse test: at each target pressure, how far apart are the three conditions?
    spread = []
    for target in TARGET_PRESSURES:
        at = [p for p in out if p["target_pressure"] == target]
        if len(at) < 2:
            continue
        hits = [p["hit_rate_lru"] for p in at]
        heads = [p["headroom_frac"] for p in at]
        spread.append({
            "target_pressure": target,
            "hit_rate_range": max(hits) - min(hits),
            "hit_rate_mean": st.fmean(hits),
            "headroom_range": max(heads) - min(heads),
            "headroom_mean": st.fmean(heads),
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

    print("Collapse test -- spread across the three conditions at equal pressure:")
    print(f"{'target p':>9} {'hit rate mean':>14} {'hit rate range':>15} "
          f"{'headroom mean':>14} {'headroom range':>15}")
    print("-" * 72)
    for s in report["collapse"]:
        print(f"{s['target_pressure']:>9.2f} {s['hit_rate_mean']:>14.3f} "
              f"{s['hit_rate_range']:>15.3f} {100*s['headroom_mean']:>13.1f}% "
              f"{100*s['headroom_range']:>14.1f}%")
    print("\nA small range means the three ways of reaching a pressure agree, i.e. the "
          "axis transfers.\nA large one means the result belongs to its configuration, "
          "not to the pressure.")


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
