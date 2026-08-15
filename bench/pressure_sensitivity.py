"""How steep is hit rate against pressure, and does that explain the validation results?

The vLLM comparisons disagree by 1.4 pp at pressures 0.64 and 1.08 and by 4.8 to 10.9 pp
at 1.27. Read as a statement about the model, that says the simulator is fine up to about
1.1 and breaks above it. There is a duller reading worth ruling in or out first: the
hit-rate-versus-pressure curve may simply be much steeper at 1.27, so that ONE small
error in effective pressure shows up as a small hit-rate gap in the flat region and a
large one in the steep region.

The two readings differ in what they imply. "The model breaks above 1.1" means something
in the cache logic is wrong. "The curve is steep there" means the model is uniformly off
by a little in effective pressure, and the apparent size of the disagreement is an
artefact of where it was measured.

This measures the simulator's own d(hit)/d(pressure) with the workload seed FIXED and the
pool size varied, so trace-to-trace variation cannot contaminate the slope -- an earlier
attempt regressed hit rate on the pressure that happened to fall out of each seed and got
R^2=0.51, which is not a slope worth dividing by.

What this can and cannot show: it gives the simulator's slope, not vLLM's. If the steep-
curve reading is right, the implied pressure offset computed at 1.27 and at 1.08 should
come out about the SAME, since it is supposed to be one property of the model. If they
disagree, the steep curve is not the whole story. That is a consistency check across
measurements already taken, not a proof, and it costs no GPU time.

    python -m bench.pressure_sensitivity
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.config import Config  # noqa: E402
from sim.engine import run_config  # noqa: E402
from sim.workload import generate_sessions, mean_context_blocks  # noqa: E402

# (pressure, simulator hit, vLLM hit) for every comparison that was exact: inflation
# 1.000 and zero preemptions on both sides. See docs/validation_findings.md.
MEASURED = [
    (0.64, 0.9151, 0.9010, "default admission"),
    (1.08, 0.7382, 0.7517, "matched, cap 8"),
    (1.27, 0.5193, 0.5668, "matched, cap 8"),
    (1.27, 0.4644, 0.5733, "matched, cap 6"),
]


def run_at_pool(pool_blocks: int, seed: int, sessions_n: int, concurrency: int,
                max_num_seqs: int, max_prompt_tokens: int) -> dict:
    cfg = Config().replace(**{
        "workload.n_sessions": sessions_n,
        "arrival.concurrency": concurrency,
        "engine.kv_pool_blocks": pool_blocks,
        "engine.max_num_seqs": max_num_seqs,
        "policy.kind": "lru",
        "seed": seed,
    })
    sessions = generate_sessions(cfg.workload, seed)
    for sess in sessions:
        keep = 0
        for t in sess.turns:
            if t.prompt_tokens > max_prompt_tokens:
                break
            keep += 1
        sess.turns = sess.turns[:keep]
    sessions = [s for s in sessions if s.turns]
    summary = run_config(cfg, sessions).summary
    return {
        "pool_blocks": pool_blocks,
        "pressure": summary["pressure_offered"],
        "hit_rate": summary["token_hit_rate"],
        "makespan_s": summary["makespan_s"],
        "n_preemptions": summary["n_preemptions"],
    }


def local_slope(rows: list[dict], at: float, window: float = 0.18) -> float | None:
    """Least-squares slope over the points within `window` of `at`.

    The window has to hold at least three sweep points or the slope is refused rather
    than estimated from two. It is wide enough that the slope is a local average over
    roughly +/-0.18 of pressure, which on the flat part of the curve is fine and on the
    steep part understates the curvature.
    """
    pts = [(r["pressure"], r["hit_rate"]) for r in rows if abs(r["pressure"] - at) <= window]
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5,
                    help="average the curve over this many seeds; the SLOPE is what is "
                         "wanted and averaging removes the trace-to-trace level shift")
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-prompt-tokens", type=int, default=30000)
    ap.add_argument("--reanalyze", action="store_true",
                    help="recompute slopes and offsets from a saved sensitivity.json "
                         "instead of re-simulating the whole curve")
    ap.add_argument("--out", default="results/pressure_sensitivity")
    args = ap.parse_args(argv)

    # Pool sizes chosen to land near pressure 0.6 .. 1.5 for this workload.
    cfg0 = Config().replace(**{"workload.n_sessions": args.sessions,
                               "arrival.concurrency": args.concurrency})
    probe = generate_sessions(cfg0.workload, 0)
    for sess in probe:
        keep = 0
        for t in sess.turns:
            if t.prompt_tokens > args.max_prompt_tokens:
                break
            keep += 1
        sess.turns = sess.turns[:keep]
    probe = [s for s in probe if s.turns]
    ctx = mean_context_blocks(probe, cfg0.engine.block_size)
    targets = [0.60, 0.70, 0.80, 0.90, 1.00, 1.08, 1.15, 1.22, 1.27, 1.35, 1.45]
    pools = sorted({int(args.concurrency * ctx / t) for t in targets}, reverse=True)

    print(f"mean context blocks : {ctx:.0f}")
    print(f"pool sizes          : {pools}")
    print(f"seeds               : 0..{args.seeds - 1}\n")

    if args.reanalyze:
        with open(os.path.join(args.out, "sensitivity.json"), encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        return report(rows, args)

    per_seed: dict[int, list[dict]] = {}
    for seed in range(args.seeds):
        per_seed[seed] = [run_at_pool(p, seed, args.sessions, args.concurrency,
                                      args.max_num_seqs, args.max_prompt_tokens)
                          for p in pools]

    # Average across seeds at each pool size. The pressure differs slightly per seed
    # because the trace does, so average that too rather than pretending it is exact.
    rows = []
    for i, pool in enumerate(pools):
        rs = [per_seed[s][i] for s in range(args.seeds)]
        rows.append({
            "pool_blocks": pool,
            "pressure": st.fmean(r["pressure"] for r in rs),
            "hit_rate": st.fmean(r["hit_rate"] for r in rs),
            "hit_rate_sd": st.stdev([r["hit_rate"] for r in rs]) if args.seeds > 1 else 0.0,
            "makespan_s": st.fmean(r["makespan_s"] for r in rs),
            "n_preemptions": st.fmean(r["n_preemptions"] for r in rs),
        })

    return report(rows, args)


def report(rows: list[dict], args) -> int:
    print(f"{'pool':>7} {'pressure':>9} {'hit':>8} {'sd':>7} {'makespan':>9} {'preempt':>8}")
    print("-" * 52)
    for r in rows:
        print(f"{r['pool_blocks']:>7} {r['pressure']:>9.3f} {r['hit_rate']:>8.4f} "
              f"{r['hit_rate_sd']:>7.4f} {r['makespan_s']:>9.0f} {r['n_preemptions']:>8.1f}")

    print("\nlocal slope d(hit)/d(pressure):")
    slopes = {}
    for at in (0.64, 0.84, 1.00, 1.08, 1.27):
        sl = local_slope(rows, at)
        slopes[at] = sl
        print(f"  at pressure {at:.2f}: "
              + (f"{sl:+.3f} per unit pressure" if sl is not None else "not enough points"))

    print("\nimplied effective-pressure offset, per exact comparison against vLLM:")
    print("  (gap / local slope -- if the steep-curve reading is right these agree)")
    offsets = []
    for pressure, sim_hit, vllm_hit, label in MEASURED:
        sl = slopes.get(round(pressure, 2)) or local_slope(rows, pressure)
        if not sl:
            continue
        gap = sim_hit - vllm_hit
        off = gap / sl
        offsets.append(off)
        print(f"  pressure {pressure:.2f} ({label:<18}): gap {gap:+.4f}, "
              f"slope {sl:+.3f} -> offset {off:+.4f}")

    if len(offsets) >= 2:
        lo, hi = min(offsets), max(offsets)
        print(f"\n  offsets span [{lo:+.4f}, {hi:+.4f}]")
        if hi - lo <= 0.05 and lo > 0:
            print("  They agree to within 0.05 of pressure and all point the same way. The")
            print("  simulator behaves like a server at slightly HIGHER pressure than its")
            print("  nominal one, by about that much, everywhere tested -- and the apparent")
            print("  'breakdown above 1.1' is the same small error read off a steeper part")
            print("  of the curve.")
        else:
            print("  They do not agree, so a single pressure offset does not explain the")
            print("  disagreements. The steep-curve reading is ruled out as the whole story.")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump({
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sessions": args.sessions,
            "concurrency": args.concurrency,
            "max_num_seqs": args.max_num_seqs,
            "seeds": args.seeds,
            "rows": rows,
            "slopes": {str(k): v for k, v in slopes.items()},
            "measured_comparisons": MEASURED,
            "implied_offsets": offsets,
        }, f, indent=2)
    print(f"\nwrote {args.out}/sensitivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
