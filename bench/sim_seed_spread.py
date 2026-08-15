"""How much does the simulator move across seeds at one operating point? CPU only.

docs/validation_findings.md reports a 4.8 pp disagreement with vLLM at pressure 1.27,
from one seed. Before buying GPU time to put an interval on it, there is a free question
worth answering: how much does the simulator alone move when only the workload seed
changes? It turns out to swing enormously -- 0.436 to 0.732 over ten seeds -- which is
worth knowing before quoting any hit-rate LEVEL at this operating point.

This does NOT measure the variance of the vLLM-versus-simulator difference, and the
distinction matters more than it first looks. That difference is PAIRED -- both systems
run the identical trace -- so whatever the trace does to the hit rate happens to both, and
most of the spread measured here cancels. EXP02 shows how much: its headroom intervals at
pressure 1.30 are [1.5, 2.8] pp, tight, even though the underlying hit-rate level moves by
tens of points across seeds, because it too compares arms on a shared trace.

So a large spread here does NOT prove the single-seed vLLM comparison is meaningless. What
it establishes is narrower: the LEVEL is highly seed-dependent at this operating point, so
any statement about the level needs seeds, and the paired difference has an unknown
variance that only paired runs on both systems can measure.

    python -m bench.sim_seed_spread --seeds 10
    python -m bench.sim_seed_spread --seeds 10 --point matched_1.08
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
from sim.workload import generate_sessions  # noqa: E402

# The operating points that have been compared against a live server, so that a spread
# measured here lines up with a disagreement measured there rather than approximating it.
POINTS = {
    "sweep_1.27": dict(sessions=60, concurrency=16, pool=10922, max_num_seqs=8),
    "matched_1.08": dict(sessions=60, concurrency=16, pool=12865, max_num_seqs=8),
    "default_0.64": dict(sessions=40, concurrency=10, pool=13663, max_num_seqs=64),
}


def run_one(point: dict, seed: int, max_prompt_tokens: int) -> dict:
    cfg = Config().replace(**{
        "workload.n_sessions": point["sessions"],
        "arrival.concurrency": point["concurrency"],
        "engine.kv_pool_blocks": point["pool"],
        "engine.max_num_seqs": point["max_num_seqs"],
        "policy.kind": "lru",
        "seed": seed,
    })
    sessions = generate_sessions(cfg.workload, seed)
    # Same context-window truncation the validation harness applies, so the trace is the
    # one the server actually saw rather than a longer cousin of it.
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
        "seed": seed,
        "hit_rate": summary["token_hit_rate"],
        "makespan_s": summary["makespan_s"],
        "n_preemptions": summary["n_preemptions"],
        "pressure_offered": summary["pressure_offered"],
        "n_calls": summary["n_calls"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--point", choices=sorted(POINTS), default="sweep_1.27")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-prompt-tokens", type=int, default=30000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    point = POINTS[args.point]
    print(f"point   : {args.point}  {point}")
    print(f"seeds   : 0..{args.seeds - 1}\n")

    rows = []
    for seed in range(args.seeds):
        row = run_one(point, seed, args.max_prompt_tokens)
        rows.append(row)
        print(f"  seed {seed:>2}  hit {row['hit_rate']:.4f}  "
              f"makespan {row['makespan_s']:>7.0f}s  "
              f"preempt {row['n_preemptions']:>4}  "
              f"pressure {row['pressure_offered']:.3f}  "
              f"calls {row['n_calls']}")

    hits = [r["hit_rate"] for r in rows]
    walls = [r["makespan_s"] for r in rows]
    spread = max(hits) - min(hits)
    sd = st.stdev(hits) if len(hits) > 1 else 0.0

    print(f"\nhit rate   mean {st.fmean(hits):.4f}  sd {sd:.4f}  "
          f"min {min(hits):.4f}  max {max(hits):.4f}  spread {spread:.4f}")
    print(f"makespan   mean {st.fmean(walls):.0f}s  "
          f"sd {st.stdev(walls) if len(walls) > 1 else 0:.0f}s  "
          f"spread {max(walls)-min(walls):.0f}s")

    print()
    if args.point == "sweep_1.27":
        gap = 0.0476  # measured against vLLM at this point, one seed
        print(f"measured disagreement with vLLM here: {gap:.4f} (one seed)")
        print(f"simulator level spread across seeds:  {spread:.4f}  ({spread/gap:.1f}x the gap)")
        print()
        print("The level is far more seed-dependent than the gap is large. That does not")
        print("make the gap meaningless -- it is a paired quantity and most of this spread")
        print("cancels -- but it does mean the gap's own variance is unmeasured, and no")
        print("amount of CPU time can measure it. Only paired runs on both systems can.")

    out = args.out or f"results/sim_seed_spread/{args.point}"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "spread.json"), "w", encoding="utf-8") as f:
        json.dump({
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "point": args.point,
            "point_config": point,
            "max_prompt_tokens": args.max_prompt_tokens,
            "rows": rows,
            "hit_rate_mean": st.fmean(hits),
            "hit_rate_sd": sd,
            "hit_rate_spread": spread,
        }, f, indent=2)
    print(f"\nwrote {out}/spread.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
