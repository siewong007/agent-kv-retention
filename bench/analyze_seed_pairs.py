"""Put an interval on the simulator-versus-vLLM gap at pressure 1.27.

Reads the paired runs from bench/seeds_at_pressure.sh plus seed 0 from
results/sweep_admission/, which was run under identical conditions, and reports:

  * the per-seed gap (simulator hit rate minus vLLM's), with a paired bootstrap interval
    over seeds at each admission width;
  * the DIFFERENCE between the two widths, which is the quantity the sweep was run for.
    A single effective-pressure offset cannot explain a width-dependent gap, because both
    widths sit at the same pressure on the same pinned pool. On one seed the two gaps
    differed by 2.3x; whether that survives seeds is what this decides.

Every row is checked before it is used. A run whose query inflation is above 1.001 on
either side is dropped with a reason rather than averaged in: vLLM's /metrics counters are
per SCHEDULING, so under preemption they count a resumed request twice and their ratio
stops being a per-request hit rate. Mixing an inflated row into a mean would reproduce the
25 pp phantom this project already retracted once.

    python -m bench.analyze_seed_pairs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import bootstrap_stat  # noqa: E402

# Seed 0 lives in the sweep round; it was run at the same pool, pressure and settings.
SEED0 = {6: "results/sweep_admission/seqs_6/validation.json",
         8: "results/sweep_admission/seqs_8/validation.json"}


def load(path: str) -> dict | None:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    v, s = d["vllm"], d["simulator"]
    row = {
        "path": path,
        "seed": d["config"]["seed"],
        "cap": d["config"]["engine"]["max_num_seqs"],
        "pool": d["config"]["engine"]["kv_pool_blocks"],
        "pressure": d["offered_pressure"],
        "vllm_hit": v["hit_rate"],
        "sim_hit": s["hit_rate"],
        "vllm_infl": v["query_inflation"],
        "sim_infl": s["query_inflation"],
        "vllm_preempt": v.get("num_preemptions"),
        "sim_preempt": s["n_preemptions"],
        "vllm_wall": v["wall_s"],
        "sim_wall": s["makespan_s"],
    }
    row["gap"] = row["sim_hit"] - row["vllm_hit"]
    row["wall_ratio"] = row["sim_wall"] / row["vllm_wall"] if row["vllm_wall"] else None
    if row["vllm_infl"] > 1.001 or row["sim_infl"] > 1.001:
        row["dropped"] = (f"inflation vLLM {row['vllm_infl']:.3f} / sim "
                          f"{row['sim_infl']:.3f}; the ratio is not a per-request hit rate")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/seeds_1p27")
    ap.add_argument("--out", default="results/seeds_1p27")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.results, "cap*_seed*", "validation.json")))
    for cap, path in SEED0.items():
        if os.path.isfile(path):
            paths.append(path)
    if not paths:
        print(f"no runs under {args.results} -- run bench/seeds_at_pressure.sh",
              file=sys.stderr)
        return 1

    rows = [load(p) for p in paths]
    kept = [r for r in rows if "dropped" not in r]
    dropped = [r for r in rows if "dropped" in r]

    pools = {r["pool"] for r in kept}
    print(f"pool blocks : {sorted(pools)}"
          f"{'   <-- NOT IDENTICAL, these runs are not comparable' if len(pools) > 1 else ''}")
    print(f"pressure    : {sorted({round(r['pressure'], 3) for r in kept})}\n")

    hdr = (f"{'cap':>4} {'seed':>5} {'hit(v)':>8} {'hit(s)':>8} {'gap':>9} "
           f"{'wall(v)':>8} {'wall(s)':>8} {'sim/vllm':>9} {'preempt v/s':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(kept, key=lambda r: (r["cap"], r["seed"])):
        pv = "-" if r["vllm_preempt"] is None else f"{r['vllm_preempt']:.0f}"
        print(f"{r['cap']:>4} {r['seed']:>5} {r['vllm_hit']:>8.4f} {r['sim_hit']:>8.4f} "
              f"{r['gap']:>+9.4f} {r['vllm_wall']:>8.0f} {r['sim_wall']:>8.0f} "
              f"{r['wall_ratio']:>9.3f} {pv + '/' + str(r['sim_preempt']):>12}")

    for r in dropped:
        print(f"\nDROPPED cap={r['cap']} seed={r['seed']}: {r['dropped']}")

    print()
    by_cap: dict[int, list[dict]] = {}
    for r in kept:
        by_cap.setdefault(r["cap"], []).append(r)

    summary = {}
    for cap in sorted(by_cap):
        gaps = [r["gap"] for r in by_cap[cap]]
        walls = [r["wall_ratio"] for r in by_cap[cap]]
        ci = bootstrap_stat(gaps, lambda xs: st.fmean(xs))
        line = (f"cap {cap:>2}: gap {st.fmean(gaps):+.4f}  n={len(gaps)}  "
                f"sd {st.stdev(gaps):.4f}" if len(gaps) > 1
                else f"cap {cap:>2}: gap {gaps[0]:+.4f}  n=1")
        if ci:
            line += f"  95% [{ci['lo']:+.4f}, {ci['hi']:+.4f}]"
        print(line)
        print(f"        makespan sim/vLLM {st.fmean(walls):.3f}"
              + (f"  sd {st.stdev(walls):.3f}" if len(walls) > 1 else ""))
        summary[cap] = {"n": len(gaps), "gap_mean": st.fmean(gaps),
                        "gap_ci": ci, "wall_ratio_mean": st.fmean(walls),
                        "gaps": gaps}

    caps = sorted(by_cap)
    if len(caps) == 2:
        lo_cap, hi_cap = caps
        # Pair by seed: the same trace on both widths, so the difference of gaps is
        # itself paired and the trace-to-trace movement cancels a second time.
        lo = {r["seed"]: r["gap"] for r in by_cap[lo_cap]}
        hi = {r["seed"]: r["gap"] for r in by_cap[hi_cap]}
        shared = sorted(set(lo) & set(hi))
        if shared:
            diffs = [lo[s] - hi[s] for s in shared]
            ci = bootstrap_stat(diffs, lambda xs: st.fmean(xs))
            print(f"\nwidth effect (cap {lo_cap} gap minus cap {hi_cap} gap), "
                  f"paired on {len(shared)} shared seeds:")
            print(f"  {st.fmean(diffs):+.4f}"
                  + (f"  95% [{ci['lo']:+.4f}, {ci['hi']:+.4f}]" if ci else "")
                  + (f"  sd {st.stdev(diffs):.4f}" if len(diffs) > 1 else ""))
            if ci and ci["lo"] < 0 < ci["hi"]:
                print("  The interval spans zero: on these seeds the gap does NOT depend")
                print("  on admission width, so the single-seed 2.3x does not survive and")
                print("  a uniform effective-pressure offset is back in contention as the")
                print("  explanation. docs/validation_findings.md needs correcting.")
            elif ci:
                print("  The interval excludes zero: the gap really does depend on")
                print("  admission width, which no single pressure offset can produce.")
                print("  The admission model is the thing that differs.")
            summary["width_effect"] = {"seeds": shared, "diffs": diffs, "ci": ci}

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "seed_pairs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rows": kept, "dropped": dropped, "summary": summary}, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
