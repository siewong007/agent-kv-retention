"""How far does the simulator's pressure axis sit from a real server's?

Reads the runs produced by bench/sweep_admission.sh -- one per max_num_seqs, all on the
same pinned KV pool and the same workload -- and lays the two systems side by side.

What each column is for:

  query inflation   whether the /metrics ratio is a per-request hit rate at all. It is
                    1.000 only when nothing was preempted, and BOTH sides must read 1.000:
                    at high enough pressure the simulator preempts too, and two ratios
                    inflated by different amounts are no more comparable than one. Rows
                    that fail this are marked invalid rather than given a difference.
  preemptions       the denominator-free comparison. Valid on every row, including the
                    ones where the hit rate is not.
  hit rate          the cache comparison, where it is valid.

The point of the sweep is the shape of those columns as admission widens. The width at
which vLLM starts preempting is the width at which the two pressure axes come apart, and
the hit-rate column says how much that costs while it is still measurable. Note that the
simulator is not immune either: whole-prompt admission stops it over-committing on ADMIT,
but a decode that crosses a block boundary with no free block still preempts, so at high
enough pressure and a wide enough cap it preempts too. Where that happens is itself a
result -- it is the point where the two admission models stop differing in kind.

    python -m bench.analyze_admission_sweep
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    paths = sorted(glob.glob("results/sweep_admission/seqs_*/validation.json"),
                   key=lambda p: int(p.split("seqs_")[1].split(os.sep)[0].split("/")[0]))
    if not paths:
        print("no runs found under results/sweep_admission/ -- run bench/sweep_admission.sh",
              file=sys.stderr)
        return 1

    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        v, s = d["vllm"], d["simulator"]
        n = d["config"]["engine"]["max_num_seqs"]
        rows.append({
            "max_num_seqs": n,
            "pool_blocks": d["config"]["engine"]["kv_pool_blocks"],
            "pressure": d["offered_pressure"],
            "vllm_inflation": v["query_inflation"],
            "sim_inflation": s["query_inflation"],
            "vllm_preemptions": v["num_preemptions"],
            "sim_preemptions": s["n_preemptions"],
            "vllm_hit": v["hit_rate"],
            "sim_hit": s["hit_rate"],
            "vllm_wall": v["wall_s"],
            "sim_wall": s["makespan_s"],
        })

    pools = {r["pool_blocks"] for r in rows}
    pressures = {round(r["pressure"], 3) for r in rows}
    print(f"pool blocks : {sorted(pools)}"
          f"{'  <-- NOT PINNED, the sweep is confounded' if len(pools) > 1 else ''}")
    print(f"pressure    : {sorted(pressures)}")
    print(f"sessions    : {len(rows)} sweep points\n")

    hdr = (f"{'seqs':>5} {'infl(v)':>8} {'infl(s)':>8} {'preempt(v)':>11} {'preempt(s)':>11} "
           f"{'hit(v)':>8} {'hit(s)':>8} {'delta':>9} {'wall(v)':>8} {'wall(s)':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        valid = r["vllm_inflation"] <= 1.001 and r["sim_inflation"] <= 1.001
        delta = (f"{r['sim_hit']-r['vllm_hit']:+.4f}" if valid else "  invalid")
        hit_v = f"{r['vllm_hit']:.4f}" if valid else "   --   "
        print(f"{r['max_num_seqs']:>5} {r['vllm_inflation']:>8.3f} {r['sim_inflation']:>8.3f} "
              f"{r['vllm_preemptions']:>11.0f} {r['sim_preemptions']:>11} "
              f"{hit_v:>8} {r['sim_hit']:>8.4f} {delta:>9} "
              f"{r['vllm_wall']:>8.0f} {r['sim_wall']:>8.0f}")

    valid_rows = [r for r in rows
                  if r["vllm_inflation"] <= 1.001 and r["sim_inflation"] <= 1.001]
    print()
    if valid_rows:
        worst = max(valid_rows, key=lambda r: abs(r["sim_hit"] - r["vllm_hit"]))
        print(f"widest valid disagreement: {worst['sim_hit']-worst['vllm_hit']:+.4f} "
              f"at max_num_seqs={worst['max_num_seqs']}")
    sim_preempts = [r for r in rows if r["sim_preemptions"] > 0]
    if sim_preempts:
        lo = min(r["max_num_seqs"] for r in sim_preempts)
        print(f"the simulator itself starts preempting at max_num_seqs={lo}: the two "
              f"admission models are not as far apart\nat the top of this sweep as the "
              f"pressure-1.02 run suggested.")
    first_preempt = next((r for r in rows if r["vllm_preemptions"] > 0), None)
    if first_preempt is None:
        print("vLLM never preempted anywhere in this sweep: at this pool and workload the "
              "two admission models do not diverge, and simulator pressure IS server "
              "pressure over the whole range tested.")
    else:
        print(f"vLLM first preempts at max_num_seqs={first_preempt['max_num_seqs']}"
              f" ({first_preempt['vllm_preemptions']:.0f} preemptions). The simulator "
              f"preempts {first_preempt['sim_preemptions']} times there.")
        print("Above that width the two pressure axes are no longer the same axis.")

    os.makedirs("results/sweep_admission", exist_ok=True)
    with open("results/sweep_admission/summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote results/sweep_admission/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
