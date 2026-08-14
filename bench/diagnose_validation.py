"""Why did the simulator and vLLM disagree by 25 points? Re-analysis, no GPU.

The first pass at bench/validate_vs_vllm.py compared the simulator's `token_hit_rate`
to vLLM's `prefix_cache_hits_total / prefix_cache_queries_total` and concluded that the
simulator under-evicts by 3.2x. That conclusion did not survive a look at the
denominators.

    pressure 0.64:  vLLM queried 9,854,329 tokens against 9,854,329 prompt tokens sent
    pressure 1.02:  vLLM queried 26,013,279 tokens against 15,360,566 prompt tokens sent

Exactly 1.000x at low pressure and 1.694x at high pressure. The counters are per
SCHEDULING, not per request: a preempted request queries the cache again when it is
resumed, and both numerator and denominator grow. The simulator's `token_hit_rate`
counts each prompt once. At 0.64 nothing was preempted so the two definitions coincide
and the comparison is valid; at 1.02 they measure different things, and the 25 pp gap
is mostly that.

This script rebuilds the identical traces on CPU and reports the simulator under BOTH
accountings, so the two runs can be compared on the definition vLLM actually uses. It
also reports what the naive comparison would have to assume to be true, which is the
part worth keeping: 14,310,464 hits is more than the theoretical maximum a
single-query accounting could produce on this trace, so the inflation is not a
hypothesis, it is forced by arithmetic.

    python -m bench.diagnose_validation
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.config import Config  # noqa: E402
from sim.engine import run_config  # noqa: E402
from sim.workload import generate_sessions  # noqa: E402

RUNS = [
    ("results/validate/validation.json", "0.64"),
    ("results/validate_pressured/validation.json", "1.02"),
]


def rebuild(cfg_dict: dict, max_prompt_tokens: int, pool_override: int | None = None):
    """Reconstruct the exact trace the validation run used, truncation included."""
    cfg = Config.from_dict(cfg_dict)
    if pool_override is not None:
        cfg = cfg.replace(**{"engine.kv_pool_blocks": pool_override})
    sessions = generate_sessions(cfg.workload, cfg.seed)
    for sess in sessions:
        keep = 0
        for t in sess.turns:
            if t.prompt_tokens > max_prompt_tokens:
                break
            keep += 1
        sess.turns = sess.turns[:keep]
    return cfg, [s for s in sessions if s.turns]


def main() -> int:
    out = {}
    for path, tag in RUNS:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        v = d["vllm"]
        prompt_tokens = d["tokenisation"]["actual_prompt_tokens"]

        cfg, sessions = rebuild(d["config"], d["max_prompt_tokens"])
        sim = run_config(cfg, sessions).summary

        # Ceiling: a pool large enough that nothing can ever be evicted. Whatever is
        # still a miss is intrinsic to the workload, not to the retention policy.
        big_cfg, big_sessions = rebuild(d["config"], d["max_prompt_tokens"],
                                        pool_override=10 * cfg.engine.kv_pool_blocks)
        ceiling = run_config(big_cfg, big_sessions).summary

        print(f"\n=== offered pressure {tag} "
              f"({cfg.workload.n_sessions} sessions, concurrency {cfg.arrival.concurrency}) ===")
        print(f"  prompt tokens sent            {prompt_tokens:>12,}")
        print(f"  vLLM queries / prompt tokens  {v['prefix_cache_queries']/prompt_tokens:>12.3f}"
              "   (1.000 => every prompt scheduled exactly once)")
        print()
        print("  per-request accounting (what the policy experiments use):")
        print(f"    simulator token_hit_rate    {sim['token_hit_rate']:>12.4f}")
        print(f"    no-eviction ceiling         {ceiling['token_hit_rate']:>12.4f}")
        print(f"    lost to eviction            {ceiling['token_hit_rate']-sim['token_hit_rate']:>12.4f}")
        print()
        print("  per-scheduling accounting (what vLLM's counters use):")
        print(f"    simulator query_hit_rate    {sim['query_hit_rate']:>12.4f}"
              f"   ({sim['query_tokens']/prompt_tokens:.3f}x prompt tokens queried)")
        print(f"    vLLM  hits/queries          {v['hit_rate']:>12.4f}")
        print(f"    delta (sim - vLLM)          {sim['query_hit_rate']-v['hit_rate']:>+12.4f}")
        print()
        print(f"  simulator preemptions         {sim['n_preemptions']:>12,}"
              f"   over {sim['n_calls']} calls")
        print(f"  simulator evictions           {sim['n_evictions']:>12,}")

        # The arithmetic that rules out reading vLLM's ratio as a per-request hit rate.
        max_single = prompt_tokens * ceiling["token_hit_rate"]
        print(f"\n  most hits any single-query accounting could yield on this trace: "
              f"{max_single:>12,.0f}")
        print(f"  hits vLLM actually reported:                                   "
              f"{v['prefix_cache_hits']:>12,.0f}"
              f"   ({v['prefix_cache_hits']/max_single:.2f}x)")

        out[tag] = {
            "prompt_tokens": prompt_tokens,
            "vllm_query_inflation": v["prefix_cache_queries"] / prompt_tokens,
            "vllm_hit_rate": v["hit_rate"],
            "sim_token_hit_rate": sim["token_hit_rate"],
            "sim_query_hit_rate": sim["query_hit_rate"],
            "sim_query_inflation": sim["query_tokens"] / prompt_tokens,
            "sim_ceiling_token_hit_rate": ceiling["token_hit_rate"],
            "sim_n_preemptions": sim["n_preemptions"],
            "sim_n_evictions": sim["n_evictions"],
            "max_hits_under_single_query_accounting": max_single,
            "vllm_hits": v["prefix_cache_hits"],
        }

    os.makedirs("results/validate_diagnosis", exist_ok=True)
    with open("results/validate_diagnosis/diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/validate_diagnosis/diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
