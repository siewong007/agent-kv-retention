"""Record what the HPC round ran on, so the platform can never be reconstructed by guess.

The project's first rule is that every number in the paper comes from one platform and one
round. A rule like that is only enforceable if each set of results says which platform
produced it; otherwise the enforcement is somebody's memory of which directory was which.
This writes that statement next to the results, including the bits that decide whether two
rounds are comparable at all -- compute capability, driver, torch and vLLM versions, the
KV pool the node actually gave, and the git commit the code was at.

It also records the elapsed wall clock and what that cost, because the HPC bills on wall
clock rather than GPU utilisation, and a round whose cost is not written down is a round
whose cost gets estimated from memory later.

`adopted_into_sim_config` is deliberately hardcoded false. These constants do not take
effect by existing; adopting them is an edit to sim/config.py plus a re-run of every
experiment, and this field is here to make a half-finished adoption visible.

    python hpc/write_manifest.py --out results/hpc --elapsed-s 8100 --rm-per-hour 3.06 \
        --model Qwen/Qwen2.5-3B-Instruct --pool-default 13800 --pool-capped 12900
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=30, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--elapsed-s", type=int, required=True)
    ap.add_argument("--rm-per-hour", type=float, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool-default", type=int, required=True)
    ap.add_argument("--pool-capped", type=int, required=True)
    args = ap.parse_args(argv)

    env_path = os.path.join(args.out, "env.json")
    if not os.path.exists(env_path):
        print(f"missing {env_path}; the environment check must run first, otherwise this "
              f"manifest would assert a platform nobody verified", file=sys.stderr)
        return 1
    with open(env_path, encoding="utf-8") as f:
        env = json.load(f)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "HPC calibration + validation round; see hpc/run_calibration.sh",
        "platform": env,
        # Whatever the scheduler exported. Which node, which partition, which job id --
        # the things needed to ask the cluster later what this run actually got.
        "job": {k: v for k, v in sorted(os.environ.items())
                if k.startswith(("SLURM_", "PBS_", "LSB_"))},
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "model": args.model,
        "kv_pool_blocks_default_admission": args.pool_default,
        "kv_pool_blocks_capped_admission": args.pool_capped,
        "elapsed_s": args.elapsed_s,
        "rm_per_hour": args.rm_per_hour,
        "estimated_cost_rm": round(args.elapsed_s / 3600 * args.rm_per_hour, 2),
        "adopted_into_sim_config": False,
        "note": ("Constants here are NOT in sim/config.py. Adopting them means editing "
                 "sim/config.py in a commit that says so and re-running every experiment. "
                 "Results produced from these constants must never share a figure with "
                 "results from the local RTX 5080 round."),
    }

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {path}")
    if manifest["git_dirty"]:
        print("\nWARNING: the working tree was dirty. This round cannot be reproduced "
              "from its commit alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
