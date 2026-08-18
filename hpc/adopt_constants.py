"""Compare the HPC-fitted engine constants against sim/config.py, and optionally adopt them.

Adoption is deliberately a separate, explicit step from measurement. The constants do not
take effect by existing in results/hpc/calib/; they take effect when sim/config.py changes,
and at that moment every existing result becomes a number from a different platform. The
project's first rule -- one platform, one round, per figure -- is only enforceable if that
moment is a commit somebody made on purpose.

So the default action is to print the diff and change nothing. `--apply` edits the five
constant lines, and refuses to run on a dirty tree, because an edit that cannot be tied to
a commit is exactly the situation the rule exists to prevent.

What is NOT adopted automatically:

  * kv_pool_blocks. The experiments sweep the pool as a design parameter -- EXP02 varies
    it by 4x on purpose -- so the number in the config is a starting point, not a
    measurement of any particular server. It is shown in the diff for information.
  * anything at all if results/hpc/manifest.json is missing. Without it there is no record
    of which platform produced the fits, and an unlabelled constant is worse than a stale
    one: a stale constant is wrong, an unlabelled one is unfalsifiable.

    python -m hpc.adopt_constants                      # show the diff, change nothing
    python -m hpc.adopt_constants --apply              # edit sim/config.py
    python -m hpc.adopt_constants --results results/hpc
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.config import Config  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "sim", "config.py")

# constant in EngineConfig -> where the fit puts it
SOURCES = {
    "step_overhead_s": ("batch_fit.json", ("fit", "@regime", "step_overhead_s")),
    "decode_s_per_seq": ("batch_fit.json", ("fit", "@regime", "decode_s_per_seq")),
    "prefill_s_per_token": ("timing_fit.json",
                            ("fit", "prefill", "quadratic_fit", "linear_s_per_token")),
    "prefill_s_per_token2": ("timing_fit.json",
                             ("fit", "prefill", "quadratic_fit", "quadratic_s_per_token2")),
    "decode_s_per_kv_token": ("timing_fit.json",
                              ("fit", "decode", "decode_s_per_kv_token")),
}


def dig(payload: dict, path: tuple):
    node = payload
    for key in path:
        if key == "@regime":
            # fit_batch reports a loaded regime (batch >= min_batch) and an all-batches
            # fallback. The loaded one is the right regime for a server under load; the
            # fallback exists for runs that never got there.
            node = node.get("loaded_regime") or node.get("all_batches")
            continue
        node = node[key]
    return node


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         timeout=30, check=False)
    return out.stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/hpc",
                    help="the round to adopt from (default: results/hpc)")
    ap.add_argument("--apply", action="store_true",
                    help="edit sim/config.py. Refuses on a dirty tree")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="apply even with uncommitted changes. Only sensible if you are "
                         "about to commit everything together and know it")
    args = ap.parse_args(argv)

    calib = os.path.join(args.results, "calib")
    manifest_path = os.path.join(args.results, "manifest.json")

    if not os.path.isfile(manifest_path):
        print(f"no manifest at {manifest_path}. The fits carry no record of which platform "
              f"produced them, and an unlabelled constant cannot be checked against the "
              f"one-platform-per-figure rule. Run hpc/run_calibration.sh, which writes it.",
              file=sys.stderr)
        return 1

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    plat = manifest.get("platform", {})

    print("adopting from")
    print(f"  results   : {args.results}")
    print(f"  device    : {plat.get('device')}  capability {plat.get('capability')}")
    print(f"  host      : {plat.get('hostname')}")
    print(f"  torch     : {plat.get('torch')} (CUDA build {plat.get('torch_cuda_build')})")
    print(f"  vllm      : {plat.get('vllm')}")
    print(f"  nvidia-smi: {plat.get('nvidia_smi')}")
    print(f"  commit    : {manifest.get('git_commit')}"
          f"{'  (DIRTY TREE)' if manifest.get('git_dirty') else ''}")
    print(f"  cost      : RM {manifest.get('estimated_cost_rm')} "
          f"over {manifest.get('elapsed_s')}s")
    print()

    payloads = {}
    for fname in ("timing_fit.json", "batch_fit.json"):
        path = os.path.join(calib, fname)
        if not os.path.isfile(path):
            print(f"missing {path}", file=sys.stderr)
            return 1
        with open(path, encoding="utf-8") as f:
            payloads[fname] = json.load(f)

    engine = Config().engine
    rows = []
    for name, (fname, path) in SOURCES.items():
        current = getattr(engine, name)
        fitted = dig(payloads[fname], path)
        change = (fitted - current) / current * 100 if current else float("inf")
        rows.append((name, current, fitted, change, fname))

    width = max(len(r[0]) for r in rows)
    print(f"{'constant':<{width}}  {'current':>13}  {'HPC-fitted':>13}  {'change':>8}  source")
    print("-" * (width + 56))
    for name, current, fitted, change, fname in rows:
        print(f"{name:<{width}}  {current:>13.6g}  {fitted:>13.6g}  {change:>+7.1f}%  {fname}")

    pool_default = manifest.get("kv_pool_blocks_default_admission")
    print(f"\n{'kv_pool_blocks':<{width}}  {engine.kv_pool_blocks:>13}  "
          f"{pool_default:>13}  {'':>8}  manifest (NOT adopted)")
    print("  The experiments sweep the pool as a design parameter -- EXP02 varies it 4x on")
    print("  purpose -- so the config value is a starting point, not a measurement. Change")
    print("  it only if you mean to move every experiment's operating point.")

    biggest = max(abs(r[3]) for r in rows)
    print(f"\nlargest change: {biggest:.1f}%")
    if biggest < 1.0:
        print("Under 1%. The platforms agree on timing; adopting would change no")
        print("conclusion, and NOT adopting keeps every existing result quotable.")
    else:
        print("Adopting these means every existing experiment result is from the other")
        print("platform and must be re-run: bash hpc/rerun_experiments.sh")

    if not args.apply:
        print("\n(nothing changed; pass --apply to edit sim/config.py)")
        return 0

    dirty = bool(git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        print("\nrefusing to edit sim/config.py with uncommitted changes in the tree. The "
              "point of adopting in its own commit is that the platform switch is "
              "attributable; commit or stash first, or pass --allow-dirty.", file=sys.stderr)
        return 1

    with open(CONFIG_PATH, encoding="utf-8") as f:
        source = f.read()

    for name, current, fitted, _change, _fname in rows:
        pattern = re.compile(rf"^(    {re.escape(name)}: float = ).*$", re.MULTILINE)
        if not pattern.search(source):
            print(f"could not find the line for {name} in sim/config.py; refusing to "
                  f"guess at a partial edit", file=sys.stderr)
            return 1
        source = pattern.sub(rf"\g<1>{fitted:.6g}", source)

    marker = (f"    # Fitted on {plat.get('device')} (capability {plat.get('capability')}), "
              f"{args.results}/manifest.json\n")
    anchor = "    step_overhead_s: float = "
    source = source.replace(anchor, marker + anchor, 1)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(source)

    print(f"\nedited {CONFIG_PATH}")
    print("Now, in this order:")
    print("  1. bash hpc/rerun_experiments.sh          (CPU, no GPU time)")
    print("  2. commit the config edit AND the new results together")
    print("  3. update the findings docs; every number in them is now from the old round")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
