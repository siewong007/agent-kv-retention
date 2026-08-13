"""Extract the ground-truth engine configuration from a vLLM startup log.

The simulator's `kv_pool_blocks` is currently DERIVED: 16 GB minus weights, divided by
36864 bytes per token. That arithmetic makes assumptions about how much vLLM reserves
for activations, CUDA graphs and fragmentation, and it will be wrong by some margin.
The server prints what it actually allocated. Read that instead.

    python bench/read_server_config.py ~/vllm_calib_server.log

Prints a JSON block and the exact config overrides needed to make the simulator match
the measured server.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# vLLM has moved this line between versions; match on the numbers, not the phrasing.
PATTERNS = {
    "gpu_blocks": [
        r"GPU KV cache size:\s*([\d,]+)\s*tokens",
        r"#\s*GPU blocks:\s*([\d,]+)",
        r"GPU blocks:\s*([\d,]+)",
    ],
    "cpu_blocks": [r"#\s*CPU blocks:\s*([\d,]+)", r"CPU blocks:\s*([\d,]+)"],
    "max_model_len": [r"max_model_len[=:\s]+([\d,]+)"],
    "block_size": [r"block_size[=:\s]+([\d,]+)"],
    "max_num_seqs": [r"max_num_seqs[=:\s]+([\d,]+)"],
    "max_num_batched_tokens": [r"max_num_batched_tokens[=:\s]+([\d,]+)"],
    "gpu_memory_utilization": [r"gpu_memory_utilization[=:\s]+([\d.]+)"],
    "concurrency": [r"maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x"],
}


def scan(text: str) -> dict:
    found: dict = {}
    for key, patterns in PATTERNS.items():
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                raw = match.group(1).replace(",", "")
                found[key] = float(raw) if "." in raw else int(raw)
                found[f"{key}_matched_by"] = pat
                break
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    args = ap.parse_args(argv)

    with open(args.log, encoding="utf-8", errors="replace") as f:
        text = f.read()

    found = scan(text)
    print(json.dumps(found, indent=2))

    gpu_blocks = found.get("gpu_blocks")
    block_size = found.get("block_size", 16)
    if gpu_blocks is None:
        print("\nCOULD NOT FIND the KV cache size in this log.", file=sys.stderr)
        print("Do not fall back to the derived value silently -- find the line, add its "
              "pattern above, and rerun. A wrong pool size rescales every pressure "
              "number in the project.", file=sys.stderr)
        return 1

    # vLLM reports tokens in some versions and blocks in others. Decide from the pattern
    # that actually matched, not from the magnitude: a 16 GB card holds a few hundred
    # thousand tokens, which is well inside the plausible range for a block count too,
    # so any threshold on the number itself will eventually pick wrong.
    matched_by = found.get("gpu_blocks_matched_by", "")
    if "token" in matched_by.lower():
        blocks = int(gpu_blocks // block_size)
        print(f"\n{gpu_blocks} TOKENS (pattern said so) -> {blocks} blocks "
              f"at block_size={block_size}")
    else:
        blocks = int(gpu_blocks)
        print(f"\n{gpu_blocks} BLOCKS")

    print("\nsimulator overrides to match this server:")
    print(f"  --set engine.kv_pool_blocks={blocks}")
    print(f"  --set engine.block_size={block_size}")
    if "max_num_seqs" in found:
        print(f"  --set engine.max_num_seqs={found['max_num_seqs']}")
    if "max_num_batched_tokens" in found:
        print(f"  --set engine.max_num_batched_tokens={found['max_num_batched_tokens']}")

    from sim.config import EngineConfig
    derived = EngineConfig().kv_pool_blocks
    delta = (blocks - derived) / derived * 100 if derived else float("nan")
    print(f"\nderived value in docs/calibration.md: {derived} blocks")
    print(f"measured: {blocks} blocks  ({delta:+.1f}%)")
    if abs(delta) > 20:
        print("\nThe derivation is off by more than 20%. Every pressure-ratio figure in "
              "EXP01-EXP03 is stated against the derived number, so they all shift. "
              "Update docs/calibration.md and re-state the pressure axis before "
              "quoting any of them.")
    return 0


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
