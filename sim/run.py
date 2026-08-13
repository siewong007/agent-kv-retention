"""Single-run entry point.

    python -m sim.run --config configs/base.json --set policy.kind=oracle --out results/x

Every output file carries the full config, the seed and the environment that produced
it. A results file without that block is not a result.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from . import __version__
from .config import Config
from .engine import run_config
from .workload import generate_sessions, workload_summary


def build_metadata(cfg: Config, extra: dict | None = None) -> dict:
    meta = {
        "sim_version": __version__,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "note": "CPU simulator output. Not comparable with measured GPU numbers.",
    }
    if extra:
        meta.update(extra)
    return meta


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one KV-retention simulation.")
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--set", action="append", default=[],
                    help="dotted override, e.g. --set policy.kind=oracle")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--records", action="store_true", help="also dump per-request JSONL")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    overrides = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        overrides[key] = _coerce(raw)
    if overrides:
        cfg = cfg.replace(**overrides)

    sessions = generate_sessions(cfg.workload, cfg.seed)
    wl = workload_summary(sessions)
    result = run_config(cfg, sessions)

    payload = {
        "metadata": build_metadata(cfg, {"workload_summary": wl}),
        "summary": result.summary,
    }
    print(json.dumps(payload, indent=2))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if args.records:
            with open(os.path.join(args.out, "records.jsonl"), "w", encoding="utf-8") as f:
                for rec in result.records:
                    f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
