"""Shared plumbing for experiment scripts: parallel execution and result files."""

from __future__ import annotations

import csv
import random
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.config import Config  # noqa: E402
from sim.engine import run_config  # noqa: E402
from sim.run import build_metadata  # noqa: E402
from sim.workload import generate_sessions  # noqa: E402


# Reported for every run of every experiment.
METRIC_KEYS = [
    "token_hit_rate", "prefill_tokens_computed", "prompt_tokens_total",
    "ttft_p50", "ttft_p95", "e2e_p50", "e2e_p95", "queue_p95",
    "makespan_s", "gpu_busy_s", "gpu_busy_frac",
    "n_preemptions", "n_evictions", "n_protected_evictions",
    "mean_live_sessions", "mean_context_blocks",
    "pressure_measured", "pressure_offered",
    "rm_total", "rm_per_1k_calls",
    "rm_gputime_total", "rm_gputime_per_1k_calls", "n_calls",
]


def run_job(job: dict) -> dict:
    """Run one config. `job['labels']` is copied into the output row verbatim.

    `job['pred_pause']` supplies the per-request predictions the `predict` and
    `predict_terminal` arms need, keyed by (session_id, turn_index).
    """
    cfg = Config.from_dict(job["config"])
    sessions = generate_sessions(cfg.workload, cfg.seed)
    summary = run_config(cfg, sessions, job.get("pred_pause")).summary
    row = dict(job.get("labels", {}))
    row.update({
        "seed": cfg.seed,
        "policy": cfg.policy.kind,
        "const_ttl_s": cfg.policy.const_ttl_s,
        "arrival_mode": cfg.arrival.mode,
        "concurrency": cfg.arrival.concurrency,
        "rate_per_s": cfg.arrival.rate_per_s,
        "pool_blocks": cfg.engine.kv_pool_blocks,
        "pause_median_s": cfg.workload.pause_seconds_median,
        "n_sessions": cfg.workload.n_sessions,
    })
    row.update({k: summary[k] for k in METRIC_KEYS})
    return row


def run_grid(jobs: list[dict], workers: int, label: str = "runs") -> list[dict]:
    print(f"{label}: {len(jobs)} runs on {workers} workers", flush=True)
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1), 1):
            rows.append(row)
            if i % 40 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    return rows


def write_results(out_dir: str, rows: list[dict], base: Config, extra: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    csv_path = os.path.join(out_dir, "runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(build_metadata(base, extra), f, indent=2)
    return csv_path


def read_results(out_dir: str) -> list[dict]:
    """Read runs.csv back, restoring numeric types. Empty cells become None."""
    text_cols = {"policy", "arrival_mode", "condition", "sweep", "loop"}
    rows = []
    with open(os.path.join(out_dir, "runs.csv"), newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {}
            for key, value in raw.items():
                if key in text_cols:
                    row[key] = value
                elif value == "":
                    row[key] = None
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def default_workers() -> int:
    return max(1, (os.cpu_count() or 4) - 2)


def bootstrap_ratio(numer_pairs: list[tuple[float, float]],
                     denom_pairs: list[tuple[float, float]],
                     n_boot: int = 5000, rng_seed: int = 12345) -> dict | None:
    """Paired bootstrap over seeds for a ratio of mean differences.

    The per-seed ratio has an unstable denominator, so its mean is heavy-tailed and its
    standard deviation overstates the uncertainty of the quantity actually being
    claimed. What is claimed is a population ratio -- "across this workload, what share
    of the total saving does X deliver" -- which is a ratio of means, and whose
    uncertainty has to come from resampling whole seeds rather than from averaging
    ratios. Resampling is seeded so the interval is reproducible.
    """
    if not numer_pairs or not denom_pairs or len(numer_pairs) != len(denom_pairs):
        return None
    rng = random.Random(rng_seed)
    n = len(numer_pairs)

    def ratio(idx):
        num = sum(numer_pairs[i][0] - numer_pairs[i][1] for i in idx)
        den = sum(denom_pairs[i][0] - denom_pairs[i][1] for i in idx)
        return num / den if den else None

    point = ratio(range(n))
    if point is None:
        return None
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        val = ratio(idx)
        if val is not None:
            draws.append(val)
    draws.sort()
    if len(draws) < 100:
        return {"point": point, "lo": None, "hi": None, "n_seeds": n}
    return {
        "point": point,
        "lo": draws[int(0.025 * (len(draws) - 1))],
        "hi": draws[int(0.975 * (len(draws) - 1))],
        "n_seeds": n,
        "n_boot": len(draws),
    }


def bootstrap_stat(per_seed_rows: list, statistic, n_boot: int = 5000,
                   rng_seed: int = 12345) -> dict | None:
    """95% interval for an arbitrary statistic of a paired-over-seeds sample.

    `per_seed_rows` is one entry per seed -- whatever `statistic` needs -- and whole
    seeds are resampled together so that any pairing inside a seed is preserved. Used
    for statistics that are not a ratio of differences, such as the spread across
    conditions in EXP02, where the quantity of interest is a max-minus-min of means and
    has no closed-form standard error.
    """
    n = len(per_seed_rows)
    if n < 2:
        return None
    rng = random.Random(rng_seed)
    point = statistic(per_seed_rows)
    if point is None:
        return None
    draws = []
    for _ in range(n_boot):
        sample = [per_seed_rows[rng.randrange(n)] for _ in range(n)]
        val = statistic(sample)
        if val is not None:
            draws.append(val)
    if len(draws) < 100:
        return {"point": point, "lo": None, "hi": None, "n_seeds": n}
    draws.sort()
    return {
        "point": point,
        "lo": draws[int(0.025 * (len(draws) - 1))],
        "hi": draws[int(0.975 * (len(draws) - 1))],
        "n_seeds": n,
        "n_boot": len(draws),
    }
