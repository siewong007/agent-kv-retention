"""Separate `step_overhead_s` from `decode_s_per_seq` with a batch-size sweep.

At batch size 1 those two constants are collinear: every step pays both exactly once,
so only their sum is identifiable. `bench/fit_timing.py` measures the sum (9.06 ms) and
says so; the split it writes into the config is arbitrary. This script identifies them.

Model:

    step_time(B) = step_overhead_s + B * (decode_s_per_seq + ctx * decode_s_per_kv_token)

Sweep B and regress: the intercept is the per-step cost that does not scale with the
batch (reading the weights, launching the graph, the scheduler), and the slope is the
per-sequence cost. `decode_s_per_kv_token` is already measured, so

    decode_s_per_seq = slope - ctx * decode_s_per_kv_token

Method -- two timed runs per batch size, no streaming:

  (a) B concurrent requests with max_tokens=1   -> prefill only
  (b) B concurrent requests with max_tokens=M   -> prefill + (M-1) decode steps

  step_time(B) = (time_b - time_a) / (M - 1)

Subtracting two measurements of the same prefill removes it exactly, rather than
subtracting a model of it. That is what went wrong in the first version of
fit_timing.py, and it is worth not repeating.

A short context is used on purpose: it keeps the prefill share small so the subtraction
is not two large numbers cancelling, and it makes the ctx * decode_s_per_kv_token term
small relative to decode_s_per_seq, which is the quantity being extracted.

    # server (in WSL): bash bench/serve_calib.sh
    python -m bench.fit_batch --base-url http://localhost:8000 --out results/calib
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.fit_timing import _ols, _post, _prompt_of, server_info  # noqa: E402


def timed_batch(base_url: str, model: str, batch: int, ctx: int,
                max_tokens: int) -> tuple[float, list[int]]:
    """Fire `batch` identical requests at once; return wall time and prompt sizes."""
    payload = {
        "model": model,
        "prompt": _prompt_of(ctx),
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }

    def one(_):
        return _post(base_url, "/v1/completions", payload)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch) as pool:
        responses = list(pool.map(one, range(batch)))
    elapsed = time.perf_counter() - t0
    return elapsed, [r["usage"]["prompt_tokens"] for r in responses]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--out", default="results/calib")
    ap.add_argument("--batches", default="1,2,4,8,12,16,24,32,48")
    ap.add_argument("--fit-min-batch", type=int, default=8,
                    help="ignore batches below this when fitting the "
                         "constants the simulator will use")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=192)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args(argv)

    info = server_info(args.base_url)
    model = info["model"]
    print(f"server: {model}")
    print(f"ctx={args.ctx}  decode_tokens={args.decode_tokens}  repeats={args.repeats}")

    batches = [int(v) for v in args.batches.split(",") if v.strip()]
    samples = []
    for batch in batches:
        prefill_times, total_times, prompt_tokens = [], [], None
        for _ in range(args.repeats):
            t_pre, sizes = timed_batch(args.base_url, model, batch, args.ctx, 1)
            t_tot, _ = timed_batch(args.base_url, model, batch, args.ctx,
                                   args.decode_tokens)
            prefill_times.append(t_pre)
            total_times.append(t_tot)
            prompt_tokens = sizes[0]
        t_pre = st.median(prefill_times)
        t_tot = st.median(total_times)
        step_time = (t_tot - t_pre) / (args.decode_tokens - 1)
        samples.append({
            "batch": batch,
            "prompt_tokens": prompt_tokens,
            "prefill_only_s": t_pre,
            "total_s": t_tot,
            "step_time_s": step_time,
        })
        print(f"  B={batch:>3}  prefill {t_pre*1000:>8.1f} ms  total {t_tot*1000:>9.1f} ms"
              f"  -> step {step_time*1000:>7.3f} ms", flush=True)

    from sim.config import EngineConfig
    e = EngineConfig()
    ctx_tokens = samples[0]["prompt_tokens"]
    kv_term = ctx_tokens * e.decode_s_per_kv_token

    def fit_over(min_batch: int) -> dict:
        pts = [s for s in samples if s["batch"] >= min_batch]
        if len(pts) < 2:
            return {}
        intercept, slope, r2 = _ols([p["batch"] for p in pts],
                                    [p["step_time_s"] for p in pts])
        return {
            "min_batch": min_batch,
            "n_points": len(pts),
            "step_overhead_s": intercept,
            "slope_s_per_seq_at_ctx": slope,
            "decode_s_per_seq": slope - kv_term,
            "r_squared": r2,
        }

    # Two fits, because step time is not a straight line in batch size. vLLM captures
    # CUDA graphs at a discrete set of batch sizes and pads up to the next one, so the
    # measured curve is a staircase. A line through the whole range is dominated by the
    # small-batch treads, where padding waste is proportionally largest; a line fitted
    # over the loaded regime describes the regime this project's workloads actually run
    # in (concurrency 12-20 means batches of roughly 10-30).
    fit = {
        "ctx_tokens": ctx_tokens,
        "kv_term_at_ctx": kv_term,
        "decode_s_per_kv_token_used": e.decode_s_per_kv_token,
        "all_batches": fit_over(1),
        "loaded_regime": fit_over(args.fit_min_batch),
        "note": ("Step time is a staircase in batch size (CUDA-graph capture sizes), "
                 "not a line. Use loaded_regime for the simulator; all_batches is kept "
                 "to show how much the choice matters."),
    }
    chosen = fit["loaded_regime"] or fit["all_batches"]
    intercept = chosen["step_overhead_s"]
    slope = chosen["slope_s_per_seq_at_ctx"]
    r2 = chosen["r_squared"]
    decode_s_per_seq = chosen["decode_s_per_seq"]

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "batch_fit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "server": info,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "args": vars(args),
            "fit": fit,
            "samples": samples,
        }, f, indent=2)

    print(f"\n=== separated constants (fitted over B >= {args.fit_min_batch}) ===")
    print(f"  step_overhead_s     {intercept*1000:>9.3f} ms   (intercept, R^2 {r2:.4f})")
    print(f"  per-seq slope       {slope*1000:>9.3f} ms   at ctx={ctx_tokens}")
    print(f"    of which KV read  {kv_term*1000:>9.3f} ms   "
          f"({ctx_tokens} x {e.decode_s_per_kv_token:.3e})")
    print(f"  decode_s_per_seq    {decode_s_per_seq*1000:>9.3f} ms   (slope - KV term)")
    print(f"\n  sum at B=1          {(intercept + slope)*1000:>9.3f} ms   "
          f"(fit_timing measured 9.06 ms at ctx 513)")
    allf = fit["all_batches"]
    print(f"\n  for contrast, the same regression over ALL batch sizes:")
    print(f"    step_overhead_s   {allf['step_overhead_s']*1000:>9.3f} ms   "
          f"(R^2 {allf['r_squared']:.4f})")
    print(f"    decode_s_per_seq  {allf['decode_s_per_seq']*1000:>9.3f} ms")
    print("  The gap between the two is the CUDA-graph staircase, not measurement noise.")
    print(f"\ncurrent config: step_overhead_s={e.step_overhead_s*1000:.3f} ms, "
          f"decode_s_per_seq={e.decode_s_per_seq*1000:.3f} ms")
    if decode_s_per_seq < 0:
        print("\nWARNING decode_s_per_seq came out negative. Either "
              "decode_s_per_kv_token is overstated, or the batch sweep hit a scheduler "
              "limit (max_num_seqs, chunked-prefill interference) and the slope is not "
              "clean. Do not write a negative constant into the config.")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
