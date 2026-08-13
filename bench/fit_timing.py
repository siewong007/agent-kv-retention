"""Fit the simulator's timing constants to a real vLLM server.

The simulator's step-time model is

    step_time = step_overhead_s
              + prefill_tokens_in_step / prefill_tps
              + sum over decoding seqs of
                (decode_s_per_seq + ctx_tokens * decode_s_per_kv_token)

Those four numbers are currently DERIVED from hardware specs, not measured, which means
every latency and ringgit figure the simulator produces has an unknown scale factor on
it. This script measures them.

Method -- isolate one term at a time rather than fitting all four jointly, because a
joint fit on end-to-end latency is badly conditioned and will happily trade prefill
throughput against decode cost:

  1. prefill_tps          one request at a time, max_tokens=1, sweep prompt length.
                          TTFT ~ a + prompt_tokens / prefill_tps. Slope gives the
                          throughput, and the intercept is a fixed per-request cost.
  2. decode terms         one request at a time, fixed short prompt, sweep output
                          length; then repeat at several context lengths. Time per
                          output token ~ step_overhead + decode_s_per_seq
                          + ctx * decode_s_per_kv_token, so sweeping ctx gives the
                          per-KV-token slope and the intercept gives the rest.
  3. step_overhead_s      separated from decode_s_per_seq by sweeping batch size: the
                          part that does not scale with the number of sequences.

Prefix caching is DISABLED for the fit. A cache hit is exactly what we are trying to
model the cost of, so it must not contaminate the measurement of what a miss costs.

Every measurement writes its raw samples alongside the fit, so the regression can be
re-checked without re-running the GPU.

    # server (in WSL):
    #   vllm serve Qwen/Qwen2.5-3B-Instruct --no-enable-prefix-caching --port 8000
    python -m bench.fit_timing --base-url http://localhost:8000 --out results/calib
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _post(base_url: str, path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(base_url: str, path: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def server_info(base_url: str) -> dict:
    """Model id and, crucially, the KV pool size the server actually allocated.

    kv_pool_blocks must be READ from the server, never derived. The derivation in
    docs/calibration.md is a sanity check on this number, not a substitute for it.
    """
    models = json.loads(_get(base_url, "/v1/models"))
    info = {"model": models["data"][0]["id"]}
    try:
        metrics = _get(base_url, "/metrics")
        for line in metrics.splitlines():
            if line.startswith("#"):
                continue
            for key, name in (("vllm:gpu_cache_usage_perc", "gpu_cache_usage_perc"),
                              ("vllm:num_requests_running", "num_requests_running")):
                if line.startswith(key):
                    info[name] = float(line.rsplit(" ", 1)[-1])
        info["metrics_available"] = True
    except (urllib.error.URLError, ValueError, IndexError):
        info["metrics_available"] = False
    return info


def _prompt_of(tokens: int, filler: str = "word ") -> str:
    """A prompt of roughly `tokens` tokens. Exact count comes back in usage."""
    return filler * max(1, tokens)


def measure_prefill(base_url: str, model: str, lengths: list[int],
                    repeats: int) -> list[dict]:
    """TTFT for a single request with max_tokens=1, over a sweep of prompt lengths."""
    samples = []
    for target in lengths:
        for rep in range(repeats):
            payload = {
                "model": model,
                "prompt": _prompt_of(target),
                "max_tokens": 1,
                "temperature": 0.0,
            }
            t0 = time.perf_counter()
            resp = _post(base_url, "/v1/completions", payload)
            elapsed = time.perf_counter() - t0
            samples.append({
                "phase": "prefill",
                "target_prompt_tokens": target,
                "prompt_tokens": resp["usage"]["prompt_tokens"],
                "output_tokens": resp["usage"]["completion_tokens"],
                "elapsed_s": elapsed,
                "repeat": rep,
            })
            print(f"  prefill {resp['usage']['prompt_tokens']:>7} tok "
                  f"-> {elapsed*1000:>8.1f} ms", flush=True)
    return samples


def measure_decode(base_url: str, model: str, contexts: list[int],
                   output_tokens: int, repeats: int) -> list[dict]:
    """Total time for a fixed number of output tokens, at several context lengths.

    Subtracting the prefill model from the total leaves decode time, and its slope
    against context length is decode_s_per_kv_token.
    """
    samples = []
    for ctx in contexts:
        for rep in range(repeats):
            payload = {
                "model": model,
                "prompt": _prompt_of(ctx),
                "max_tokens": output_tokens,
                "min_tokens": output_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            }
            t0 = time.perf_counter()
            resp = _post(base_url, "/v1/completions", payload)
            elapsed = time.perf_counter() - t0
            samples.append({
                "phase": "decode",
                "target_prompt_tokens": ctx,
                "prompt_tokens": resp["usage"]["prompt_tokens"],
                "output_tokens": resp["usage"]["completion_tokens"],
                "elapsed_s": elapsed,
                "repeat": rep,
            })
            print(f"  decode  ctx {resp['usage']['prompt_tokens']:>7} tok "
                  f"x {resp['usage']['completion_tokens']:>4} out "
                  f"-> {elapsed*1000:>8.1f} ms", flush=True)
    return samples


def _ols2(xs: list[float], ys: list[float]) -> dict:
    """Least squares y = a + b x + c x^2, via numpy.

    Prefill is not linear in prompt length: attention is quadratic, and over the range
    an agent workload actually uses (8k-32k tokens) that term is large enough that a
    straight line fits with a NEGATIVE intercept -- a fixed cost of minus 38 ms, which
    is not a thing. The quadratic term is what the simulator's linear model is missing.
    """
    import numpy as np

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    coeffs = np.polyfit(x, y, 2)  # c, b, a
    pred = np.polyval(coeffs, x)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "intercept_s": float(coeffs[2]),
        "linear_s_per_token": float(coeffs[1]),
        "quadratic_s_per_token2": float(coeffs[0]),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "predict": lambda n: float(np.polyval(coeffs, n)),
    }


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b x. Returns (intercept, slope, r_squared)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return my, 0.0, 0.0
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return intercept, slope, r2


def fit(samples: list[dict], output_tokens: int) -> dict:
    """Turn raw timings into the four simulator constants."""
    prefill = [s for s in samples if s["phase"] == "prefill"]
    decode = [s for s in samples if s["phase"] == "decode"]

    # Median across repeats: the first sample of a length pays for cold kernels and
    # a mean would carry that into the slope.
    def medians(rows):
        by_len: dict[int, list[float]] = {}
        for s in rows:
            by_len.setdefault(s["prompt_tokens"], []).append(s["elapsed_s"])
        return sorted((k, st.median(v)) for k, v in by_len.items())

    pf = medians(prefill)
    pf_x = [x for x, _ in pf]
    pf_y = [y for _, y in pf]
    a_pf, b_pf, r2_pf = _ols(pf_x, pf_y)
    quad = _ols2(pf_x, pf_y)
    prefill_measured = dict(pf)  # exact prompt_tokens -> median seconds

    # The simulator charges prefill linearly, so it needs a single tokens/second number.
    # Take it from the range the workload actually lives in rather than from a global
    # line: a fit dominated by short prompts overstates throughput at agent context
    # lengths, which is where every result in this project sits.
    long_pts = [(x, y) for x, y in pf if x >= 4000]
    if len(long_pts) >= 2:
        tps_long = (long_pts[-1][0] - long_pts[0][0]) / (long_pts[-1][1] - long_pts[0][1])
    else:
        tps_long = float("nan")

    result = {
        "prefill": {
            "linear_fit": {
                "fixed_cost_s": a_pf,
                "seconds_per_prompt_token": b_pf,
                "prefill_tps": 1.0 / b_pf if b_pf > 0 else float("nan"),
                "r_squared": r2_pf,
            },
            "quadratic_fit": {k: v for k, v in quad.items() if k != "predict"},
            "prefill_tps_long_context": tps_long,
            "n_points": len(pf),
            "note": ("A negative fixed_cost_s in the linear fit means prefill is "
                     "superlinear over this range -- attention is quadratic in prompt "
                     "length. The simulator's model is linear, so use "
                     "prefill_tps_long_context, fitted over >=4k tokens, and record the "
                     "linear model as a known fidelity limit."),
        }
    }

    if decode:
        dec = medians(decode)
        # Subtract MEASURED prefill at the same prompt length wherever possible, and
        # fall back to the quadratic model only where no matching measurement exists.
        # Subtracting a model that does not fit (see above) was what drove the first
        # version of this fit to R^2 = 0.30.
        per_token = []
        for ctx, total in dec:
            if ctx in prefill_measured:
                prefill_time = prefill_measured[ctx]
                source = "measured"
            else:
                prefill_time = quad["predict"](ctx)
                source = "quadratic_model"
            per_token.append((ctx, (total - prefill_time) / output_tokens, source))

        a_dec, b_dec, r2_dec = _ols([c for c, _, _ in per_token],
                                    [t for _, t, _ in per_token])
        result["decode"] = {
            # At batch size 1 the fit cannot separate the fixed per-step cost from the
            # fixed per-sequence cost -- they are collinear. Reported jointly, and split
            # by a batch sweep in a later pass.
            "step_plus_seq_fixed_s": a_dec,
            "decode_s_per_kv_token": b_dec,
            "r_squared": r2_dec,
            "n_points": len(per_token),
            "output_tokens": output_tokens,
            "per_token_samples": [
                {"ctx_tokens": c, "seconds_per_output_token": t, "prefill_source": s}
                for c, t, s in per_token
            ],
            "note": ("step_overhead_s and decode_s_per_seq are collinear at batch size 1; "
                     "their sum is reported here. Sweep batch size to separate them."),
        }
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--out", default="results/calib")
    ap.add_argument("--prefill-lengths", default="256,512,1024,2048,4096,8192,16384")
    ap.add_argument("--decode-contexts", default="512,4096,8192,16384")
    ap.add_argument("--decode-output-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args(argv)

    info = server_info(args.base_url)
    model = info["model"]
    print(f"server: {model}")
    print(f"metrics endpoint: {'available' if info['metrics_available'] else 'MISSING'}")

    print("warmup")
    for _ in range(args.warmup):
        _post(args.base_url, "/v1/completions",
              {"model": model, "prompt": _prompt_of(512), "max_tokens": 8,
               "temperature": 0.0})

    lengths = [int(v) for v in args.prefill_lengths.split(",") if v.strip()]
    contexts = [int(v) for v in args.decode_contexts.split(",") if v.strip()]

    print("measuring prefill")
    samples = measure_prefill(args.base_url, model, lengths, args.repeats)
    print("measuring decode")
    samples += measure_decode(args.base_url, model, contexts,
                              args.decode_output_tokens, args.repeats)

    fits = fit(samples, args.decode_output_tokens)

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "server": info,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "fit": fits,
        "samples": samples,
        "warning": ("Fitted with prefix caching DISABLED. These constants describe the "
                    "cost of a cache MISS, which is what the simulator charges."),
    }
    path = os.path.join(args.out, "timing_fit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== fitted constants ===")
    pf = fits["prefill"]
    lin, qd = pf["linear_fit"], pf["quadratic_fit"]
    print(f"  prefill_tps (>=4k ctx) {pf['prefill_tps_long_context']:>12.0f}   "
          f"<- use this one")
    print(f"  prefill_tps (all, lin) {lin['prefill_tps']:>12.0f}   "
          f"(R^2 {lin['r_squared']:.4f}, intercept {lin['fixed_cost_s']*1000:+.1f} ms)")
    print(f"  quadratic term         {qd['quadratic_s_per_token2']:>12.3e} s/tok^2  "
          f"(R^2 {qd['r_squared']:.4f})")
    if lin["fixed_cost_s"] < 0:
        print("    NOTE negative intercept: prefill is superlinear here, the "
              "simulator's linear model understates long prompts.")
    if "decode" in fits:
        dc = fits["decode"]
        print(f"  decode_s_per_kv_token  {dc['decode_s_per_kv_token']:>12.3e}   "
              f"(R^2 {dc['r_squared']:.4f})")
        print(f"  step+seq fixed         {dc['step_plus_seq_fixed_s']*1000:>12.3f} ms   "
              f"(collinear at batch 1; see note)")
        srcs = {s["prefill_source"] for s in dc["per_token_samples"]}
        print(f"    prefill subtracted from: {', '.join(sorted(srcs))}")
    print(f"\ncurrent simulator defaults, for comparison:")
    from sim.config import EngineConfig
    e = EngineConfig()
    print(f"  prefill_tps            {e.prefill_tps:>12.0f}")
    print(f"  decode_s_per_kv_token  {e.decode_s_per_kv_token:>12.3e}")
    print(f"  step_overhead_s        {e.step_overhead_s*1000:>12.3f} ms")
    print(f"  decode_s_per_seq       {e.decode_s_per_seq*1000:>12.3f} ms")
    print(f"\nwrote {path}")
    print("Update docs/calibration.md: these rows move from DERIVED to MEASURED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
