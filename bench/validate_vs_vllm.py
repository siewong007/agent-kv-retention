"""Does the simulator behave like vLLM, or only borrow its constants?

Every result in this project comes from a simulator whose timing constants were fitted
to vLLM (docs/calibration.md) but whose *behaviour* has never been compared to it. That
is the largest credibility gap in the work: a fitted constant says the arithmetic is
right, not that the scheduler, the block pool and the prefix-cache lookup behave the
same way under load.

This replays a synthetic agent workload against a live vLLM server with prefix caching
ON, and against the simulator on the byte-identical trace, and compares the one quantity
the whole project turns on: the prefix-cache hit rate.

What makes the comparison fair, and what does not:

  * the same generated sessions drive both, turn by turn, with the same prompt growth;
  * the KV pool is read from the server's startup log and given to the simulator, so
    neither is guessed;
  * pauses are real `sleep`s, so vLLM's scheduler sees the same idle gaps the simulator
    models;
  * prompts are built as raw TOKEN IDS, not text. That gives exact control of both
    length and content, and it is not a convenience: the first version of this script
    sent `"word " * n` for every prompt, which made all forty sessions byte-identical
    and mutual prefixes of one another. vLLM duly reported a 99.6% hit rate -- it was
    serving every session from every other session's cache. The simulator, which models
    a shared system prompt and session-private content after it, reported 91.5%. The
    8-point "disagreement" was entirely an artefact of the replay. Content structure has
    to mirror the model being tested, or the comparison measures the harness.
  * AND the workload is truncated to fit the model's context window. Qwen2.5-3B stops at
    32768 tokens while the generator's sessions reach ~50k, so any turn whose prompt
    would exceed the limit is dropped -- from BOTH sides, so the comparison stays exact.
    The cost is that this validation does not cover the longest contexts, which is
    stated in the output rather than buried.

  * and the two sides are compared on the SAME accounting. vLLM's /metrics prefix-cache
    counters are incremented once per SCHEDULING and counted in tokens, so a preempted
    request that resumes is counted twice in both numerator and denominator. The
    simulator's token_hit_rate counts each prompt once. At pressure 1.02 vLLM's queries
    came to 1.694x the prompt tokens sent, and comparing its ratio to the simulator's
    produced a 25 pp "disagreement" that was almost entirely this.

    The fix that works on a stock build is to remove the preemption rather than to
    correct for it: pass the same --max-num-seqs to this script and to the server, low
    enough that vLLM queues instead of over-committing, which is what the simulator's
    whole-prompt admission already does. The script prints the inflation on both sides
    and only trusts the comparison when both read 1.000. (A server build that reports
    usage.prompt_tokens_details.cached_tokens would give the per-request number
    directly; vLLM 0.26 leaves that field None on /v1/completions.)

A disagreement here is a result, not a bug to be tuned out. If the hit rates differ,
that difference bounds how much any simulated headroom figure can be trusted, and it
belongs in the thesis rather than in a fix.

    # server, prefix caching ON (note: NOT serve_calib.sh, which turns it off):
    #   bash bench/serve_validate.sh
    python -m bench.validate_vs_vllm --sessions 40 --concurrency 10 --pool-blocks 13663 --out results/validate

    # above pressure 1.0, matched admission so the comparison stays valid:
    #   bash bench/serve_validate.sh   # with --max-num-seqs 8 added to the vllm line
    python -m bench.validate_vs_vllm --sessions 60 --concurrency 16 --max-num-seqs 8 --pool-blocks 12865 --out results/validate_matched_admission
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.fit_timing import _post, server_info  # noqa: E402
from sim.config import Config  # noqa: E402
from sim.engine import run_config  # noqa: E402
from sim.workload import generate_sessions, mean_context_blocks  # noqa: E402


def read_prefix_cache_metrics(base_url: str) -> dict:
    """Pull vLLM's own prefix-cache counters.

    Metric names have moved between vLLM versions, so several spellings are accepted and
    the one that matched is reported. If none match, the caller is told rather than
    silently handed a zero.
    """
    import urllib.request

    with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=30) as resp:
        text = resp.read().decode("utf-8")

    wanted = {
        "queries": ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"),
        "hits": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
        # Preemptions need no denominator, which makes them the cleanest behavioural
        # comparison available: the simulator admits only whole prompts and therefore
        # never preempts, while vLLM admits optimistically and does.
        "preemptions": ("vllm:num_preemptions_total",),
    }
    found: dict = {}
    for key, names in wanted.items():
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for name in names:
                if line.startswith(name):
                    try:
                        found[key] = found.get(key, 0.0) + float(line.rsplit(" ", 1)[-1])
                        found[f"{key}_metric"] = name
                    except ValueError:
                        pass
    return found


class Replayer:
    """Drives generated sessions against a live server, `concurrency` sessions at a time."""

    def __init__(self, base_url: str, model: str, pause_scale: float,
                 shared_prefix_tokens: int, vocab_range=(1000, 100000)):
        self.base_url = base_url
        self.model = model
        self.pause_scale = pause_scale
        self.vocab_range = vocab_range
        # One fixed prefix for every session, matching WorkloadConfig.system_prompt_tokens.
        srng = random.Random(20260802)
        self.shared_prefix = [srng.randrange(*vocab_range)
                              for _ in range(shared_prefix_tokens)]
        self.records: list[dict] = []
        self.lock = threading.Lock()

    def run_session(self, session) -> None:
        # Token ids, built to mirror the simulator's content model exactly:
        #   [shared system prefix]  identical across every session, so it is the only
        #                           cross-session reuse, as in sim/cache.py
        #   [session-private tail]  grown turn by turn, never shared with another session
        ids = list(self.shared_prefix)
        rng = random.Random(1_000_000 + session.session_id)
        for turn in session.turns:
            while len(ids) < turn.prompt_tokens:
                ids.append(rng.randrange(*self.vocab_range))
            payload = {
                "model": self.model,
                "prompt": ids[:turn.prompt_tokens],
                "max_tokens": turn.output_tokens,
                "min_tokens": turn.output_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            }
            t0 = time.perf_counter()
            resp = _post(self.base_url, "/v1/completions", payload)
            elapsed = time.perf_counter() - t0
            # Per-request cached tokens, if the server reports them. This is the only
            # quantity directly comparable to the simulator's token_hit_rate: the
            # /metrics counters are per SCHEDULING, so under preemption they count a
            # resumed request twice and their ratio is not a per-request hit rate.
            usage = resp["usage"]
            details = usage.get("prompt_tokens_details") or {}
            with self.lock:
                self.records.append({
                    "session_id": session.session_id,
                    "turn": turn.index,
                    "intended_prompt_tokens": turn.prompt_tokens,
                    "actual_prompt_tokens": usage["prompt_tokens"],
                    "cached_prompt_tokens": details.get("cached_tokens"),
                    "output_tokens": usage["completion_tokens"],
                    "elapsed_s": elapsed,
                })
            if turn.pause_after_s > 0:
                time.sleep(turn.pause_after_s * self.pause_scale)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-blocks", type=int, required=True,
                    help="KV blocks the server reported at startup "
                         "(bench/read_server_config.py); given to the simulator so "
                         "neither side is guessed")
    ap.add_argument("--max-prompt-tokens", type=int, default=30000,
                    help="drop turns whose prompt would exceed the model's context "
                         "window; applied identically to the simulator's trace")
    ap.add_argument("--pause-scale", type=float, default=1.0,
                    help="multiply every pause. 1.0 keeps the workload's real timing; "
                         "lower values shorten the run but change what the scheduler "
                         "sees, so the comparison is only clean at 1.0")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="cap concurrently RUNNING sequences on the simulator side. "
                         "Pass the same value to the server (--max-num-seqs) to remove "
                         "the admission difference between the two: vLLM then queues "
                         "instead of over-committing, which is what the simulator's "
                         "whole-prompt admission already does. With preemption gone, "
                         "vLLM's per-scheduling counters collapse to a per-request hit "
                         "rate and the eviction models become directly comparable. "
                         "This isolates a mechanism; it is NOT vLLM as deployed")
    ap.add_argument("--out", default="results/validate")
    args = ap.parse_args(argv)

    info = server_info(args.base_url)
    model = info["model"]

    cfg = Config().replace(**{
        "workload.n_sessions": args.sessions,
        "arrival.concurrency": args.concurrency,
        "engine.kv_pool_blocks": args.pool_blocks,
        "policy.kind": "lru",
        "seed": args.seed,
    })
    if args.max_num_seqs is not None:
        cfg = cfg.replace(**{"engine.max_num_seqs": args.max_num_seqs})
    sessions = generate_sessions(cfg.workload, args.seed)

    # Truncate to the model's context window, on BOTH sides. A session is cut at the
    # first turn that would not fit rather than having that turn dropped from the middle,
    # because the prefix chain is what is being measured and a hole in it would not be a
    # shorter version of the same workload.
    dropped = 0
    for sess in sessions:
        keep = 0
        for t in sess.turns:
            if t.prompt_tokens > args.max_prompt_tokens:
                break
            keep += 1
        dropped += len(sess.turns) - keep
        sess.turns = sess.turns[:keep]
    sessions = [s for s in sessions if s.turns]
    total_turns = sum(len(s.turns) for s in sessions)

    ctx_blocks = mean_context_blocks(sessions, cfg.engine.block_size)
    pressure = args.concurrency * ctx_blocks / args.pool_blocks

    print(f"model            : {model}")
    print(f"pool blocks      : {args.pool_blocks} (read from the server)")
    print(f"sessions         : {args.sessions} at concurrency {args.concurrency}")
    print(f"offered pressure : {pressure:.2f}")
    print(f"pause scale      : {args.pause_scale}")
    print(f"max_num_seqs     : {cfg.engine.max_num_seqs} "
          f"(must match the server's --max-num-seqs)")
    print(f"turns            : {total_turns} kept, {dropped} dropped for exceeding "
          f"{args.max_prompt_tokens} prompt tokens")

    before = read_prefix_cache_metrics(args.base_url)
    if "queries" not in before:
        print("\nERROR: no prefix-cache metric found on /metrics. Either the server was "
              "started with prefix caching disabled, or this vLLM version renamed the "
              "counter. Refusing to report a hit rate that would be a guess.",
              file=sys.stderr)
        return 1

    replayer = Replayer(args.base_url, model, args.pause_scale,
                        cfg.workload.system_prompt_tokens)
    print("\nreplaying against vLLM ...", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(replayer.run_session, sessions))
    wall = time.perf_counter() - t0

    after = read_prefix_cache_metrics(args.base_url)
    q = after["queries"] - before["queries"]
    h = after["hits"] - before["hits"]
    vllm_hit = h / q if q else float("nan")

    print("running the simulator on the identical trace ...", flush=True)
    sim = run_config(cfg, sessions).summary

    intended = sum(r["intended_prompt_tokens"] for r in replayer.records)
    actual = sum(r["actual_prompt_tokens"] for r in replayer.records)

    # Is the /metrics ratio a per-request hit rate at all? It is only when every prompt
    # was scheduled exactly once. Anything above 1.0 means requests were preempted and
    # re-queried, and both counters are inflated by an unknown amount.
    inflation = q / actual if actual else float("nan")
    per_request = [r["cached_prompt_tokens"] for r in replayer.records]
    have_per_request = all(x is not None for x in per_request)
    vllm_per_request_hit = (sum(per_request) / actual
                            if have_per_request and actual else None)

    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": info,
        "config": cfg.to_dict(),
        "offered_pressure": pressure,
        "turns_kept": total_turns,
        "turns_dropped_for_context_limit": dropped,
        "max_prompt_tokens": args.max_prompt_tokens,
        "vllm": {
            "prefix_cache_queries": q,
            "prefix_cache_hits": h,
            "hit_rate": vllm_hit,
            "query_inflation": inflation,
            "per_request_hit_rate": vllm_per_request_hit,
            "num_preemptions": (after.get("preemptions", 0.0)
                                - before.get("preemptions", 0.0)),
            "metric_used": after.get("queries_metric"),
            "wall_s": wall,
            "n_calls": len(replayer.records),
            "ttft_proxy_p50_s": st.median([r["elapsed_s"] for r in replayer.records]),
        },
        "simulator": {
            "hit_rate": sim["token_hit_rate"],
            "query_hit_rate": sim["query_hit_rate"],
            "query_inflation": sim["query_tokens"] / sim["prompt_tokens_total"],
            "n_preemptions": sim["n_preemptions"],
            "makespan_s": sim["makespan_s"],
            "n_calls": sim["n_calls"],
            "prefill_tokens_computed": sim["prefill_tokens_computed"],
        },
        "tokenisation": {
            "intended_prompt_tokens": intended,
            "actual_prompt_tokens": actual,
            "ratio_actual_over_intended": actual / intended if intended else float("nan"),
        },
        "records": replayer.records,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "validation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== prefix-cache hit rate ===")
    sim_infl = sim["query_tokens"] / sim["prompt_tokens_total"]
    n_preempt = after.get("preemptions", 0.0) - before.get("preemptions", 0.0)
    print(f"  query inflation:  vLLM {inflation:.4f}x   simulator {sim_infl:.4f}x")
    print(f"  preemptions:      vLLM {n_preempt:.0f}          "
          f"simulator {sim['n_preemptions']}")
    if vllm_per_request_hit is not None:
        print(f"  per-request:      vLLM {vllm_per_request_hit:.4f}   "
              f"simulator {sim['token_hit_rate']:.4f}   "
              f"delta {sim['token_hit_rate']-vllm_per_request_hit:+.4f}")
    print(f"  per-scheduling:   vLLM {vllm_hit:.4f}   "
          f"simulator {sim['query_hit_rate']:.4f}   "
          f"delta {sim['query_hit_rate']-vllm_hit:+.4f}"
          f"   (metric {after.get('queries_metric')})")
    if inflation > 1.001:
        print("\n  WARNING: vLLM scheduled prompts more than once, so its /metrics ratio")
        print("  counts resumed requests twice in BOTH numerator and denominator. It is")
        print("  NOT a per-request hit rate and must not be compared to one -- that")
        print("  mistake produced a 25 pp phantom disagreement. Either match the two")
        print("  admission models with --max-num-seqs until this reads 1.000, or use a")
        print("  server build that reports usage.prompt_tokens_details.cached_tokens.")
    else:
        print("\n  Inflation is 1.000 on both sides: every prompt was scheduled exactly")
        print("  once, so the two hit rates are the same quantity and directly comparable.")
    print()
    print(f"  tokenisation: replay sent {actual/intended:.3f}x the intended tokens")
    print(f"  wall clock:   vLLM {wall:.0f}s, simulator predicted "
          f"{sim['makespan_s']:.0f}s ({sim['makespan_s']/wall:.2f}x)")
    print()
    print("  vLLM's /metrics counters are in TOKENS, incremented once per scheduling.")
    print("  The simulator reports both accountings so the comparison can be made on")
    print("  whichever one the server actually produced.")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
