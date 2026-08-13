"""Turn a real agent trace into WorkloadConfig parameters.

The pause and tool-result distributions are the last INVENTED numbers in the project,
and every headroom figure is a function of them. This fits them from a recorded agent
run. See docs/trace_schema.md for the input format.

    python -m bench.fit_workload --trace runs.jsonl --out results/workload_fit
    python -m bench.fit_workload --self-test

The self-test is not optional decoration. It generates traces from the project's own
generator at known parameters and checks the fitter recovers them. A fitter that cannot
invert its own generator has no business being pointed at real data, and the failure
mode is silent -- it returns plausible numbers either way.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.config import Config, WorkloadConfig  # noqa: E402
from sim.workload import generate_sessions  # noqa: E402

REQUIRED = ("session_id", "turn", "t_request", "t_response_end",
            "prompt_tokens", "output_tokens")


def load_trace(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: not valid JSON ({exc})")
            missing = [k for k in REQUIRED if k not in obj]
            if missing:
                raise SystemExit(
                    f"{path}:{lineno}: missing required field(s) {missing}. "
                    f"See docs/trace_schema.md")
            rows.append(obj)
    if not rows:
        raise SystemExit(f"{path}: no records")
    return rows


def _lognormal_params(values: list[float]) -> tuple[float, float, int]:
    """Return (median, sigma, n_used) for strictly positive samples.

    Fitted in log space: the median is exp(mean of logs) and sigma is the standard
    deviation of the logs, matching how sim/workload.py parameterises its draws.
    Non-positive values are dropped and counted, because a zero pause is a recording
    artefact rather than a sample from the distribution being fitted.
    """
    logs = [math.log(v) for v in values if v > 0]
    if len(logs) < 2:
        return (0.0, 0.0, len(logs))
    return (math.exp(st.fmean(logs)), st.stdev(logs), len(logs))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def fit(rows: list[dict]) -> dict:
    by_session: dict = defaultdict(list)
    for r in rows:
        by_session[r["session_id"]].append(r)
    for turns in by_session.values():
        turns.sort(key=lambda r: r["turn"])

    pauses: list[float] = []
    pauses_by_tool: dict = defaultdict(list)
    outputs: list[float] = []
    tool_results: list[float] = []
    turn_counts: list[int] = []
    first_prompts: list[int] = []
    tools_seen: set = set()
    same_tool, tool_transitions = 0, 0
    dropped_negative_pause = 0
    derived_tool_results = 0

    for turns in by_session.values():
        turn_counts.append(len(turns))
        first_prompts.append(turns[0]["prompt_tokens"])
        for i, cur in enumerate(turns):
            outputs.append(float(cur["output_tokens"]))
            tool = cur.get("tool")
            if tool is not None:
                tools_seen.add(tool)
            if i + 1 >= len(turns):
                continue  # last turn has no pause and no following tool result
            nxt = turns[i + 1]

            # Tool time only: the gap between this response ending and the next request
            # arriving. Using request-to-request would fold engine latency into the
            # pause, and engine latency is what the simulator predicts -- circular.
            pause = nxt["t_request"] - cur["t_response_end"]
            if pause > 0:
                pauses.append(pause)
                if tool is not None:
                    pauses_by_tool[tool].append(pause)
            else:
                dropped_negative_pause += 1

            if "tool_result_tokens" in cur:
                tool_results.append(float(cur["tool_result_tokens"]))
            else:
                grown = (nxt["prompt_tokens"] - cur["prompt_tokens"]
                         - cur["output_tokens"])
                if grown > 0:
                    tool_results.append(float(grown))
                    derived_tool_results += 1

            nxt_tool = nxt.get("tool")
            if tool is not None and nxt_tool is not None:
                tool_transitions += 1
                if tool == nxt_tool:
                    same_tool += 1

    explicit_sys = [r["system_prompt_tokens"] for r in rows
                    if "system_prompt_tokens" in r]
    if explicit_sys:
        system_prompt_tokens = int(st.median(explicit_sys))
        sys_source = "explicit field"
    else:
        system_prompt_tokens = int(min(first_prompts)) if first_prompts else 0
        sys_source = ("estimated as min turn-0 prompt -- an UPPER BOUND, since that "
                      "prompt still contains the smallest observed task statement. "
                      "Record system_prompt_tokens explicitly to remove the bias.")

    task_tokens = [float(p - system_prompt_tokens) for p in first_prompts
                   if p > system_prompt_tokens]

    pause_med, pause_marginal_sig, n_pause = _lognormal_params(pauses)

    # Sigma must be fitted WITHIN tool. The marginal spread of all pauses pooled is
    # sqrt(sigma^2 + tool_pause_spread^2) -- it mixes the within-tool variability that
    # `pause_seconds_sigma` means with the between-tool variability that
    # `tool_pause_spread` means. Fitting it marginally double-counts the tool effect and
    # inflates sigma by 24-40% at the spreads this project uses.
    within_logs: list[float] = []
    for tool_pauses in pauses_by_tool.values():
        logs = [math.log(v) for v in tool_pauses if v > 0]
        if len(logs) < 2:
            continue
        centre = st.fmean(logs)
        within_logs.extend(x - centre for x in logs)
    if len(within_logs) >= 2:
        pause_sig = st.stdev(within_logs)
        sigma_source = "within-tool (pooled)"
    else:
        pause_sig = pause_marginal_sig
        sigma_source = "marginal (no tool field: conflated with tool_pause_spread)"
    out_med, out_sig, n_out = _lognormal_params(outputs)
    tr_med, tr_sig, n_tr = _lognormal_params(tool_results)
    task_med, task_sig, n_task = _lognormal_params(task_tokens)

    # How much of a tool's runtime is explained by which tool it is. This is the ceiling
    # on what any predictor can learn from tool identity alone, so it decides whether the
    # week-2 predictor result means anything.
    per_tool_medians = [st.median(v) for v in pauses_by_tool.values() if len(v) >= 3]
    if len(per_tool_medians) >= 2:
        tool_pause_spread = st.stdev([math.log(m) for m in per_tool_medians if m > 0])
    else:
        tool_pause_spread = 0.0

    # Invert the generator's transition rule. It keeps the previous tool with
    # probability p and otherwise resamples UNIFORMLY, which can land on the same tool
    # again, so the observable is P(same) = p + (1-p)/n rather than p itself.
    p_same = (same_tool / tool_transitions) if tool_transitions else 0.0
    n_tools_obs = max(1, len(tools_seen))
    if n_tools_obs > 1:
        markov = (p_same - 1.0 / n_tools_obs) / (1.0 - 1.0 / n_tools_obs)
        markov = min(1.0, max(0.0, markov))
    else:
        markov = 0.0

    workload = {
        "n_sessions": len(by_session),
        "turns_min": int(_percentile([float(c) for c in turn_counts], 0.05)),
        "turns_max": int(_percentile([float(c) for c in turn_counts], 0.95)),
        "system_prompt_tokens": system_prompt_tokens,
        "task_tokens_median": task_med,
        "task_tokens_sigma": task_sig,
        "output_tokens_median": out_med,
        "output_tokens_sigma": out_sig,
        "tool_result_tokens_median": tr_med,
        "tool_result_tokens_sigma": tr_sig,
        "pause_seconds_median": pause_med,
        "pause_seconds_sigma": pause_sig,
        "n_tools": max(1, len(tools_seen)),
        "tool_pause_spread": tool_pause_spread,
        "tool_markov_self_prob": markov,
    }

    provenance = {
        "n_calls": len(rows),
        "n_sessions": len(by_session),
        "n_pause_samples": n_pause,
        "n_output_samples": n_out,
        "n_tool_result_samples": n_tr,
        "n_task_samples": n_task,
        "tool_result_derived_from_prompt_growth": derived_tool_results,
        "dropped_non_positive_pauses": dropped_negative_pause,
        "system_prompt_source": sys_source,
        "n_tools_with_enough_samples": len(per_tool_medians),
        "tool_transitions": tool_transitions,
        "pause_sigma_source": sigma_source,
        "pause_sigma_marginal": pause_marginal_sig,
        "p_same_tool_observed": p_same,
        "system_prompt_is_upper_bound": not explicit_sys,
        "warnings": [],
    }
    if n_pause < 200:
        provenance["warnings"].append(
            f"only {n_pause} pause samples; the sigma of a lognormal is poorly "
            f"determined below a few hundred")
    if not tools_seen:
        provenance["warnings"].append(
            "no `tool` field anywhere: tool_pause_spread forced to 0 and "
            "tool_markov_self_prob to 0. Any predictor result fitted on this trace "
            "would be measuring a world with no tool structure in it at all")
    if len(per_tool_medians) < 2 and tools_seen:
        provenance["warnings"].append(
            "fewer than two tools have >=3 pause samples; tool_pause_spread is not "
            "identifiable and was set to 0")
    if dropped_negative_pause:
        provenance["warnings"].append(
            f"{dropped_negative_pause} non-positive pauses dropped -- overlapping "
            f"timestamps suggest the trace was collected under concurrency")

    return {"workload": workload, "provenance": provenance}


# --------------------------------------------------------------------------- self-test

SELF_TEST_CASES = [
    {"pause_seconds_median": 2.0, "pause_seconds_sigma": 1.0,
     "tool_pause_spread": 0.9, "tool_markov_self_prob": 0.45},
    {"pause_seconds_median": 0.5, "pause_seconds_sigma": 0.6,
     "tool_pause_spread": 0.0, "tool_markov_self_prob": 0.10},
    {"pause_seconds_median": 12.0, "pause_seconds_sigma": 1.4,
     "tool_pause_spread": 1.5, "tool_markov_self_prob": 0.80},
]

# Tolerances are relative for scale parameters and absolute for the two probabilities.
TOLERANCE = {
    "pause_seconds_median": ("rel", 0.15),
    "pause_seconds_sigma": ("rel", 0.15),
    "output_tokens_median": ("rel", 0.15),
    "output_tokens_sigma": ("rel", 0.15),
    "tool_result_tokens_median": ("rel", 0.20),
    "tool_result_tokens_sigma": ("rel", 0.20),
    "tool_markov_self_prob": ("abs", 0.08),
    "tool_pause_spread": ("abs", 0.35),
    # Upper-bounded, not identified: min turn-0 prompt still carries the
    # smallest task statement. Bias is positive and workload-dependent.
    "system_prompt_tokens": ("rel", 0.12),
}


def synth_trace(cfg: WorkloadConfig, seed: int) -> list[dict]:
    """Emit a schema-conforming trace from the project's own generator.

    Timestamps are constructed so that the *pause* is exactly the generator's value and
    the engine time is not zero -- if service time were zero, a fitter that mistakenly
    used request-to-request gaps would still pass, and that is the bug worth catching.
    """
    rows = []
    for s in generate_sessions(cfg, seed):
        now = 1_700_000_000.0 + s.session_id * 3600.0
        for t in s.turns:
            service = 0.05 + t.prompt_tokens * 1e-5 + t.output_tokens * 5e-4
            row = {
                "session_id": f"s{s.session_id}",
                "turn": t.index,
                "t_request": now,
                "t_response_end": now + service,
                "prompt_tokens": t.prompt_tokens,
                "output_tokens": t.output_tokens,
                "tool": f"tool_{t.tool_id}",
            }
            rows.append(row)
            now = now + service + t.pause_after_s
    return rows


def self_test(n_sessions: int = 400) -> int:
    print("self-test: can the fitter recover known parameters from its own generator?")
    print("(system_prompt_tokens is compared loosely on purpose: without an explicit "
          "field it\n can only be bounded from above -- see docs/trace_schema.md)\n")
    failures = 0
    for i, overrides in enumerate(SELF_TEST_CASES):
        cfg = Config().replace(**{f"workload.{k}": v for k, v in overrides.items()})
        cfg = cfg.replace(**{"workload.n_sessions": n_sessions})
        truth = cfg.workload
        rows = synth_trace(truth, seed=100 + i)
        got = fit(rows)["workload"]

        print(f"case {i + 1}: pause median {overrides['pause_seconds_median']}s, "
              f"spread {overrides['tool_pause_spread']}, "
              f"markov {overrides['tool_markov_self_prob']}  "
              f"({len(rows)} calls)")
        for field, (kind, tol) in TOLERANCE.items():
            want = getattr(truth, field)
            have = got[field]
            if kind == "rel":
                ok = want == 0 or abs(have - want) / abs(want) <= tol
                err = f"{(have - want) / want * 100:+.1f}%" if want else "n/a"
            else:
                ok = abs(have - want) <= tol
                err = f"{have - want:+.3f}"
            flag = "ok  " if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"   {flag} {field:<28} want {want:>10.4g}  got {have:>10.4g}  {err}")
        print()

    if failures:
        print(f"{failures} parameter(s) outside tolerance -- do not trust this fitter "
              f"on real data until they are understood")
        return 1
    print("all parameters recovered within tolerance")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", help="JSONL agent trace, see docs/trace_schema.md")
    ap.add_argument("--out", default="results/workload_fit")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sessions", type=int, default=400,
                    help="self-test only: sessions per synthetic case")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args.sessions)
    if not args.trace:
        ap.error("either --trace or --self-test is required")

    rows = load_trace(args.trace)
    result = fit(rows)

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trace": os.path.abspath(args.trace),
        "fit": result["workload"],
        "provenance": result["provenance"],
    }
    path = os.path.join(args.out, "workload_fit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    prov = result["provenance"]
    print(f"trace: {prov['n_calls']} calls over {prov['n_sessions']} sessions")
    print(f"system prompt: {prov['system_prompt_source']}")
    print()
    defaults = WorkloadConfig()
    print(f"{'parameter':<30} {'INVENTED default':>18} {'fitted':>14}")
    print("-" * 66)
    for key, value in result["workload"].items():
        if key == "n_sessions":
            continue
        old = getattr(defaults, key)
        fmt = f"{value:>14.4g}" if isinstance(value, float) else f"{value:>14}"
        oldfmt = f"{old:>18.4g}" if isinstance(old, float) else f"{old:>18}"
        print(f"{key:<30} {oldfmt} {fmt}")

    if prov["warnings"]:
        print("\nWARNINGS")
        for w in prov["warnings"]:
            print(f"  - {w}")

    print(f"\nwrote {path}")
    print("\nApply with:  python -m sim.run --set workload.pause_seconds_median=... "
          "(or edit configs/base.json)")
    print("Then rerun EXP01-EXP03: every headroom figure is a function of these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
