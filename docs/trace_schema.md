# Agent trace schema

The one input this project needs from the real world.

Every headroom figure in EXP01–EXP03 is a function of the pause distribution, and that
distribution is currently invented (see [calibration.md](calibration.md)). This file
defines the minimum a real agent run has to record for `bench/fit_workload.py` to turn
it into `WorkloadConfig` parameters.

**The trace is used only to fit distributions.** The experiments keep running on
synthetic traces, so they stay reproducible, seed-controlled and free of API keys. That
is what makes the synthetic generator defensible rather than a guess.

## Format

JSON Lines, one object per LLM call, in any order. Required fields:

| field | type | meaning |
|---|---|---|
| `session_id` | string or int | groups calls belonging to one agent task |
| `turn` | int | 0-based index within the session |
| `t_request` | float | unix seconds, when the request was sent to the engine |
| `t_response_end` | float | unix seconds, when the last output token arrived |
| `prompt_tokens` | int | prompt length as the engine counted it |
| `output_tokens` | int | generated length as the engine counted it |

Optional but valuable:

| field | type | why it matters |
|---|---|---|
| `tool` | string | needed for `tool_pause_spread` and `tool_markov_self_prob`, i.e. for any claim about how predictable the workload is |
| `tool_result_tokens` | int | otherwise derived as `prompt_tokens[t+1] - prompt_tokens[t] - output_tokens[t]` |
| `system_prompt_tokens` | int | otherwise estimated as the minimum turn-0 prompt across sessions, which is an **upper bound** -- that prompt still contains the smallest task statement (biased high by 4-9% on synthetic checks) |

Example:

```json
{"session_id": "swebench-astropy-12907", "turn": 0, "t_request": 1785549486.11, "t_response_end": 1785549489.42, "prompt_tokens": 1683, "output_tokens": 214, "tool": "read_file"}
{"session_id": "swebench-astropy-12907", "turn": 1, "t_request": 1785549490.87, "t_response_end": 1785549494.10, "prompt_tokens": 2731, "output_tokens": 190, "tool": "read_file"}
```

## What gets derived

| WorkloadConfig field | derived from |
|---|---|
| `pause_seconds_median` | `t_request[t+1] - t_response_end[t]`, lognormal |
| `pause_seconds_sigma` | the same pauses, but pooled **within tool**. Marginally the spread is sqrt(sigma^2 + tool_pause_spread^2), so fitting it across all tools double-counts the tool effect |
| `output_tokens_median`, `output_tokens_sigma` | `output_tokens`, lognormal |
| `tool_result_tokens_median`, `tool_result_tokens_sigma` | explicit field, or the prompt-growth identity above |
| `turns_min`, `turns_max` | per-session turn counts (5th/95th percentile, to keep one runaway session from setting the range) |
| `system_prompt_tokens` | explicit field, or min turn-0 prompt (upper bound) |
| `task_tokens_median`, `task_tokens_sigma` | turn-0 prompt minus the system prompt |
| `n_tools` | distinct `tool` values |
| `tool_pause_spread` | spread of log per-tool median pause |
| `tool_markov_self_prob` | fraction of consecutive turns reusing the same tool, inverted for the generator's uniform resample: `p = (P_same - 1/n) / (1 - 1/n)` |

## Two traps

**The pause must exclude engine time.** `t_request[t+1] - t_response_end[t]` is tool
execution plus agent-framework overhead — time the serving engine is not working on this
session. Using `t_request[t+1] - t_request[t]` instead would fold the engine's own
latency into the pause, and since the engine's latency is what the simulator predicts,
the fit would be circular.

**A trace collected under load is not a clean sample.** If the agent was competing for
the same GPU, its pauses are unaffected but its `t_response_end` is not, and sessions
that queued longer will look like they had shorter pauses. Collect with one agent at a
time, or record the engine-side queue time and subtract it.

## Where to get one

Any of these produce the required fields with light instrumentation, and none requires
changing the agent's behaviour:

- an OpenAI-compatible proxy in front of the agent, logging request/response timestamps
  and the `usage` block;
- vLLM's own request logs plus the agent's tool-call log, joined on request id;
- OpenHands / SWE-agent episode logs, which already carry per-step timestamps and tool
  names.

`bench/fit_workload.py --self-test` checks the fitter can recover known parameters from
the project's own generator before it is trusted on real data.
