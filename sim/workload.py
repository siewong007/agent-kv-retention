"""Synthetic agent trace generator.

A trace is generated *independently of any serving policy*: token counts, tool choices
and pause durations are properties of the agent and its tools, not of the engine that
serves it. The engine only decides *when* each turn gets served. This is what makes the
policy comparison paired -- every arm sees byte-identical work.

Predictability is an explicit dial, not an accident:
  * tool_pause_spread    -- how much a tool's identity determines its runtime.
                            0.0 makes pauses unpredictable from tool identity, so any
                            predictor gain must come from elsewhere.
  * tool_markov_self_prob -- tool-use inertia (cf. AutoTool). Higher means the previous
                            tool predicts the next one.
Report results as a function of these, never at a single point: the whole predictor
story lives or dies on how much structure the generator was given.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .config import WorkloadConfig


@dataclass
class Turn:
    """One LLM call."""

    index: int
    prompt_tokens: int
    output_tokens: int
    tool_id: int
    # Wall-clock gap between this turn's decode finishing and the next turn arriving.
    # 0.0 on the last turn of a session.
    pause_after_s: float


@dataclass
class Session:
    session_id: int
    turns: list[Turn] = field(default_factory=list)

    @property
    def n_turns(self) -> int:
        return len(self.turns)


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    """Lognormal parameterised by its median (exp(mu)) rather than by mu."""
    if median <= 0:
        return 0.0
    import math

    return math.exp(math.log(median) + sigma * rng.gauss(0.0, 1.0))


def _tool_pause_multipliers(rng: random.Random, n_tools: int, spread: float) -> list[float]:
    """Per-tool speed. spread=0 -> every tool takes the same time.

    Normalised to a geometric mean of exactly 1, so that `pause_seconds_median` is the
    realised population median rather than a base that the drawn multipliers then shift.
    Without this, a sample of 8 multipliers at spread 0.9 moves the effective median by
    up to ~20% depending on the seed, which means (a) the config parameter does not mean
    what it says, (b) seed variance carries a nuisance component that has nothing to do
    with the workload, and (c) fitting a trace and feeding the result back in does not
    round-trip. bench/fit_workload.py --self-test is what caught this.
    """
    import math

    raw = [spread * rng.gauss(0.0, 1.0) for _ in range(n_tools)]
    mean_log = sum(raw) / len(raw) if raw else 0.0
    return [math.exp(x - mean_log) for x in raw]


def generate_sessions(cfg: WorkloadConfig, seed: int) -> list[Session]:
    """Deterministic given (cfg, seed). Never depends on the policy.

    Every knob lives in `cfg` on purpose: a trace that depends on a function default
    is a trace whose result file does not describe it.
    """
    rng = random.Random(seed)
    multipliers = _tool_pause_multipliers(rng, cfg.n_tools, cfg.tool_pause_spread)

    sessions: list[Session] = []
    for sid in range(cfg.n_sessions):
        n_turns = rng.randint(cfg.turns_min, cfg.turns_max)
        task_tokens = int(_lognormal(rng, cfg.task_tokens_median, cfg.task_tokens_sigma))
        context = cfg.system_prompt_tokens + task_tokens

        session = Session(session_id=sid)
        tool_id = rng.randrange(cfg.n_tools)
        for t in range(n_turns):
            if t > 0:
                # Tool-use inertia: repeat the previous tool with probability self_prob.
                if rng.random() >= cfg.tool_markov_self_prob:
                    tool_id = rng.randrange(cfg.n_tools)

            # Termination signalling. Everything in this block is skipped entirely at
            # strength 0, including its RNG draws, so the default generator is
            # byte-identical to the version that produced the v2 results.
            closeness = 0.0
            if cfg.termination_signal_strength > 0:
                # 1.0 on the final turn, 0.5 one before it, 0.33 two before, ...
                closeness = 1.0 / (1.0 + (n_turns - 1 - t))
                if rng.random() < cfg.termination_signal_strength * closeness:
                    tool_id = cfg.finishing_tool_id

            out_tokens = max(1, int(_lognormal(rng, cfg.output_tokens_median, cfg.output_tokens_sigma)))
            if cfg.termination_signal_strength > 0:
                # Outputs shorten as the session converges. Indirect and noisy: the
                # lognormal spread is wide enough that a single short output is weak
                # evidence, and only the trend is informative.
                out_tokens = max(
                    1,
                    int(out_tokens * math.exp(-cfg.termination_signal_strength * closeness)),
                )

            is_last = t == n_turns - 1
            if is_last:
                pause = 0.0
            else:
                pause = _lognormal(
                    rng,
                    cfg.pause_seconds_median * multipliers[tool_id],
                    cfg.pause_seconds_sigma,
                )

            session.turns.append(
                Turn(
                    index=t,
                    prompt_tokens=context,
                    output_tokens=out_tokens,
                    tool_id=tool_id,
                    pause_after_s=pause,
                )
            )

            tool_result = int(_lognormal(rng, cfg.tool_result_tokens_median, cfg.tool_result_tokens_sigma))
            context += out_tokens + tool_result

        sessions.append(session)

    return sessions


def mean_context_blocks(sessions: list[Session], block_size: int) -> float:
    """Mean KV footprint of one live session, in blocks.

    Measured at the end of a turn -- prompt plus the tokens just generated -- because
    that is the state that has to survive the pause. Averaged over all turns, since a
    session observed at a random instant is at a random point in its own growth.
    """
    import math

    blocks = [math.ceil((t.prompt_tokens + t.output_tokens) / block_size)
              for s in sessions for t in s.turns]
    return sum(blocks) / len(blocks) if blocks else 0.0


def offered_pressure(sessions: list[Session], block_size: int, pool_blocks: int,
                     concurrency: int) -> float:
    """Working set divided by KV pool capacity, for closed-loop arrivals.

    This is the variable the phenomenon actually depends on. Concurrency and pool size
    are two handles on it, and a result reported against either one alone does not
    transfer to a different GPU. Only defined for closed-loop arrivals, where the number
    of live sessions is fixed by construction; in open loop, use the measured value the
    engine reports instead.
    """
    if not pool_blocks:
        return float("inf")
    return concurrency * mean_context_blocks(sessions, block_size) / pool_blocks


def workload_summary(sessions: list[Session]) -> dict:
    """Descriptive stats, for sanity-checking that the generator makes plausible agents."""
    import statistics as st

    n_calls = sum(s.n_turns for s in sessions)
    prompts = [t.prompt_tokens for s in sessions for t in s.turns]
    pauses = [t.pause_after_s for s in sessions for t in s.turns if t.index < s.n_turns - 1]
    finals = [s.turns[-1].prompt_tokens for s in sessions]
    return {
        "n_sessions": len(sessions),
        "n_calls": n_calls,
        "turns_per_session_mean": n_calls / len(sessions),
        "prompt_tokens_median": st.median(prompts),
        "prompt_tokens_p95": sorted(prompts)[int(0.95 * (len(prompts) - 1))],
        "final_context_tokens_median": st.median(finals),
        "final_context_tokens_p95": sorted(finals)[int(0.95 * (len(finals) - 1))],
        "pause_s_median": st.median(pauses) if pauses else 0.0,
        "pause_s_p95": sorted(pauses)[int(0.95 * (len(pauses) - 1))] if pauses else 0.0,
        "total_prompt_tokens": sum(prompts),
    }
