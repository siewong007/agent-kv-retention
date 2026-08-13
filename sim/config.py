"""Experiment configuration.

Every knob of the simulator lives here. A config is a plain nested dict so that it
round-trips through JSON without loss and can be embedded verbatim into result files.
Reproducibility rule: a result file is only valid if it carries the exact config that
produced it, plus the seed.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class WorkloadConfig:
    """Synthetic agent trace generator.

    All token counts are in tokens. All durations are in seconds.

    NOTE ON PROVENANCE: these defaults are *placeholders shaped like* a coding agent,
    not measurements. Before any number from this simulator goes into the thesis, the
    pause distribution and the tool-result size distribution must be calibrated against
    a real agent trace. See docs/calibration.md.
    """

    n_sessions: int = 200

    # Turns per session (one turn == one LLM call).
    turns_min: int = 12
    turns_max: int = 30

    # Shared system prompt / tool schema, identical across every session.
    # This is the only cross-session prefix sharing in the model.
    system_prompt_tokens: int = 1200

    # First user turn (the task statement).
    task_tokens_median: float = 400.0
    task_tokens_sigma: float = 0.5

    # Model output per turn (reasoning + tool call).
    output_tokens_median: float = 160.0
    output_tokens_sigma: float = 0.6

    # Tool result appended to the context after each turn. Heavy tailed on purpose:
    # a `cat` of a source file dwarfs a `ls`.
    tool_result_tokens_median: float = 700.0
    tool_result_tokens_sigma: float = 1.1

    # Pause between the end of a turn's decode and the arrival of the next turn,
    # i.e. how long the tool takes to run outside the serving engine.
    pause_seconds_median: float = 2.0
    pause_seconds_sigma: float = 1.0

    # Number of distinct tool identities.
    n_tools: int = 8

    # How much structure the generated world contains -- i.e. the ceiling on what any
    # predictor could possibly learn. These are the two knobs that decide whether the
    # week-2 predictor result means anything, so they live in the config and get written
    # into every result file. Never report a predictor number at a single setting of
    # these; report a curve.
    #   tool_pause_spread:     log-spread of per-tool runtime. 0.0 makes tool identity
    #                          carry no information about pause length at all.
    #   tool_markov_self_prob: probability the next turn reuses the same tool
    #                          (tool-use inertia, cf. AutoTool).
    tool_pause_spread: float = 0.9
    tool_markov_self_prob: float = 0.45

    # How strongly a session signals that it is about to end.
    #
    # With this at 0 -- the default, which reproduces the generator byte-for-byte --
    # `n_turns` is drawn independently of everything else, so P(this turn is the last)
    # is a pure function of turn_index and NOTHING can predict termination better than
    # the hazard curve. EXP04 established that the hard way: a gradient-boosted
    # classifier reached the Bayes limit at recall 0.10 and the experiment turned out to
    # be measuring `rng.randint(12, 30)` rather than any method.
    #
    # Above 0, two indirect and noisy behaviours appear near the end of a session,
    # both of which real coding agents show as they converge:
    #   * outputs shorten -- the agent stops emitting long tool calls and starts
    #     summarising;
    #   * the tool mix shifts toward a designated "finishing" tool.
    # Neither reveals the remaining turn count directly. A predictor has to infer it,
    # which is the point: a generator that hands over the answer would overstate what a
    # predictor can do just as badly as one that hides it entirely understates it.
    #
    # Report predictor results as a curve over this dial, never at a single setting.
    termination_signal_strength: float = 0.0
    # Which tool identity plays the "finishing" role when the signal is on.
    finishing_tool_id: int = 0


@dataclass
class EngineConfig:
    """Serving engine model.

    A deliberately small model of a continuous-batching engine (vLLM-shaped):
    a step loop, a token budget per step, a block-granular KV pool with a prefix cache.

    NOTE ON PROVENANCE: the timing constants below are MEASURED against vLLM 0.26 +
    Qwen2.5-3B on the RTX 5080 (2026-08-01). Reproduce with bench/serve_calib.sh then
    bench/fit_timing.py and bench/fit_batch.py; the ledger is docs/calibration.md.
    Re-measure before quoting any latency or cost number on different hardware.
    """

    block_size: int = 16
    # MEASURED, 2026-08-01: vLLM 0.26 reported "GPU KV cache size: 205,888 tokens"
    # = 12868 blocks, on the RTX 5080 with Qwen2.5-3B at gpu_memory_utilization 0.85.
    # The old derived value was 16000, i.e. 24% too generous -- the derivation did not
    # account for the ~1.3 GiB the Windows desktop holds, nor for vLLM's own activation
    # and CUDA-graph reservations. See results/calib/ and docs/calibration.md.
    kv_pool_blocks: int = 12868

    max_num_seqs: int = 64  # max concurrently running sequences
    max_num_batched_tokens: int = 8192  # per-step token budget (chunked prefill)

    # Timing model:
    #   step_time = step_overhead                                  (read the weights)
    #             + sum over prefilling seqs of chunk cost         (see below)
    #             + sum over decoding seqs of
    #               (decode_s_per_seq + ctx_tokens * decode_s_per_kv_token)
    #
    # Prefill is charged per POSITION, not per token. Measured total prefill time is
    # quadratic in prompt length -- t = a + b*n + c*n^2, R^2 1.0000 -- because attention
    # is quadratic. Differentiating, the token at position p costs b + 2*c*p, so a chunk
    # covering positions [s, e) costs
    #
    #   prefill_s_per_token * (e - s)  +  prefill_s_per_token2 * (e^2 - s^2)
    #
    # This matters for exactly this project's subject: after a partial prefix-cache hit,
    # the recomputed suffix starts at a high position and each of its tokens must attend
    # over the whole cached prefix. A linear model charges those tokens the same as
    # tokens at position zero and therefore understates the cost of a partial hit --
    # which is the central quantity here.
    #
    # Decode likewise scales with context length: reading the KV cache is what a
    # long-context decode step actually spends its time on.
    #
    # MEASURED 2026-08-01 against vLLM 0.26 + Qwen2.5-3B on the RTX 5080, prefix caching
    # off. Raw samples and fits in results/calib/timing_fit.json. Every one of the four
    # derived guesses was wrong by 20-58%, which is why they were measured.
    #
    #   parameter                derived   measured    note
    #   prefill linear term      8.3e-5    6.669e-5    marginal rate at position 0
    #   prefill quadratic term   0 (!)     1.954e-9    absent from the derived model
    #   decode_s_per_kv_token    3.8e-8    5.994e-8    +58%   (R^2 0.972)
    #   step_overhead_s          6.4 ms    9.554 ms    +49%   (batch sweep, R^2 0.996)
    #   decode_s_per_seq         0.1 ms    0.024 ms    -76%   (batch sweep)
    #   kv_pool_blocks           16000     12868       -20%   (read from vLLM startup)
    #
    # At position 0 the marginal prefill rate is 1/6.669e-5 = 15.0k tok/s; by position
    # 16k it has fallen to 7.6k tok/s. A linear model cannot express that at all.
    #
    # step_overhead_s and decode_s_per_seq were collinear at batch size 1 and are
    # separated by bench/fit_batch.py, which sweeps batch size: the intercept is the
    # cost that does not scale with the batch, the slope is per-sequence. Independent
    # cross-check: the two sum to 9.61 ms at B=1, against 9.06 ms measured end-to-end by
    # bench/fit_timing.py at a similar context. See results/calib/batch_fit.json.
    step_overhead_s: float = 0.009554
    prefill_s_per_token: float = 6.669e-5
    prefill_s_per_token2: float = 1.954e-9
    decode_s_per_seq: float = 2.4e-5
    decode_s_per_kv_token: float = 5.994e-8

    enable_prefix_caching: bool = True


@dataclass
class PolicyConfig:
    """KV retention policy.

    Five arms across two mechanisms (see sim/cache.py for the mechanisms themselves).

    TTL mechanism -- a protection window from the moment a block is released:
      kind="lru"        TTL = 0 everywhere, so eviction is pure LRU. This is the
                        incumbent: vLLM does NOT drop prefix blocks at request end, it
                        drops them when something else needs the space. Any claim of
                        the form "current engines throw the cache away" has to beat
                        this arm, not a strawman.
      kind="const_ttl"  TTL = one tuned constant, applied to every session including
                        ones that have finished. Continuum-shaped. This is the arm that
                        can falsify the whole project: if a tuned constant captures most
                        of the oracle's gain, prediction is not worth building.
      kind="ttl_oracle" TTL = the session's true upcoming pause. This is the *naive*
                        way to spend perfect information, kept as an ablation: it
                        protects long-pause sessions longest, which is backwards under
                        scarcity. Included to separate "better information" from
                        "better mechanism".

    Priority mechanism -- evict whichever block is needed furthest in the future:
      kind="lru_priority"   Carries no information; must reproduce `lru` exactly.
                        A validation arm, not a result: it proves the two mechanisms
                        are comparable rather than differently biased.
      kind="oracle_terminal" Knows only whether a session has ended, LRU otherwise.
                        Splits the oracle's advantage into "knowing the session is
                        over" and "knowing when it comes back" -- the first is a far
                        easier prediction problem, so if it carries most of the gain
                        the practical recommendation changes completely.
      kind="belady"     Fed the true next-use time. NOT an upper bound, despite the
                        name: Belady's optimality needs a fixed offline reference
                        stream, and here eviction changes when work is recomputed, so
                        the stream responds to the policy. It is also myopic -- it sees
                        the next use, not the whole future. `belady_pause` beats it by
                        6-42%. Treat it as a strong oracle reference.
      kind="belady_pause" Same oracle information, ranked by pause length alone with an
                        LRU tie-break. The oracle counterpart of `predict_guarded`.
      kind="predict"    Fed a predicted next-use time. Same mechanism as belady, so the
                        belady-minus-predict gap is attributable to prediction error
                        and nothing else.
      kind="predict_terminal" Uses only the predicted terminal flag, LRU otherwise. The
                        deployable counterpart of `oracle_terminal`; the gap between the
                        two is classifier error, nothing else.

    Only the oracle arms may read `is_last_turn` or the true pause. See
    sim/engine.py:_make_rank_fn.
    """

    kind: str = "lru"
    const_ttl_s: float = 0.0

    # ttl_oracle / belady: scale the true pause before use. 1.0 = perfect information;
    # sweeping it away from 1.0 shows how fast the benefit decays with prediction error.
    oracle_ttl_scale: float = 1.0

    @property
    def family(self) -> str:
        return "ttl" if self.kind in ("lru", "const_ttl", "ttl_oracle") else "priority"

    @property
    def is_oracle(self) -> bool:
        return self.kind in ("ttl_oracle", "oracle_terminal", "belady")


@dataclass
class ArrivalConfig:
    """How sessions enter the system.

    mode="closed": a fixed number of session slots; a new session starts the instant one
                   finishes. Load is generated by the system's own speed, so makespan
                   is policy-dependent and RM cost differences are real.
    mode="poisson": sessions arrive at a fixed rate regardless of how the system copes.
                   Makespan is pinned by the arrival rate, so latency moves but cost
                   barely does. Use this one to test whether cache wins convert to money.
    """

    mode: str = "closed"
    concurrency: int = 32  # closed mode only
    rate_per_s: float = 0.5  # poisson mode only


@dataclass
class CostConfig:
    """Money. Rates are the Sunway HPC on-demand list prices (RM per wall-clock hour)."""

    rm_per_hour: float = 3.06  # T4 16GB
    label: str = "T4-16GB"


def kv_pool_blocks_from_memory(
    gpu_memory_gb: float,
    weights_gb: float,
    kv_bytes_per_token: float,
    block_size: int,
    activation_headroom_gb: float = 1.0,
) -> int:
    """Blocks that fit on a GPU. Keeps the pool size traceable to hardware, not folklore.

    Qwen2.5-3B, fp16: 36 layers x 2 KV heads x 128 head_dim x 2 (K and V) x 2 bytes
    = 36864 bytes per token. On a 16 GB card with ~6.2 GB of weights and 1 GB of
    headroom that is (16 - 6.2 - 1) * 2**30 / 36864 = ~256k tokens = ~16k blocks.
    """
    free_bytes = (gpu_memory_gb - weights_gb - activation_headroom_gb) * (2 ** 30)
    return max(1, int(free_bytes / (kv_bytes_per_token * block_size)))


@dataclass
class Config:
    seed: int = 0
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    arrival: ArrivalConfig = field(default_factory=ArrivalConfig)
    cost: CostConfig = field(default_factory=CostConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Config":
        d = copy.deepcopy(d)
        return Config(
            seed=d.get("seed", 0),
            workload=WorkloadConfig(**d.get("workload", {})),
            engine=EngineConfig(**d.get("engine", {})),
            policy=PolicyConfig(**d.get("policy", {})),
            arrival=ArrivalConfig(**d.get("arrival", {})),
            cost=CostConfig(**d.get("cost", {})),
        )

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return Config.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def replace(self, **overrides: Any) -> "Config":
        """Return a copy with dotted-path overrides applied, e.g. policy.kind='oracle'."""
        d = self.to_dict()
        for key, value in overrides.items():
            section, _, leaf = key.partition(".")
            if not leaf:
                d[section] = value
                continue
            if section not in d or not isinstance(d[section], dict):
                raise KeyError(f"unknown config section: {section}")
            if leaf not in d[section]:
                raise KeyError(f"unknown config key: {key}")
            d[section][leaf] = value
        return Config.from_dict(d)
