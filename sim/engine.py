"""Discrete-event model of a continuous-batching LLM server under agent workloads.

Scheduler shape (vLLM-flavoured):
  * one step processes a mixed batch under a token budget: every running decode consumes
    one token of budget, the remainder goes to prefill chunks, FCFS;
  * a request is admitted only if its whole prompt fits in the KV pool;
  * running out of blocks mid-decode preempts the most recently admitted request, which
    is requeued and recomputed from scratch.

Timing: see EngineConfig. Decode scales with context length because reading the KV cache
is what dominates a long-context decode step.

The timing constants are placeholders until fitted to measured Qwen2.5-3B numbers on the
target GPU. They set the *scale* of every latency and cost number this file produces;
they do not decide the *ranking* of retention policies, which is what the falsification
experiment asks about.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

from .cache import BlockPool, block_keys
from .config import Config
from .workload import Session


WAITING, PREFILL, DECODE, DONE = 0, 1, 2, 3


@dataclass
class Request:
    session_id: int
    turn_index: int
    prompt_tokens: int
    output_tokens: int
    tool_id: int
    pause_after_s: float
    is_last_turn: bool

    arrival_time: float = 0.0
    state: int = WAITING
    keys: list = field(default_factory=list)
    ctx_tokens: int = 0
    prefilled_tokens: int = 0
    generated_tokens: int = 0

    cached_tokens: int = 0
    prefill_tokens: int = 0  # prompt tokens actually recomputed
    first_schedule_time: float = -1.0
    ttft: float = -1.0
    finish_time: float = -1.0
    n_preemptions: int = 0
    admit_seq: int = 0


@dataclass
class RunResult:
    records: list[dict]
    summary: dict
    config: dict


RankFn = Callable[["Request", float], object]

# Priority-family ranks are (tier, key), lowest evicted first.
TIER_DEAD = 0.0  # known to never be used again
TIER_LIVE = 1.0


def _make_rank_fn(cfg: Config, pred_pause: dict[tuple[int, int], float] | None) -> RankFn:
    """Return the eviction rank a finishing request stamps onto its blocks.

    IMPORTANT: only arms whose name says "oracle", "belady" or "predict" may look at
    `req.is_last_turn` or `req.pause_after_s`. Knowing that a turn is a session's last
    is *itself* a prediction problem -- arguably the harder one -- so letting a
    deployable baseline peek at it would quietly hand it oracle powers and make the
    measured headroom too small.
    """
    kind = cfg.policy.kind
    scale = cfg.policy.oracle_ttl_scale

    if kind == "lru":
        return lambda req, now: 0.0
    if kind == "const_ttl":
        return lambda req, now: cfg.policy.const_ttl_s
    if kind == "lru_priority":
        # Validation arm: the priority mechanism carrying no information at all must
        # reproduce `lru` exactly. If it does not, the mechanism itself is biased.
        return lambda req, now: (TIER_LIVE, now)
    if kind == "ttl_oracle":
        # Perfect information spent through the TTL mechanism: protects the sessions
        # that come back *latest* for the longest. Kept as an ablation.
        return lambda req, now: 0.0 if req.is_last_turn else req.pause_after_s * scale
    if kind == "oracle_terminal":
        # Knows only whether the session has ended; falls back to LRU otherwise.
        # Isolates the value of termination prediction from pause-length prediction.
        return lambda req, now: (TIER_DEAD, 0.0) if req.is_last_turn else (TIER_LIVE, now)
    if kind == "belady":
        def belady(req: Request, now: float):
            if req.is_last_turn:
                return (TIER_DEAD, 0.0)
            return (TIER_LIVE, -(now + req.pause_after_s * scale))
        return belady
    if kind == "predict":
        if pred_pause is None:
            raise ValueError("policy.kind='predict' requires pred_pause")
        table = pred_pause

        def predict(req: Request, now: float):
            p = table.get((req.session_id, req.turn_index), math.inf)
            if not math.isfinite(p):  # predicted to be the session's last turn
                return (TIER_DEAD, 0.0)
            return (TIER_LIVE, -(now + p))
        return predict
    if kind == "belady_pause":
        # Belady's information, but ranked by the pause alone instead of by the absolute
        # next-use time. Kept as the oracle counterpart of `predict_guarded` so that the
        # cost of dropping the `now` term is measured separately from prediction error.
        def belady_pause(req: Request, now: float):
            if req.is_last_turn:
                return (TIER_DEAD, 0.0)
            return (TIER_LIVE, -req.pause_after_s * scale)
        return belady_pause
    if kind == "predict_guarded":
        # The deployable policy with a safe failure mode.
        #
        # `predict` ranks by -(now + p) and therefore inverts into MRU when p carries no
        # information: with p constant the rank orders by `now`, and evicting the largest
        # `now` evicts the most recently released block. Ranking by -p alone removes the
        # `now` term, so an uninformative predictor produces ties, and ties break by heap
        # insertion order -- which is release order, i.e. LRU. The policy degrades to the
        # incumbent instead of to its inverse.
        if pred_pause is None:
            raise ValueError("policy.kind='predict_guarded' requires pred_pause")
        table = pred_pause

        def predict_guarded(req: Request, now: float):
            p = table.get((req.session_id, req.turn_index), math.inf)
            if not math.isfinite(p):
                return (TIER_DEAD, 0.0)
            return (TIER_LIVE, -p)
        return predict_guarded
    if kind == "predict_terminal":
        # Uses only the predicted terminal flag and falls back to LRU otherwise -- the
        # deployable counterpart of `oracle_terminal`. Comparing the two separates "how
        # much is this information worth" from "how well can it be predicted", which are
        # the two things a predictor result otherwise silently mixes together.
        if pred_pause is None:
            raise ValueError("policy.kind='predict_terminal' requires pred_pause")
        table = pred_pause

        def predict_terminal(req: Request, now: float):
            p = table.get((req.session_id, req.turn_index), math.inf)
            if not math.isfinite(p):
                return (TIER_DEAD, 0.0)
            return (TIER_LIVE, now)
        return predict_terminal
    raise ValueError(f"unknown policy kind: {kind}")


class Engine:
    def __init__(self, cfg: Config, sessions: list[Session],
                 pred_pause: dict[tuple[int, int], float] | None = None):
        self.cfg = cfg
        self.sessions = sessions
        self.rank_fn = _make_rank_fn(cfg, pred_pause)

        ec = cfg.engine
        self.bs = ec.block_size
        self.shared_blocks = cfg.workload.system_prompt_tokens // self.bs
        self.pool = BlockPool(ec.kv_pool_blocks, self.bs, ec.enable_prefix_caching,
                              family=cfg.policy.family)

        self.now = 0.0
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.pending_arrivals: list[tuple[float, int, Request]] = []  # heap
        self.records: list[dict] = []

        self._arrival_seq = 0
        self._admit_seq = 0
        self.gpu_busy_s = 0.0
        self.n_steps = 0

        # Time-weighted count of sessions that have started and not yet finished. This
        # is what sets memory pressure, and unlike `concurrency` it is also defined for
        # open-loop arrivals, where the engine's own speed decides how many pile up.
        self._live_sessions: set[int] = set()
        self._live_session_seconds = 0.0

        self._init_sessions()

    # ------------------------------------------------------------- session driver

    def _request_for(self, session: Session, turn_index: int, arrival: float) -> Request:
        turn = session.turns[turn_index]
        return Request(
            session_id=session.session_id,
            turn_index=turn_index,
            prompt_tokens=turn.prompt_tokens,
            output_tokens=turn.output_tokens,
            tool_id=turn.tool_id,
            pause_after_s=turn.pause_after_s,
            is_last_turn=turn_index == session.n_turns - 1,
            arrival_time=arrival,
        )

    def _push_arrival(self, req: Request) -> None:
        import heapq

        self._arrival_seq += 1
        heapq.heappush(self.pending_arrivals, (req.arrival_time, self._arrival_seq, req))

    def _init_sessions(self) -> None:
        mode = self.cfg.arrival.mode
        self._session_queue = list(self.sessions)
        if mode == "closed":
            n_start = min(self.cfg.arrival.concurrency, len(self._session_queue))
            for _ in range(n_start):
                session = self._session_queue.pop(0)
                self._push_arrival(self._request_for(session, 0, 0.0))
        elif mode == "poisson":
            rng = random.Random(self.cfg.seed ^ 0x5EED)
            t = 0.0
            while self._session_queue:
                session = self._session_queue.pop(0)
                self._push_arrival(self._request_for(session, 0, t))
                t += rng.expovariate(self.cfg.arrival.rate_per_s)
        else:
            raise ValueError(f"unknown arrival mode: {mode}")

    def _on_session_turn_done(self, req: Request) -> None:
        session = self.sessions[req.session_id]
        if req.is_last_turn:
            self._live_sessions.discard(req.session_id)
        if not req.is_last_turn:
            nxt = self._request_for(session, req.turn_index + 1,
                                    self.now + req.pause_after_s)
            self._push_arrival(nxt)
            return
        if self.cfg.arrival.mode == "closed" and self._session_queue:
            new_session = self._session_queue.pop(0)
            self._push_arrival(self._request_for(new_session, 0, self.now))

    # ------------------------------------------------------------------ admission

    def _try_admit(self, req: Request) -> bool:
        n_total = math.ceil(req.prompt_tokens / self.bs)
        n_full = req.prompt_tokens // self.bs
        keys = block_keys(req.session_id, n_total, self.shared_blocks)

        hit = self.pool.lookup_prefix(keys[:n_full])
        new_keys = keys[hit:]
        if not self.pool.acquire(keys[:hit], new_keys, self.now):
            return False

        for i in range(hit, n_full):
            self.pool.mark_full(keys[i])

        req.keys = keys
        req.ctx_tokens = req.prompt_tokens
        req.cached_tokens = hit * self.bs
        req.prefill_tokens = req.prompt_tokens - req.cached_tokens
        req.prefilled_tokens = req.cached_tokens
        req.state = PREFILL
        self._admit_seq += 1
        req.admit_seq = self._admit_seq
        if req.first_schedule_time < 0:
            req.first_schedule_time = self.now
        self.running.append(req)
        return True

    def _grow_one_block(self, req: Request) -> bool:
        """Decode crossed a block boundary: seal the tail, allocate the next one."""
        tail_key = req.keys[-1]
        self.pool.mark_full(tail_key)
        self.pool.mark_ready([tail_key])
        new_index = len(req.keys)
        new_key = block_keys(req.session_id, new_index + 1, self.shared_blocks)[-1]
        if not self.pool.acquire([], [new_key], self.now):
            return False
        req.keys.append(new_key)
        return True

    def _emit_token(self, req: Request) -> None:
        """Produce one decoded token, growing the block chain first if it is needed.

        Space is secured *before* the token exists. Incrementing first and failing to
        allocate afterwards would leave a sequence holding fewer blocks than its context
        needs, which silently corrupts every subsequent step.
        """
        if req.ctx_tokens + 1 > len(req.keys) * self.bs:
            if not self._grow_one_block(req):
                # No room. Free some and let this sequence retry on the next step;
                # if it preempted itself, its state guard will skip it.
                self._preempt_one()
                return
        req.generated_tokens += 1
        req.ctx_tokens += 1
        if req.generated_tokens >= req.output_tokens:
            self._finish(req)

    def _preempt_one(self) -> bool:
        """Evict the most recently admitted request and recompute it later."""
        candidates = [r for r in self.running if r.keys]
        if not candidates:
            return False
        victim = max(candidates, key=lambda r: r.admit_seq)
        # Recompute-preemption frees the blocks outright rather than caching them at
        # some "evict me first" rank. Caching them would make preemption behave
        # differently under each policy -- under `lru` a dropped block would sit in
        # normal LRU order, under any non-zero TTL it would jump the queue -- which is
        # an arm-dependent asymmetry that has nothing to do with the policy under test.
        self.pool.release(victim.keys, self.now, self.pool.drop_rank(), cacheable=False)
        victim.keys = []
        victim.state = WAITING
        victim.prefilled_tokens = 0
        victim.generated_tokens = 0
        victim.ctx_tokens = 0
        victim.cached_tokens = 0
        victim.prefill_tokens = 0
        victim.n_preemptions += 1
        self.running.remove(victim)
        self.waiting.insert(0, victim)
        return True

    def _finish(self, req: Request) -> None:
        if req.ctx_tokens % self.bs == 0 and req.keys:
            self.pool.mark_full(req.keys[-1])
            self.pool.mark_ready([req.keys[-1]])
        rank = self.rank_fn(req, self.now)
        self.pool.release(req.keys, self.now, rank)
        req.state = DONE
        req.finish_time = self.now
        self.records.append({
            "session_id": req.session_id,
            "turn_index": req.turn_index,
            "tool_id": req.tool_id,
            "arrival_time": req.arrival_time,
            "first_schedule_time": req.first_schedule_time,
            "ttft": req.ttft,
            "finish_time": req.finish_time,
            "e2e_s": req.finish_time - req.arrival_time,
            "queue_s": req.first_schedule_time - req.arrival_time,
            "prompt_tokens": req.prompt_tokens,
            "cached_tokens": req.cached_tokens,
            "prefill_tokens": req.prefill_tokens,
            "output_tokens": req.output_tokens,
            "pause_after_s": req.pause_after_s,
            "is_last_turn": req.is_last_turn,
            "n_preemptions": req.n_preemptions,
            "evict_rank": repr(rank),
            "pool_util": self.pool.utilization(),
        })
        self._on_session_turn_done(req)

    # ----------------------------------------------------------------- step loop

    def _drain_arrivals(self) -> None:
        import heapq

        while self.pending_arrivals and self.pending_arrivals[0][0] <= self.now + 1e-12:
            _, _, req = heapq.heappop(self.pending_arrivals)
            if req.turn_index == 0:
                self._live_sessions.add(req.session_id)
            self.waiting.append(req)

    def _advance(self, dt: float, busy: bool) -> None:
        """Move the clock, accumulating the time-weighted live-session count."""
        if dt <= 0:
            return
        self._live_session_seconds += len(self._live_sessions) * dt
        self.now += dt
        if busy:
            self.gpu_busy_s += dt

    def run(self) -> RunResult:
        ec = self.cfg.engine
        stall_guard = 0
        while True:
            self._drain_arrivals()
            if not self.waiting and not self.running:
                if not self.pending_arrivals:
                    break
                self._advance(self.pending_arrivals[0][0] - self.now, busy=False)
                continue

            decoding = [r for r in self.running if r.state == DECODE]
            prefilling = [r for r in self.running if r.state == PREFILL]

            budget = ec.max_num_batched_tokens - len(decoding)

            # Admit new work into any leftover budget and pool space.
            while (self.waiting and budget > 0
                   and len(self.running) < ec.max_num_seqs):
                req = self.waiting[0]
                if not self._try_admit(req):
                    break
                self.waiting.pop(0)
                prefilling.append(req)
                budget -= 1  # at least one token of prefill will be spent

            if not decoding and not prefilling:
                # Nothing fits. Preempt if anything is resident, else the pool is too
                # small for a single request -- a config error worth failing loudly on.
                if self.running and self._preempt_one():
                    continue
                stall_guard += 1
                if stall_guard > 10:
                    raise RuntimeError(
                        "scheduler stalled: KV pool too small to admit any request "
                        f"(pool={ec.kv_pool_blocks} blocks, "
                        f"largest waiting prompt={max(r.prompt_tokens for r in self.waiting)} tokens)"
                    )
                continue
            stall_guard = 0

            # ---- allocate the step's token budget
            budget = ec.max_num_batched_tokens - len(decoding)
            prefill_tokens_this_step = 0
            prefill_cost = 0.0
            completed_prefill: list[Request] = []
            for req in prefilling:
                if budget <= 0:
                    break
                remaining = req.prompt_tokens - req.prefilled_tokens
                chunk = min(remaining, budget)
                start = req.prefilled_tokens
                end = start + chunk
                # Charged per position, not per token: the token at position p costs
                # b + 2cp, so the chunk [start, end) costs b*(end-start) + c*(end^2 -
                # start^2). A suffix recomputed after a partial cache hit starts at a
                # high position and is correctly more expensive per token.
                prefill_cost += (ec.prefill_s_per_token * chunk
                                 + ec.prefill_s_per_token2 * (end * end - start * start))
                req.prefilled_tokens = end
                prefill_tokens_this_step += chunk
                budget -= chunk
                if req.prefilled_tokens >= req.prompt_tokens:
                    completed_prefill.append(req)

            decode_cost = sum(ec.decode_s_per_seq + r.ctx_tokens * ec.decode_s_per_kv_token
                              for r in decoding)
            step_time = ec.step_overhead_s + prefill_cost + decode_cost
            self._advance(step_time, busy=True)
            self.n_steps += 1

            # ---- prefill completions turn into decodes and emit their first token
            for req in completed_prefill:
                if req.state != PREFILL:
                    continue  # preempted earlier in this same step
                self.pool.mark_ready(req.keys)
                req.state = DECODE
                req.ttft = self.now - req.arrival_time
                self._emit_token(req)

            # ---- decode advances one token per running sequence
            for req in decoding:
                if req.state != DECODE:
                    continue  # preempted earlier in this same step
                self._emit_token(req)

            self.running = [r for r in self.running if r.state in (PREFILL, DECODE)]

        return RunResult(records=self.records, summary=self._summarize(), config=self.cfg.to_dict())

    # ------------------------------------------------------------------ reporting

    def _summarize(self) -> dict:
        import statistics as st

        recs = self.records
        n = len(recs)
        makespan = max(r["finish_time"] for r in recs) if recs else 0.0
        total_prompt = sum(r["prompt_tokens"] for r in recs)
        total_cached = sum(r["cached_tokens"] for r in recs)
        total_prefill = sum(r["prefill_tokens"] for r in recs)

        def pct(values, q):
            if not values:
                return 0.0
            s = sorted(values)
            return s[min(len(s) - 1, int(q * (len(s) - 1)))]

        ttfts = [r["ttft"] for r in recs]
        e2es = [r["e2e_s"] for r in recs]
        queues = [r["queue_s"] for r in recs]

        hours = makespan / 3600.0
        rm_total = hours * self.cfg.cost.rm_per_hour

        from .workload import mean_context_blocks, offered_pressure

        ctx_blocks = mean_context_blocks(self.sessions, self.bs)
        mean_live = self._live_session_seconds / makespan if makespan else 0.0
        pool = self.cfg.engine.kv_pool_blocks
        return {
            "n_calls": n,
            "makespan_s": makespan,
            # The variable the phenomenon actually depends on. `offered` is the
            # closed-loop design value; `measured` is time-weighted and also valid in
            # open loop, where the engine's own speed decides how many sessions pile up.
            "mean_live_sessions": mean_live,
            "mean_context_blocks": ctx_blocks,
            "pressure_measured": mean_live * ctx_blocks / pool if pool else float("inf"),
            "pressure_offered": (
                offered_pressure(self.sessions, self.bs, pool, self.cfg.arrival.concurrency)
                if self.cfg.arrival.mode == "closed" else None),
            "gpu_busy_s": self.gpu_busy_s,
            "gpu_busy_frac": self.gpu_busy_s / makespan if makespan else 0.0,
            "n_steps": self.n_steps,
            "token_hit_rate": total_cached / total_prompt if total_prompt else 0.0,
            "prefill_tokens_computed": total_prefill,
            "prompt_tokens_total": total_prompt,
            "ttft_p50": st.median(ttfts) if ttfts else 0.0,
            "ttft_p95": pct(ttfts, 0.95),
            "ttft_mean": st.fmean(ttfts) if ttfts else 0.0,
            "e2e_p50": st.median(e2es) if e2es else 0.0,
            "e2e_p95": pct(e2es, 0.95),
            "queue_p50": st.median(queues) if queues else 0.0,
            "queue_p95": pct(queues, 0.95),
            "n_preemptions": sum(r["n_preemptions"] for r in recs),
            "n_evictions": self.pool.n_evictions,
            "n_protected_evictions": self.pool.n_protected_evictions,
            "pool_blocks": self.cfg.engine.kv_pool_blocks,
            "rm_per_hour": self.cfg.cost.rm_per_hour,
            # Two billing models, because they disagree and the disagreement matters.
            #
            # rm_*_reserved  charges wall clock. This is how the Sunway HPC session is
            #                billed, and it is the honest number for a dedicated box --
            #                but under open-loop arrivals the makespan is pinned by the
            #                arrival schedule, so no retention policy can move it and
            #                this metric is near-blind by construction.
            # rm_*_gputime   charges only the seconds the GPU was actually working. This
            #                is the shared / autoscaled model, and it is the one that
            #                responds to a policy under open-loop arrivals.
            #
            # Quoting one without saying which is how a cache result gets oversold.
            "rm_total": rm_total,
            "rm_per_1k_calls": (rm_total / n * 1000.0) if n else 0.0,
            "rm_gputime_total": self.gpu_busy_s / 3600.0 * self.cfg.cost.rm_per_hour,
            "rm_gputime_per_1k_calls": (
                (self.gpu_busy_s / 3600.0 * self.cfg.cost.rm_per_hour) / n * 1000.0
                if n else 0.0),
        }


def run_config(cfg: Config, sessions: list[Session],
               pred_pause: dict[tuple[int, int], float] | None = None) -> RunResult:
    return Engine(cfg, sessions, pred_pause).run()
