"""Invariants the simulator must hold before any of its numbers mean anything.

Run: python -m tests.test_invariants
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.config import Config  # noqa: E402
from sim.engine import run_config  # noqa: E402
from sim.workload import generate_sessions  # noqa: E402


def small(**over) -> Config:
    base = Config().replace(**{
        "workload.n_sessions": 25,
        "arrival.concurrency": 12,
        "engine.kv_pool_blocks": 6000,
    })
    return base.replace(**over) if over else base


def run(cfg: Config):
    return run_config(cfg, generate_sessions(cfg.workload, cfg.seed))


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def test_deterministic():
    """Same config, same seed, byte-identical result. Non-negotiable."""
    a, b = run(small()), run(small())
    assert a.summary == b.summary, "identical configs produced different summaries"


@check
def test_seed_matters():
    a = run(small(seed=0)).summary
    b = run(small(seed=1)).summary
    assert a["prefill_tokens_computed"] != b["prefill_tokens_computed"], \
        "changing the seed changed nothing -- the workload is not actually random"


@check
def test_priority_mechanism_is_unbiased():
    """`lru_priority` carries no information, so it must reproduce `lru` exactly.

    If these two diverge, any belady-vs-lru gap is contaminated by a difference between
    the two eviction mechanisms rather than by the information they carry.
    """
    a = run(small(**{"policy.kind": "lru"})).summary
    b = run(small(**{"policy.kind": "lru_priority"})).summary
    for key in ("token_hit_rate", "prefill_tokens_computed", "makespan_s", "n_evictions"):
        assert math.isclose(a[key], b[key], rel_tol=1e-9), (
            f"mechanism bias on {key}: lru={a[key]} lru_priority={b[key]}")


@check
def test_token_accounting():
    """Every prompt token is either reused from cache or recomputed. No third option."""
    result = run(small())
    for rec in result.records:
        assert rec["cached_tokens"] + rec["prefill_tokens"] == rec["prompt_tokens"], rec
        assert 0 <= rec["cached_tokens"] <= rec["prompt_tokens"], rec


@check
def test_every_turn_runs_exactly_once():
    cfg = small()
    sessions = generate_sessions(cfg.workload, cfg.seed)
    result = run_config(cfg, sessions)
    expected = {(s.session_id, t.index) for s in sessions for t in s.turns}
    got = [(r["session_id"], r["turn_index"]) for r in result.records]
    assert len(got) == len(expected), f"ran {len(got)} turns, expected {len(expected)}"
    assert set(got) == expected, "turn set mismatch"


@check
def test_turns_are_ordered_and_respect_pauses():
    """Turn t+1 of a session cannot arrive before turn t finished plus its pause."""
    cfg = small()
    sessions = generate_sessions(cfg.workload, cfg.seed)
    result = run_config(cfg, sessions)
    by_session: dict[int, list[dict]] = {}
    for rec in result.records:
        by_session.setdefault(rec["session_id"], []).append(rec)
    for sid, recs in by_session.items():
        recs.sort(key=lambda r: r["turn_index"])
        for prev, nxt in zip(recs, recs[1:]):
            assert nxt["turn_index"] == prev["turn_index"] + 1
            expected_arrival = prev["finish_time"] + prev["pause_after_s"]
            assert nxt["arrival_time"] >= expected_arrival - 1e-6, (
                f"session {sid} turn {nxt['turn_index']} arrived at "
                f"{nxt['arrival_time']}, before {expected_arrival}")


@check
def test_pool_capacity_never_exceeded():
    cfg = small()
    result = run(cfg)
    assert max(r["pool_util"] for r in result.records) <= 1.0 + 1e-9


@check
def test_prefix_caching_off_means_no_hits():
    result = run(small(**{"engine.enable_prefix_caching": False}))
    assert result.summary["token_hit_rate"] == 0.0
    assert result.summary["prefill_tokens_computed"] == result.summary["prompt_tokens_total"]


@check
def test_huge_pool_reaches_the_ceiling():
    """With unlimited KV there is nothing to evict, so every policy must tie."""
    kw = {"engine.kv_pool_blocks": 400000}
    ref = run(small(**kw)).summary
    for kind in ("const_ttl", "ttl_oracle", "oracle_terminal", "belady"):
        got = run(small(**{**kw, "policy.kind": kind, "policy.const_ttl_s": 30.0})).summary
        assert math.isclose(ref["prefill_tokens_computed"], got["prefill_tokens_computed"],
                            rel_tol=1e-9), f"{kind} differs from lru with an unlimited pool"


@check
def test_workload_is_policy_independent():
    """Every arm must see identical work; otherwise the comparison is not paired."""
    cfg = small()
    sessions = generate_sessions(cfg.workload, cfg.seed)
    total = sum(t.prompt_tokens for s in sessions for t in s.turns)
    for kind in ("lru", "const_ttl", "ttl_oracle", "oracle_terminal", "belady"):
        c = cfg.replace(**{"policy.kind": kind, "policy.const_ttl_s": 8.0})
        assert run_config(c, sessions).summary["prompt_tokens_total"] == total, kind


@check
def test_termination_signal_off_by_default():
    """At strength 0 the generator must be byte-identical to the version without it.

    The v2 experiment results were produced before the termination signal existed. If
    the default path consumed even one extra RNG draw, every one of them would silently
    become unreproducible.
    """
    cfg = Config().replace(**{"workload.n_sessions": 40})
    assert cfg.workload.termination_signal_strength == 0.0, "the signal is on by default"
    a = generate_sessions(cfg.workload, 7)
    b = generate_sessions(cfg.workload.__class__(**{
        **cfg.workload.__dict__, "termination_signal_strength": 0.0}), 7)
    assert [(t.index, t.prompt_tokens, t.output_tokens, t.tool_id, t.pause_after_s)
            for s in a for t in s.turns] == \
           [(t.index, t.prompt_tokens, t.output_tokens, t.tool_id, t.pause_after_s)
            for s in b for t in s.turns]


@check
def test_termination_signal_is_learnable_but_not_trivial():
    """Turning the dial up must make the end of a session detectable -- and not for free.

    Two failure modes are guarded here. If the signal does nothing, EXP04 measures the
    generator's silence again. If it hands over the answer, EXP04 measures a giveaway.
    Both produce a number that looks like a predictor result and is not one.
    """
    import statistics as st

    base = Config().replace(**{"workload.n_sessions": 300})

    def separation(strength: float) -> float:
        cfg = base.replace(**{"workload.termination_signal_strength": strength})
        sessions = generate_sessions(cfg.workload, 3)
        last, other = [], []
        for s in sessions:
            for t in s.turns:
                (last if t.index == s.n_turns - 1 else other).append(t.output_tokens)
        return st.median(other) / st.median(last)

    off, on = separation(0.0), separation(2.0)
    assert 0.9 < off < 1.1, f"output length already separates terminal turns at strength 0 ({off:.2f})"
    assert on > 1.3, f"strength 2.0 barely shortens final outputs ({on:.2f}); nothing to learn"
    assert on < 6.0, f"strength 2.0 makes the last turn unmistakable ({on:.2f}); that is a giveaway"


@check
def test_prefill_is_charged_per_position():
    """A recomputed suffix must cost more per token than the same tokens at position 0.

    This is the whole point of the quadratic term. If it ever regresses to a linear
    charge, the simulator will understate the cost of a partial prefix-cache hit, which
    is the central quantity this project measures.
    """
    from sim.config import EngineConfig
    e = EngineConfig()
    assert e.prefill_s_per_token2 > 0, "quadratic prefill term has been zeroed"

    def chunk_cost(start, n):
        end = start + n
        return (e.prefill_s_per_token * n
                + e.prefill_s_per_token2 * (end * end - start * start))

    cold = chunk_cost(0, 4096)
    after_hit = chunk_cost(16384, 4096)
    assert after_hit > cold * 1.5, (
        f"4096 tokens at position 16384 cost {after_hit:.4f}s vs {cold:.4f}s at "
        f"position 0 -- the position dependence is too weak to be doing anything")


@check
def test_uniform_ttl_is_exactly_lru():
    """A constant TTL cannot differ from LRU, at any value. This is structural.

    In a demand-evicted pool a uniform TTL gives every block the same protection
    window, so expiry order equals release order equals LRU order, and the eviction
    sequence is unchanged. The consequence is the headline of EXP01: all of a TTL
    policy's value comes from its *variance across sessions*, i.e. from prediction.
    A tuned constant is not a competitor, it is the same policy under another name.
    """
    # n_protected_evictions is excluded on purpose: it counts how the mechanism
    # classified each eviction, not what the engine did. Under `lru` nothing is ever
    # protected and under `const_ttl` everything is, so it differs by definition while
    # the eviction *sequence* is the same.
    behavioural = [k for k in run(small()).summary if k != "n_protected_evictions"]
    ref = run(small(**{"policy.kind": "lru"}))
    for ttl in (0.5, 4.0, 60.0, 1e9):
        got = run(small(**{"policy.kind": "const_ttl", "policy.const_ttl_s": ttl}))
        for key in behavioural:
            assert ref.summary[key] == got.summary[key], (
                f"const_ttl({ttl}) diverged from lru on {key}: "
                f"{ref.summary[key]} vs {got.summary[key]}")
        # evict_rank is the number the policy stamped, i.e. the policy's own identity.
        def strip(records):
            return [{k: v for k, v in r.items() if k != "evict_rank"} for r in records]

        assert strip(ref.records) == strip(got.records), \
            f"const_ttl({ttl}) matched in aggregate but differs per request"


@check
def test_survives_heavy_preemption():
    """Regression: a request preempted mid-step must not be advanced as if it were live.

    Under enough memory pressure the scheduler preempts a sequence in the same step it
    finished prefilling. Before the fix, that sequence kept being decoded after its
    blocks were released and the run crashed in _grow_one_block.
    """
    cfg = Config().replace(**{
        "workload.n_sessions": 20,
        "arrival.concurrency": 20,
        # Big enough for the largest single context in this workload (~4100 blocks),
        # far too small for 20 of them. Below that the run legitimately raises.
        "engine.kv_pool_blocks": 8000,
    })
    for kind in ("lru", "const_ttl", "ttl_oracle", "oracle_terminal", "belady"):
        result = run(cfg.replace(**{"policy.kind": kind, "policy.const_ttl_s": 16.0}))
        assert result.summary["n_preemptions"] > 0, \
            f"{kind}: this config was supposed to thrash but never preempted"
        for rec in result.records:
            assert rec["cached_tokens"] + rec["prefill_tokens"] == rec["prompt_tokens"]


@check
def test_predict_with_perfect_input_equals_belady():
    """The predict arm fed the truth must be indistinguishable from belady."""
    cfg = small(**{"policy.kind": "belady"})
    sessions = generate_sessions(cfg.workload, cfg.seed)
    ref = run_config(cfg, sessions).summary

    table = {}
    for s in sessions:
        for t in s.turns:
            last = t.index == s.n_turns - 1
            table[(s.session_id, t.index)] = math.inf if last else t.pause_after_s
    got = run_config(cfg.replace(**{"policy.kind": "predict"}), sessions, table).summary
    for key in ("prefill_tokens_computed", "makespan_s"):
        assert math.isclose(ref[key], got[key], rel_tol=1e-9), (
            f"predict(truth) != belady on {key}: {ref[key]} vs {got[key]}")


@check
def test_no_result_directory_is_stale():
    """Every results/ directory must have been produced by the current config schema.

    This exists because of a real failure. The tool-multiplier normalisation changed the
    generator, and the decision not to rerun EXP01 was justified by checking that the
    change was unbiased for the *pause median* -- while the quantity actually being
    decided about was the *headroom*, which is a nonlinear function of it. The headroom
    at the peak moved 28%, and the stale numbers stayed in the findings docs until they
    were noticed by accident, days later.

    A config-schema mismatch is not a perfect staleness detector -- a change to a value
    rather than a field would slip through -- but it catches exactly the class of change
    that caused this one, and it costs nothing to run.
    """
    import json
    import os

    current = set(Config().to_dict()["workload"].keys())
    stale = []
    results = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")
    if not os.path.isdir(results):
        return
    for name in sorted(os.listdir(results)):
        meta = os.path.join(results, name, "metadata.json")
        if not os.path.isfile(meta):
            continue
        with open(meta, encoding="utf-8") as f:
            wl = json.load(f).get("config", {}).get("workload", {})
        missing = current - set(wl.keys())
        if missing:
            stale.append(f"{name} (missing {sorted(missing)})")

    known_stale = {
        # Kept on purpose: superseded runs are evidence of how the numbers moved, and
        # every findings doc that cites one says so and points at its replacement.
        "exp01", "exp01_seeds15", "exp02", "exp03",
        "v2_exp01_seeds15", "v2_exp02", "v2_exp03",
    }
    unexpected = [s for s in stale if s.split(" ")[0] not in known_stale]
    assert not unexpected, (
        "result directories produced by an older config schema and not declared stale: "
        + ", ".join(unexpected)
        + ". Either rerun them, or add them to known_stale with a note in the findings "
          "doc saying which run supersedes them.")


def main() -> int:
    failures = 0
    for fn in CHECKS:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
