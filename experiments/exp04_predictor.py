"""EXP04 -- how much of the oracle's headroom can a real predictor actually get?

EXP01 established that the headroom exists (18.3% at the peak), that no tuned constant
can reach any of it, and that most of it at the operating point comes from knowing
whether a session has ENDED rather than how long it pauses. This experiment builds the
deployable version and measures what it captures.

Two predictors, matching the two oracle arms so the comparison is clean:

    terminal classifier   "is this the session's last turn?"  -> `predict_terminal`,
                          which is `oracle_terminal` with predictions instead of truth
    pause regressor       "how long until this session comes back?" -> combined with the
                          classifier into `predict`, which is `belady` with predictions

Gradient boosting, not a neural network: the feature set is six columns and a few
thousand rows, and the point is to find out whether the *information* is there, not to
squeeze a benchmark.

Discipline that decides whether the number means anything:

  * **Trained on different sessions than it is evaluated on.** Training uses its own
    seeds; evaluation uses the experiment seeds. Fitting and evaluating on one workload
    would report memorisation.
  * **Only causally available features.** A prediction is made when a turn FINISHES,
    so it may use the turn index, the prompt and output sizes, the tool just used, the
    previous tool and the previous pause. It may not use anything from the future.
  * **Swept over `tool_pause_spread`.** That knob sets how much of the pause a tool's
    identity explains -- i.e. how learnable the synthetic world is. A predictor result
    at a single setting says as much about the generator as about the predictor, so the
    answer is reported as a curve. At spread 0 the pause is unlearnable by construction
    and any apparent skill is leakage.

    python -m experiments.exp04_predictor --seeds 10 --out results/exp04
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    default_workers, read_results, run_grid, write_results,
)
from sim.config import Config  # noqa: E402
from sim.workload import Session, generate_sessions  # noqa: E402

# Primary axis. EXP01 says most of the headroom at the operating point comes from
# knowing a session has ended, and the first EXP04 attempt showed the generator had
# no termination signal at all to learn. This dial supplies one, indirectly.
SIGNAL_GRID = [0.0, 0.75, 1.5, 3.0]
ARMS = ["lru", "oracle_terminal", "belady", "belady_pause",
        "predict_terminal", "predict", "predict_guarded"]
CONCURRENCY = 10          # the headroom peak: pressure 0.84 with the measured pool
TRAIN_SEED_BASE = 9000    # disjoint from the evaluation seeds by construction
TRAIN_SESSIONS = 600

FEATURES = ["turn_index", "prompt_tokens", "output_tokens", "prev_pause_s",
            "tool_id", "prev_tool_id"]


def extract(sessions: list[Session]) -> tuple[list[list[float]], list[int], list[float]]:
    """Features and both labels, one row per turn.

    Everything here is knowable at the moment the turn finishes. `prev_pause_s` is the
    pause that already elapsed before this turn started, not the one being predicted.
    """
    X: list[list[float]] = []
    y_terminal: list[int] = []
    y_pause: list[float] = []
    for s in sessions:
        prev_pause = 0.0
        prev_tool = -1
        for t in s.turns:
            X.append([
                float(t.index),
                float(t.prompt_tokens),
                float(t.output_tokens),
                float(prev_pause),
                float(t.tool_id),
                float(prev_tool),
            ])
            is_last = t.index == s.n_turns - 1
            y_terminal.append(1 if is_last else 0)
            y_pause.append(t.pause_after_s)
            prev_pause = t.pause_after_s
            prev_tool = t.tool_id
    return X, y_terminal, y_pause


def train(cfg: Config) -> dict:
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  GradientBoostingRegressor)

    train_cfg = cfg.replace(**{"workload.n_sessions": TRAIN_SESSIONS})
    sessions = []
    for k in range(3):  # a few seeds so the training set is not one draw of tool speeds
        sessions += generate_sessions(train_cfg.workload, TRAIN_SEED_BASE + k)
    X, y_term, y_pause = extract(sessions)

    clf = GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3)
    clf.fit(X, y_term)

    # Regress log(pause): the target is lognormal, so squared error on the raw seconds
    # would be dominated by the tail and fit the mean of a heavy-tailed variable.
    live = [(row, p) for row, t, p in zip(X, y_term, y_pause) if not t and p > 0]
    reg = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3)
    reg.fit([r for r, _ in live], [math.log(p) for _, p in live])

    return {"clf": clf, "reg": reg, "n_train_rows": len(X), "n_live_rows": len(live)}


def predict_table(model: dict, sessions: list[Session],
                  threshold: float = 0.5) -> tuple[dict, dict]:
    """Map (session_id, turn) -> predicted pause seconds, or inf for predicted-terminal."""
    X, y_term, y_pause = extract(sessions)
    p_term = model["clf"].predict_proba(X)[:, 1]
    log_pause = model["reg"].predict(X)

    table: dict = {}
    idx = 0
    tp = fp = tn = fn = 0
    log_err: list[float] = []
    for s in sessions:
        for t in s.turns:
            terminal = p_term[idx] >= threshold
            table[(s.session_id, t.index)] = (math.inf if terminal
                                              else math.exp(log_pause[idx]))
            truth = y_term[idx] == 1
            if terminal and truth:
                tp += 1
            elif terminal and not truth:
                fp += 1
            elif not terminal and not truth:
                tn += 1
            else:
                fn += 1
            if not truth and y_pause[idx] > 0:
                log_err.append(log_pause[idx] - math.log(y_pause[idx]))
            idx += 1

    quality = {
        "terminal_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "terminal_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "terminal_tp": tp, "terminal_fp": fp, "terminal_fn": fn, "terminal_tn": tn,
        # A false positive evicts a session that is coming back; a false negative just
        # leaves it in LRU order. They are not symmetric and precision is the one that
        # costs.
        "pause_log_rmse": (math.sqrt(st.fmean([e * e for e in log_err]))
                           if log_err else float("nan")),
        "pause_median_ratio_error": (math.exp(st.median(log_err))
                                     if log_err else float("nan")),
    }
    return table, quality


def build_jobs(base: Config, seeds: list[int]) -> tuple[list[dict], dict]:
    jobs = []
    quality_by_spread = {}
    for spread in SIGNAL_GRID:
        cfg = base.replace(**{"workload.termination_signal_strength": spread})
        print(f"training predictors at termination_signal_strength={spread} ...",
              flush=True)
        model = train(cfg)
        qualities = []
        for seed in seeds:
            eval_cfg = cfg.replace(seed=seed)
            sessions = generate_sessions(eval_cfg.workload, seed)
            table, quality = predict_table(model, sessions)
            qualities.append(quality)
            for arm in ARMS:
                job_cfg = eval_cfg.replace(**{"policy.kind": arm})
                job = {"config": job_cfg.to_dict(),
                       "labels": {"termination_signal_strength": spread}}
                if arm.startswith("predict"):
                    job["pred_pause"] = table
                jobs.append(job)
        quality_by_spread[spread] = {
            "terminal_precision": st.fmean([q["terminal_precision"] for q in qualities]),
            "terminal_recall": st.fmean([q["terminal_recall"] for q in qualities]),
            "pause_log_rmse": st.fmean([q["pause_log_rmse"] for q in qualities]),
            "pause_median_ratio_error": st.fmean(
                [q["pause_median_ratio_error"] for q in qualities]),
            "n_train_rows": model["n_train_rows"],
        }
        q = quality_by_spread[spread]
        print(f"  terminal precision {q['terminal_precision']:.3f} "
              f"recall {q['terminal_recall']:.3f} | "
              f"pause log-RMSE {q['pause_log_rmse']:.3f}", flush=True)
    return jobs, quality_by_spread


def analyze(rows: list[dict], quality: dict, out_dir: str) -> dict:
    points = {}
    for r in rows:
        points.setdefault(r["termination_signal_strength"], []).append(r)

    out = []
    for spread, group in sorted(points.items()):
        def mean_of(policy, metric="prefill_tokens_computed"):
            vals = [r[metric] for r in group if r["policy"] == policy]
            return st.fmean(vals) if vals else None

        lru = mean_of("lru")
        bel = mean_of("belady")
        gap = lru - bel if (lru and bel) else 0.0

        def share(policy):
            v = mean_of(policy)
            return ((lru - v) / gap) if (gap > 0 and v is not None) else None

        entry = {
            "termination_signal_strength": spread,
            "n_seeds": len({r["seed"] for r in group}),
            "lru": lru,
            "belady": bel,
            "oracle_terminal": mean_of("oracle_terminal"),
            "belady_pause": mean_of("belady_pause"),
            "predict_terminal": mean_of("predict_terminal"),
            "predict": mean_of("predict"),
            "predict_guarded": mean_of("predict_guarded"),
            "headroom_frac": gap / lru if lru else 0.0,
            "share_oracle_terminal": share("oracle_terminal"),
            "share_belady_pause": share("belady_pause"),
            "share_predict_terminal": share("predict_terminal"),
            "share_predict": share("predict"),
            "share_predict_guarded": share("predict_guarded"),
            "quality": quality.get(spread) or quality.get(str(spread)),
        }
        out.append(entry)

    report = {"points": out}
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def print_report(report: dict) -> None:
    print()
    print("=" * 100)
    print("EXP04  share of the Belady headroom captured, as the world is made more "
          "learnable")
    print("=" * 100)
    print(f"{'signal':>7} {'headroom':>9} | {'orc_term':>9} {'bel_pause':>10} "
          f"{'pred_term':>10} {'pred_grd':>9} {'predict':>10} | "
          f"{'prec':>6} {'recall':>7}")
    print("-" * 100)
    for p in report["points"]:
        q = p["quality"] or {}

        def pct(v):
            return f"{100 * v:>9.1f}%" if v is not None else "        --"

        print(f"{p['termination_signal_strength']:>7.2f} "
              f"{100*p['headroom_frac']:>8.1f}% | "
              f"{pct(p['share_oracle_terminal']):>9} "
              f"{pct(p['share_belady_pause']):>10} "
              f"{pct(p['share_predict_terminal']):>10} "
              f"{pct(p['share_predict_guarded']):>9} "
              f"{pct(p['share_predict']):>10} | "
              f"{q.get('terminal_precision', float('nan')):>6.3f} "
              f"{q.get('terminal_recall', float('nan')):>7.3f}")
    print()
    print("orc_term vs pred_term  = what the classifier loses against perfect "
          "termination knowledge.")
    print("bel_pause vs belady    = ranking by pause alone versus by absolute next-use")
    print("                         time. Values over 100% mean `belady` is NOT an")
    print("                         upper bound in this system: it is myopic (next use")
    print("                         only, not the whole future) and the reference")
    print("                         stream is not fixed, since eviction changes when")
    print("                         work is recomputed.")
    print("pred_grd / predict     = pause-length prediction layered on top. Both go")
    print("                         sharply negative: overriding LRU with a noisy")
    print("                         ordering destroys the signal LRU already exploits,")
    print("                         namely that an old block belongs to a dead session.")
    print("At signal 0.00 termination is unlearnable by construction, so the predict")
    print("arms there are a floor, not a result.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/exp04")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--reanalyze", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if args.reanalyze:
        with open(os.path.join(args.out, "metadata.json"), encoding="utf-8") as f:
            quality = json.load(f)["predictor_quality"]
        print_report(analyze(read_results(args.out), quality, args.out))
        return 0

    base = Config.load(args.config).replace(**{
        "workload.n_sessions": args.sessions,
        "arrival.concurrency": CONCURRENCY,
    })
    seeds = list(range(args.seeds))
    jobs, quality = build_jobs(base, seeds)
    rows = run_grid(jobs, args.workers, "EXP04")
    csv_path = write_results(args.out, rows, base, {
        "experiment": "exp04_predictor",
        "seeds": seeds,
        "signal_grid": SIGNAL_GRID,
        "arms": ARMS,
        "concurrency": CONCURRENCY,
        "train_seed_base": TRAIN_SEED_BASE,
        "train_sessions": TRAIN_SESSIONS,
        "features": FEATURES,
        "predictor_quality": quality,
    })
    print_report(analyze(rows, quality, args.out))
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
