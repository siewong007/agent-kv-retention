"""EXP05 -- where should the termination classifier's decision threshold sit?

EXP04 ran `predict_terminal` at a decision threshold of 0.5, which is the default for a
balanced problem. This problem is not balanced:

    false positive   a live session is marked dead, its cache is evicted first, and it
                     recomputes its whole context on return -- the full cost of a miss
    false negative   a dead session stays in LRU order, exactly where it would have been
                     without any predictor at all -- close to free

With costs that asymmetric, 0.5 is the wrong operating point almost by definition. This
experiment sweeps the threshold and asks two things:

  1. does a higher threshold rescue `predict_terminal` at signal strengths where EXP04
     found it losing?
  2. is there an optimum, and does it sit where the precision/recall trade-off predicts?

Run at two signal strengths taken from EXP04: a weak one (0.75) where the deployable arm
is indistinguishable from LRU, and a strong one (3.00) where it measurably wins.

    python -m experiments.exp05_threshold --seeds 15 --out results/exp05_seeds15
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    bootstrap_ratio as _bootstrap_ratio,
    default_workers, read_results, run_grid, write_results,
)
from experiments.exp04_predictor import CONCURRENCY, predict_table, train  # noqa: E402
from sim.config import Config  # noqa: E402
from sim.workload import generate_sessions  # noqa: E402

SIGNAL_GRID = [0.75, 3.0]
THRESHOLD_GRID = [0.30, 0.50, 0.70, 0.85, 0.95, 0.99]
REFERENCE_ARMS = ["lru", "oracle_terminal", "belady"]  # independent of the threshold


def build_jobs(base: Config, seeds: list[int]) -> tuple[list[dict], dict]:
    jobs = []
    quality: dict = {}
    for signal in SIGNAL_GRID:
        cfg = base.replace(**{"workload.termination_signal_strength": signal})
        print(f"training at termination_signal_strength={signal} ...", flush=True)
        model = train(cfg)

        for seed in seeds:
            eval_cfg = cfg.replace(seed=seed)
            sessions = generate_sessions(eval_cfg.workload, seed)

            # The reference arms do not depend on the threshold, so they are run once
            # per (signal, seed) and reused as the denominator for every threshold.
            for arm in REFERENCE_ARMS:
                jobs.append({
                    "config": eval_cfg.replace(**{"policy.kind": arm}).to_dict(),
                    "labels": {"signal": signal, "threshold": -1.0},
                })

            for thr in THRESHOLD_GRID:
                table, q = predict_table(model, sessions, threshold=thr)
                quality.setdefault((signal, thr), []).append(q)
                jobs.append({
                    "config": eval_cfg.replace(
                        **{"policy.kind": "predict_terminal"}).to_dict(),
                    "labels": {"signal": signal, "threshold": thr},
                    "pred_pause": table,
                })

    summarised = {}
    for (signal, thr), qs in quality.items():
        summarised[f"{signal}|{thr}"] = {
            "precision": st.fmean([q["terminal_precision"] for q in qs]),
            "recall": st.fmean([q["terminal_recall"] for q in qs]),
            "n_flagged_per_seed": st.fmean([q["terminal_tp"] + q["terminal_fp"] for q in qs]),
            "false_positives_per_seed": st.fmean([q["terminal_fp"] for q in qs]),
        }
    return jobs, summarised


def analyze(rows: list[dict], quality: dict, out_dir: str) -> dict:
    # Keyed by seed, not pooled, because the share is a ratio of two differences and its
    # uncertainty has to come from resampling whole seeds. Pooling first and dividing
    # afterwards throws away the pairing that makes the comparison paired in the first
    # place.
    ref: dict = {}
    for r in rows:
        if r["threshold"] < 0:
            ref[(r["signal"], r["policy"], r["seed"])] = r["prefill_tokens_computed"]

    by_point: dict = {}
    for r in rows:
        if r["threshold"] >= 0:
            by_point.setdefault((r["signal"], r["threshold"]), {})[r["seed"]] = \
                r["prefill_tokens_computed"]

    out = []
    for (signal, thr), got_by_seed in sorted(by_point.items()):
        seeds = sorted(got_by_seed)
        lru_s = [ref[(signal, "lru", sd)] for sd in seeds]
        bel_s = [ref[(signal, "belady", sd)] for sd in seeds]
        term_s = [ref[(signal, "oracle_terminal", sd)] for sd in seeds]
        got_s = [got_by_seed[sd] for sd in seeds]

        lru, bel, term, got = (st.fmean(v) for v in (lru_s, bel_s, term_s, got_s))
        gap = lru - bel
        q = quality.get(f"{signal}|{thr}", {})

        share_ci = _bootstrap_ratio(list(zip(lru_s, got_s)), list(zip(lru_s, bel_s)))
        gain_ci = _bootstrap_ratio(list(zip(lru_s, got_s)),
                                   [(v, 0.0) for v in lru_s])
        out.append({
            "signal": signal,
            "threshold": thr,
            "n_seeds": len(seeds),
            "lru": lru,
            "belady": bel,
            "oracle_terminal": term,
            "predict_terminal": got,
            "share_of_gap": ((lru - got) / gap) if gap else None,
            "share_oracle_terminal": ((lru - term) / gap) if gap else None,
            "gain_vs_lru_frac": (lru - got) / lru if lru else None,
            "share_ci": share_ci,
            "gain_ci": gain_ci,
            "precision": q.get("precision"),
            "recall": q.get("recall"),
            "false_positives_per_seed": q.get("false_positives_per_seed"),
        })

    report = {"points": out}
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def print_report(report: dict) -> None:
    print()
    print("=" * 96)
    print("EXP05  decision threshold for the termination classifier")
    print("=" * 96)
    for signal in sorted({p["signal"] for p in report["points"]}):
        pts = sorted([p for p in report["points"] if p["signal"] == signal],
                     key=lambda p: p["threshold"])
        orc = pts[0]["share_oracle_terminal"]
        print(f"\n  termination_signal_strength = {signal}   "
              f"(oracle_terminal captures {100*orc:.1f}% of the gap)")
        print(f"  {'thresh':>7} {'prec':>6} {'recall':>7} {'FP/seed':>8} "
              f"{'gain vs LRU % [95% CI]':>26}  {'share of gap %':>14}")
        print("  " + "-" * 72)
        best = max(pts, key=lambda p: p["share_of_gap"])
        for p in pts:
            g = p.get("gain_ci") or {}
            lo, hi = g.get("lo"), g.get("hi")
            if lo is None:
                ci = f"{100*p['gain_vs_lru_frac']:>8.2f}"
            else:
                beats = "  " if (lo <= 0 <= hi) else " *"
                ci = (f"{100*p['gain_vs_lru_frac']:>7.2f} "
                      f"[{100*lo:>6.2f},{100*hi:>6.2f}]{beats}")
            mark = "  <-- best" if p is best else ""
            print(f"  {p['threshold']:>7.2f} {p['precision']:>6.3f} "
                  f"{p['recall']:>7.3f} {p['false_positives_per_seed']:>8.0f} "
                  f"{ci:>26}  {100*p['share_of_gap']:>13.1f}%{mark}")
    print("\n'*' marks rows whose 95% interval excludes zero, i.e. the predictor "
          "measurably beats")
    print("doing nothing. Rows without it are not distinguishable from LRU no matter "
          "what their")
    print("point estimate says. Intervals are paired bootstrap over seeds, 5000 "
          "resamples, seeded.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/exp05")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--reanalyze", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if args.reanalyze:
        with open(os.path.join(args.out, "metadata.json"), encoding="utf-8") as f:
            quality = json.load(f)["classifier_quality"]
        print_report(analyze(read_results(args.out), quality, args.out))
        return 0

    base = Config.load(args.config).replace(**{
        "workload.n_sessions": args.sessions,
        "arrival.concurrency": CONCURRENCY,
    })
    seeds = list(range(args.seeds))
    jobs, quality = build_jobs(base, seeds)
    rows = run_grid(jobs, args.workers, "EXP05")
    csv_path = write_results(args.out, rows, base, {
        "experiment": "exp05_threshold",
        "seeds": seeds,
        "signal_grid": SIGNAL_GRID,
        "threshold_grid": THRESHOLD_GRID,
        "concurrency": CONCURRENCY,
        "classifier_quality": quality,
    })
    print_report(analyze(rows, quality, args.out))
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
