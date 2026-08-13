"""EXP05 -- where should the termination classifier's decision threshold sit?

EXP04 found that `predict_terminal` loses to LRU below roughly 0.7 precision. It ran at
a decision threshold of 0.5, which is the default for a balanced problem and this problem
is not balanced:

    false positive   a live session is marked dead, its cache is evicted first, and it
                     recomputes its whole context on return -- the full cost of a miss
    false negative   a dead session stays in LRU order, exactly where it would have been
                     without any predictor at all -- close to free

With costs that asymmetric, 0.5 is the wrong operating point almost by definition. This
experiment sweeps the threshold and asks two things:

  1. does a higher threshold rescue `predict_terminal` at signal strengths where EXP04
     found it losing?
  2. is there an optimum, and does it sit where the precision/recall trade-off predicts?

Run at two signal strengths taken from EXP04: one below its crossover (0.75, where
predict_terminal scored -9.5%) and one above (3.00, where it scored +57.9%).

    python -m experiments.exp05_threshold --seeds 10 --out results/exp05
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
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
    ref = {}
    for r in rows:
        if r["threshold"] < 0:
            ref.setdefault((r["signal"], r["policy"]), []).append(
                r["prefill_tokens_computed"])

    out = []
    by_point: dict = {}
    for r in rows:
        if r["threshold"] >= 0:
            by_point.setdefault((r["signal"], r["threshold"]), []).append(r)

    for (signal, thr), group in sorted(by_point.items()):
        lru = st.fmean(ref[(signal, "lru")])
        bel = st.fmean(ref[(signal, "belady")])
        term = st.fmean(ref[(signal, "oracle_terminal")])
        got = st.fmean([r["prefill_tokens_computed"] for r in group])
        gap = lru - bel
        q = quality.get(f"{signal}|{thr}", {})
        out.append({
            "signal": signal,
            "threshold": thr,
            "lru": lru,
            "belady": bel,
            "oracle_terminal": term,
            "predict_terminal": got,
            "share_of_gap": ((lru - got) / gap) if gap else None,
            "share_oracle_terminal": ((lru - term) / gap) if gap else None,
            "gain_vs_lru_frac": (lru - got) / lru if lru else None,
            "precision": q.get("precision"),
            "recall": q.get("recall"),
            "false_positives_per_seed": q.get("false_positives_per_seed"),
            "n_seeds": len({r["seed"] for r in group}),
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
        print(f"  {'thresh':>7} {'precision':>10} {'recall':>8} {'FP/seed':>9} "
              f"{'share of gap':>13} {'vs LRU':>9}")
        print("  " + "-" * 62)
        best = max(pts, key=lambda p: p["share_of_gap"])
        for p in pts:
            mark = "  <-- best" if p is best else ""
            print(f"  {p['threshold']:>7.2f} {p['precision']:>10.3f} "
                  f"{p['recall']:>8.3f} {p['false_positives_per_seed']:>9.0f} "
                  f"{100*p['share_of_gap']:>12.1f}% "
                  f"{100*p['gain_vs_lru_frac']:>8.1f}%{mark}")
    print("\nA positive 'vs LRU' means the predictor beats doing nothing. EXP04 measured")
    print("only the 0.50 row; if higher thresholds turn a negative row positive, EXP04's")
    print("precision floor was an artefact of the operating point, not of the method.")


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
