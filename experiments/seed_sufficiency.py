"""Is the seed count enough for the claim being made?

Adding seeds has overturned a published number in this project four times, so "run more
seeds" has become the reflex. It is the wrong reflex on its own: more seeds are only
worth buying if the interval is currently too wide *for the specific claim*, and the
cost grows as the square of the precision wanted.

This script answers the question directly, without running any simulation:

  1. **Current width.** For each quantity, the 95% paired-bootstrap interval at the
     seeds already collected.
  2. **Empirical shrinkage.** The same interval recomputed on random subsets of the
     seeds, which shows whether the width is actually falling like 1/sqrt(n) or whether
     something systematic is holding it up. EXP02 is the cautionary case: going 10 -> 15
     barely moved it, because the residual spread was a real effect and not noise.
  3. **Projected cost.** Seeds needed to reach a target half-width, from the fitted
     shrinkage rather than from the textbook rate.

A claim of the form "X beats Y" needs an interval excluding zero, and that is cheap. A
claim of the form "X equals Y" needs an interval narrow enough to exclude any difference
that would matter, and that is expensive -- often 10x the seeds. The two are reported
separately because they are not the same purchase.

    python -m experiments.seed_sufficiency --results results/v2_exp01_seeds15 --exp exp01
    python -m experiments.seed_sufficiency --results results/v2_exp03 --exp exp03
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import bootstrap_ratio, read_results  # noqa: E402

SUBSET_SIZES = [4, 6, 8, 10, 12]
SUBSET_DRAWS = 24


def _ci_width(lru: dict, arm: dict, seeds: list[int], denom: dict | None = None) -> float | None:
    """Width of the 95% interval for (lru-arm)/lru, or /(lru-denom) if given."""
    pairs = [(lru[s], arm[s]) for s in seeds]
    dens = ([(lru[s], denom[s]) for s in seeds] if denom
            else [(lru[s], 0.0) for s in seeds])
    ci = bootstrap_ratio(pairs, dens, n_boot=1500)
    if not ci or ci.get("lo") is None:
        return None
    return ci["hi"] - ci["lo"]


def shrinkage(lru: dict, arm: dict, denom: dict | None, seeds: list[int]) -> dict:
    """Interval width at the full seed set and at random subsets of it."""
    rng = random.Random(4242)
    widths = {}
    for k in [s for s in SUBSET_SIZES if s < len(seeds)] + [len(seeds)]:
        if k == len(seeds):
            w = _ci_width(lru, arm, seeds, denom)
            widths[k] = w
            continue
        draws = []
        for _ in range(SUBSET_DRAWS):
            sub = rng.sample(seeds, k)
            w = _ci_width(lru, arm, sub, denom)
            if w is not None:
                draws.append(w)
        widths[k] = st.fmean(draws) if draws else None

    # Fit width = c * n^(-p). Pure sampling noise gives p = 0.5; a systematic
    # disagreement that seeds cannot remove drags p toward 0.
    pts = [(k, w) for k, w in widths.items() if w and w > 0]
    if len(pts) >= 3:
        xs = [math.log(k) for k, _ in pts]
        ys = [math.log(w) for _, w in pts]
        mx, my = st.fmean(xs), st.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx) if sxx else 0.0
        exponent = -slope
        intercept = my - slope * mx
    else:
        exponent, intercept = float("nan"), float("nan")
    return {"widths": widths, "exponent": exponent, "log_intercept": intercept}


def seeds_for_width(fit: dict, target_width: float) -> float | None:
    p, c = fit["exponent"], fit["log_intercept"]
    if not (p and p == p and p > 0.05):
        return None
    return math.exp((c - math.log(target_width)) / p)


def report(name: str, lru: dict, arm: dict, denom: dict | None, seeds: list[int],
           target: float) -> None:
    fit = shrinkage(lru, arm, denom, seeds)
    ci = bootstrap_ratio([(lru[s], arm[s]) for s in seeds],
                         [(lru[s], denom[s]) for s in seeds] if denom
                         else [(lru[s], 0.0) for s in seeds])
    lo, hi = (ci or {}).get("lo"), (ci or {}).get("hi")
    excl = "" if lo is None else ("yes" if not (lo <= 0 <= hi) else "NO")
    widths = " ".join(f"n={k}:{100*w:.1f}" for k, w in sorted(fit["widths"].items()) if w)
    need = seeds_for_width(fit, target)
    need_txt = f"{need:.0f}" if need and need < 1e5 else "impractical"
    print(f"  {name:<34} point {100*(ci or {}).get('point', float('nan')):>7.1f}%  "
          f"width {100*(hi-lo) if lo is not None else float('nan'):>6.1f}pp  "
          f"excl.0 {excl:>3}  decay n^-{fit['exponent']:.2f}  "
          f"n for +/-{100*target/2:.0f}pp: {need_txt}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--exp", required=True, choices=["exp01", "exp03"])
    ap.add_argument("--target-width", type=float, default=0.10,
                    help="target full interval width, as a fraction (0.10 = +/-5pp)")
    args = ap.parse_args(argv)

    rows = read_results(args.results)
    seeds = sorted({int(r["seed"]) for r in rows})
    print(f"{args.results}: {len(seeds)} seeds\n")

    def series(sel, policy, metric):
        return {int(r["seed"]): r[metric] for r in rows
                if r["policy"] == policy and sel(r)}

    if args.exp == "exp01":
        concs = sorted({r["concurrency"] for r in rows if r["sweep"] == "concurrency"})
        for metric, label in (("prefill_tokens_computed", "tokens"),
                              ("rm_per_1k_calls", "cost")):
            print(f"--- {label}: headroom vs LRU, and the termination share of it ---")
            for c in concs:
                sel = lambda r, c=c: r["sweep"] == "concurrency" and r["concurrency"] == c
                lru = series(sel, "lru", metric)
                bel = series(sel, "belady", metric)
                term = series(sel, "oracle_terminal", metric)
                ttl = series(sel, "ttl_oracle", metric)
                sd = sorted(set(lru) & set(bel) & set(term) & set(ttl))
                if len(sd) < 4:
                    continue
                report(f"conc {c:.0f} headroom", lru, bel, None, sd, args.target_width)
                report(f"conc {c:.0f} termination share", lru, term, bel, sd,
                       args.target_width)
                report(f"conc {c:.0f} ttl_oracle vs LRU", lru, ttl, None, sd,
                       args.target_width)
            print()
    else:
        for loop in ("closed", "open_matched"):
            print(f"--- {loop}: token headroom vs LRU ---")
            pauses = sorted({r["pause_median_s"] for r in rows if r["loop"] == loop})
            for p in pauses:
                sel = lambda r, p=p, loop=loop: r["loop"] == loop and r["pause_median_s"] == p
                lru = series(sel, "lru", "prefill_tokens_computed")
                bel = series(sel, "belady", "prefill_tokens_computed")
                sd = sorted(set(lru) & set(bel))
                if len(sd) < 4:
                    continue
                report(f"pause {p:g}s headroom", lru, bel, None, sd, args.target_width)
            print()

        # The one claim EXP03 rests on: open and closed agree at a 30 s pause. Overlapping
        # intervals are weak evidence for that; an interval on the DIFFERENCE is the right
        # test, and it is not the same computation.
        print("--- the open-vs-closed agreement at 30 s, tested properly ---")
        def head(loop):
            sel = lambda r: r["loop"] == loop and r["pause_median_s"] == 30.0
            lru = series(sel, "lru", "prefill_tokens_computed")
            bel = series(sel, "belady", "prefill_tokens_computed")
            sd = sorted(set(lru) & set(bel))
            return {s: (lru[s] - bel[s]) / lru[s] for s in sd}

        a, b = head("closed"), head("open_matched")
        sd = sorted(set(a) & set(b))
        rng = random.Random(999)
        diffs = []
        for _ in range(5000):
            idx = [rng.choice(sd) for _ in sd]
            diffs.append(st.fmean([a[i] for i in idx]) - st.fmean([b[i] for i in idx]))
        diffs.sort()
        lo, hi = diffs[124], diffs[4875]
        point = st.fmean([a[s] for s in sd]) - st.fmean([b[s] for s in sd])
        verdict = ("indistinguishable" if lo <= 0 <= hi
                   else "MEASURABLY DIFFERENT -- the agreement claim does not hold")
        print(f"  closed minus open headroom: {100*point:+.1f}pp "
              f"[{100*lo:+.1f}, {100*hi:+.1f}]  -> {verdict}")
        print(f"  note: seeds are paired only by index here, not by workload -- the two")
        print(f"  loops schedule the same sessions differently, so this is a two-sample")
        print(f"  comparison and the interval is wider than a truly paired one would be.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
