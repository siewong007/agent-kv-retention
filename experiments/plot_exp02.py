"""Figure for EXP02: does the phenomenon live on the pressure axis?

    python -m experiments.plot_exp02 --results results/exp02

Panel A plots hit rate against measured pressure, one line per way of reaching that
pressure. Panel B does the same for the Belady headroom. Panel C plots the disagreement
between the three conditions -- the actual test. Where the spread is small, a result
stated at that pressure transfers to other hardware; where it is large, the result
belongs to the machine it was measured on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COND_STYLE = {
    "vary_concurrency": ("#3b7dd8", "o", "-", "pool fixed at 16 GB, sessions scaled"),
    "vary_pool": ("#e0a11b", "s", "--", "16 sessions, pool scaled"),
    "bigger_gpu": ("#2a9d5c", "D", "-.", "pool fixed at 2.5x, sessions scaled"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/exp02")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(args.results, "analysis.json"), encoding="utf-8") as f:
        report = json.load(f)
    with open(os.path.join(args.results, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 4.8))

    for cond, (color, marker, ls, label) in COND_STYLE.items():
        pts = sorted([p for p in report["points"] if p["condition"] == cond],
                     key=lambda p: p["pressure_measured"])
        if not pts:
            continue
        x = [p["pressure_measured"] for p in pts]
        ax_a.plot(x, [100 * p["hit_rate_lru"] for p in pts],
                  color=color, marker=marker, ls=ls, ms=4.5, label=label)
        ax_b.plot(x, [100 * p["headroom_frac"] for p in pts],
                  color=color, marker=marker, ls=ls, ms=4.5, label=label)

    for ax, ylabel, title in (
            (ax_a, "LRU prefix-cache hit rate (%)", "A. The collapse, on the pressure axis"),
            (ax_b, "Belady headroom over LRU (%)", "B. Headroom, on the pressure axis")):
        ax.axvline(1.0, color="#8a8f98", lw=0.9, ls=":")
        ax.annotate("working set = pool", xy=(1.0, ax.get_ylim()[1]),
                    xytext=(1.02, 0.94), textcoords="axes fraction",
                    fontsize=7.5, color="#6b7280")
        ax.set_xlabel("measured pressure  (live sessions x context blocks / pool blocks)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(alpha=0.25)

    # ---- C: the collapse test itself
    collapse = report["collapse"]
    xc = [c["target_pressure"] for c in collapse]
    hit_range = [100 * c["hit_rate_range"] for c in collapse]
    head_range = [100 * c["headroom_range"] for c in collapse]
    w = 0.05
    ax_c.bar([x - w for x in xc], hit_range, 2 * w, color="#3b7dd8", alpha=0.85,
             label="hit rate: spread across the three conditions")
    ax_c.bar([x + w for x in xc], head_range, 2 * w, color="#2a9d5c", alpha=0.85,
             label="headroom: spread across the three conditions")
    ax_c.axvline(1.0, color="#8a8f98", lw=0.9, ls=":")
    ax_c.set_xlabel("target pressure")
    ax_c.set_ylabel("disagreement between conditions (percentage points)")
    ax_c.set_title("C. Where the axis transfers, and where it stops")
    ax_c.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax_c.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"EXP02 -- the same pressure reached three ways. "
        f"{len(meta['seeds'])} seeds, {meta['mean_context_blocks']:.0f} blocks per live session. "
        f"Low bars in C mean the result transfers to other hardware. | "
        f"CPU SIMULATION | timing constants MEASURED against vLLM 0.26 + Qwen2.5-3B on RTX 5080 (docs/calibration.md)",
        fontsize=9, y=1.02)
    fig.tight_layout()

    out = args.out or os.path.join(args.results, "exp02.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
