"""Figure for EXP03: what a cache saving is worth, and under which billing model.

    python -m experiments.plot_exp03 --results results/v2_exp03

Panel A  Belady's advantage over LRU across the pause sweep, on three metrics: tokens
         recomputed, cost under wall-clock billing, cost under GPU-time billing. The
         gap between the green bar and the two blue ones is the conversion loss; the
         gap between the two blue ones is the billing model.
Panel B  the open-loop arm's stability. Where the runaway fraction is non-zero, the
         seeds are a bimodal mixture and that pause point's aggregates mean nothing --
         drawn so the reader can see which points to discard rather than being told.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/v2_exp03")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(args.results, "analysis.json"), encoding="utf-8") as f:
        report = json.load(f)
    with open(os.path.join(args.results, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    closed = sorted([p for p in report["points"] if p["loop"] == "closed"],
                    key=lambda p: p["pause_median_s"])
    openm = sorted([p for p in report["points"] if p["loop"] == "open_matched"],
                   key=lambda p: p["pause_median_s"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ---- A: three metrics, closed loop (the only arm that is stable everywhere)
    x = list(range(len(closed)))
    labels = [f"{p['pause_median_s']:g}s" for p in closed]
    tok = [100 * p["prefill_tokens_computed"]["headroom_frac"] for p in closed]
    wall = [100 * p["rm_per_1k_calls"]["headroom_frac"] for p in closed]
    gput = [100 * p["rm_gputime_per_1k_calls"]["headroom_frac"] for p in closed]
    busy = [100 * p["gpu_busy_frac"] for p in closed]

    w = 0.26
    ax_a.bar([i - w for i in x], tok, w, color="#2a9d5c", alpha=0.9,
             label="tokens recomputed")
    ax_a.bar(x, gput, w, color="#3b7dd8", alpha=0.9, label="cost, GPU-time billing")
    ax_a.bar([i + w for i in x], wall, w, color="#8fb8e8", alpha=0.95,
             label="cost, wall-clock billing")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_xlabel(f"median tool pause (closed loop, concurrency "
                    f"{meta['closed_concurrency']}, pressure ~{closed[0]['pressure_measured']:.2f})")
    ax_a.set_ylabel("Belady's advantage over LRU (%)")
    ax_a.set_title("A. A token saving is not a cost saving")
    ax_a2 = ax_a.twinx()
    ax_a2.plot(x, busy, color="#c8553d", marker="o", ms=5, lw=1.6, label="GPU busy")
    ax_a2.set_ylim(0, 105)
    ax_a2.set_ylabel("GPU busy (%)", fontsize=8, color="#c8553d")
    ax_a2.tick_params(labelsize=7, colors="#c8553d")
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper left")
    ax_a.grid(alpha=0.25, axis="y")

    # ---- B: open-loop instability
    xo = list(range(len(openm)))
    runaway = [100 * p["runaway_frac"] for p in openm]
    ax_b.bar(xo, runaway, 0.5, color="#c8553d", alpha=0.85,
             label="seeds whose live-session count ran away")
    for i, p in enumerate(openm):
        ax_b.plot([i, i], [0, 100], color="#d0d0d0", lw=0.6, zorder=0)
        ax_b.annotate(f"{p['live_min']:.0f}–{p['live_max']:.0f}",
                      xy=(i, min(runaway[i] + 4, 96)), ha="center", fontsize=7,
                      color="#4b5563")
    ax_b.set_xticks(xo)
    ax_b.set_xticklabels([f"{p['pause_median_s']:g}s" for p in openm])
    ax_b.set_ylim(0, 105)
    ax_b.set_xlabel("median tool pause (open loop, arrival rate calibrated to "
                    f"{meta['target_live']:.0f} mean live sessions)")
    ax_b.set_ylabel("fraction of seeds that ran away (%)")
    ax_b.set_title("B. Open loop has no stable operating point here")
    ax_b.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax_b.grid(alpha=0.25, axis="y")
    ax_b.text(0.02, 0.02,
              "labels: min–max live sessions across seeds\n"
              "a wide range at the same offered load means the run is bistable",
              transform=ax_b.transAxes, fontsize=7, color="#4b5563", va="bottom")

    fig.suptitle(
        f"EXP03 -- {meta['config']['workload']['n_sessions']} sessions x "
        f"{len(meta['seeds'])} seeds | CPU SIMULATION | timing constants MEASURED "
        f"against vLLM 0.26 + Qwen2.5-3B on RTX 5080 (docs/calibration.md)",
        fontsize=9, y=1.02)
    fig.tight_layout()

    out = args.out or os.path.join(args.results, "exp03.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
