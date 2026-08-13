"""Figure for EXP01: the collapse, the headroom, and whether the headroom is money.

    pip install matplotlib
    python -m experiments.plot_exp01 --results results/exp01

Two panels:
  A  prefill tokens recomputed vs concurrency, one line per arm -- where the cache
     collapses and how far the upper bound is from the incumbent;
  B  the headroom decomposed -- how much of the belady gap a tuned constant TTL already
     captures, how much knowing only "the session ended" adds, how much is left for
     actual pause-length prediction.

The cost question used to be a third panel here. It now belongs to EXP03, which answers
it properly: 15 seeds, both billing models, and a pause sweep that does not confound
"longer pause" with "lower load". See experiments/plot_exp03.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARM_STYLE = {
    "lru": ("#6b7280", "-", "o", "LRU = any constant TTL (identical, see B)"),
    "ttl_oracle": ("#c8553d", ":", "v", "true pause used as a TTL (wrong mechanism)"),
    "oracle_terminal": ("#e0a11b", "--", "^", "oracle: only knows the session ended"),
    "belady": ("#2a9d5c", "-", "D", "Belady (upper bound)"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/exp01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(args.results, "analysis.json"), encoding="utf-8") as f:
        report = json.load(f)
    with open(os.path.join(args.results, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    pts = sorted([p for p in report["points"] if p["sweep"] == "concurrency"],
                 key=lambda p: p["concurrency"])
    x = [p["concurrency"] for p in pts]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # ---- A: the collapse
    for arm, (color, ls, marker, label) in ARM_STYLE.items():
        y = [p["prefill_tokens_computed"][arm] / 1e6 for p in pts]
        lw = 2.4 if arm == "lru" else 1.4
        ax_a.plot(x, y, color=color, linestyle=ls, marker=marker, ms=4, lw=lw, label=label)
    hit = [100 * p["token_hit_rate"]["lru"] for p in pts]
    ax_a2 = ax_a.twinx()
    ax_a2.plot(x, hit, color="#6b7280", lw=0.8, alpha=0.45)
    ax_a2.set_ylabel("LRU prefix-cache hit rate (%)", fontsize=8, color="#6b7280")
    ax_a2.tick_params(labelsize=7, colors="#6b7280")
    ax_a.set_xlabel("concurrent agent sessions")
    ax_a.set_ylabel("prompt tokens recomputed (millions)")
    ax_a.set_title("A. Cache collapse, and how far the bound is from it")
    ax_a.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax_a.grid(alpha=0.25)

    # ---- B: headroom decomposition, with bootstrap intervals over seeds
    head = [100 * p["prefill_tokens_computed"]["headroom_frac"] for p in pts]
    const = [100 * p["prefill_tokens_computed"]["const_gain_frac"] for p in pts]
    term = [100 * p["prefill_tokens_computed"]["terminal_gain_frac"] for p in pts]
    band = [max(c, t) for c, t in zip(const, term)]
    ax_b.fill_between(x, 0, band, color="#e0a11b", alpha=0.8,
                      label="knowing only that the session ended")
    ax_b.fill_between(x, band, head, color="#2a9d5c", alpha=0.8,
                      label="left for pause-length prediction")

    hci = [p["prefill_tokens_computed"].get("headroom_ci") for p in pts]
    if all(c and c.get("lo") is not None for c in hci):
        point = [100 * c["point"] for c in hci]
        lo = [100 * (c["point"] - c["lo"]) for c in hci]
        hi = [100 * (c["hi"] - c["point"]) for c in hci]
        n_seeds = hci[0]["n_seeds"]
        ax_b.errorbar(x, point, yerr=[lo, hi], color="#1d1d1f", lw=1.4,
                      capsize=3, marker="o", ms=3.5,
                      label=f"total headroom vs LRU (95% CI, {n_seeds} seeds)")
    else:
        ax_b.plot(x, head, color="#1d1d1f", lw=1.4, label="total headroom vs LRU")

    ax_b.plot(x, const, color="#3b7dd8", lw=2.0,
              label="captured by a tuned constant TTL (exactly zero)")
    ax_b.set_xlabel("concurrent agent sessions")
    ax_b.set_ylabel("reduction in recomputed tokens vs LRU (%)")
    ax_b.set_title("B. Who can get the headroom")
    ax_b.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax_b.grid(alpha=0.25)

    wl = meta["workload_summary"]
    fig.suptitle(
        f"EXP01 -- synthetic agent workload, {meta['config']['workload']['n_sessions']} sessions x "
        f"{len(meta['seeds'])} seeds, median prompt {wl['prompt_tokens_median']:.0f} tok, "
        f"KV pool {meta['config']['engine']['kv_pool_blocks']} blocks (MEASURED) | "
        f"CPU SIMULATION | timing constants MEASURED against vLLM 0.26 + Qwen2.5-3B on RTX 5080 (docs/calibration.md)",
        fontsize=9, y=1.02)
    fig.tight_layout()

    out = args.out or os.path.join(args.results, "exp01.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
