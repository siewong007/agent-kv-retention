#!/usr/bin/env bash
# Re-run all five experiments on whatever constants sim/config.py currently holds.
#
# Run this after hpc/adopt_constants.py --apply. Nothing here touches a GPU: the simulator
# is stdlib Python and the experiments are CPU sweeps, so the entire cost of switching the
# thesis to HPC-calibrated numbers is the ~2.5 GPU-hours of the calibration round itself.
# This part is free and can run on a laptop while the cluster queue does something else.
#
# Output goes to results/hpc_round/ rather than over the existing directories. That is not
# tidiness: the old results are the evidence for how the numbers moved between platforms,
# and every findings doc that cites a superseded run is expected to say so and point at
# its replacement. Overwriting them would destroy the only record of the difference.
#
# EXP01 at 100 seeds dominates the runtime by a wide margin. It is first so that a mistake
# in the config shows up in the first few minutes rather than after everything else has
# run; each experiment prints its own elapsed time.
set -uo pipefail

OUT="${OUT:-results/hpc_round}"
SESSIONS="${SESSIONS:-200}"
SEEDS="${SEEDS:-15}"
SEEDS_SHARE="${SEEDS_SHARE:-100}"
PY="${PY:-python}"

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$OUT"

START=$(date +%s)

step() {
  local name="$1"; shift
  local t0 t1
  echo
  echo "=============================================================="
  echo "$name"
  echo "=============================================================="
  t0=$(date +%s)
  "$@" || { echo "FAILED: $name"; exit 1; }
  t1=$(date +%s)
  echo "-- $name took $(( t1 - t0 ))s"
}

# Sanity first. If the invariants do not hold, the constants are not the problem and
# nothing produced below would be worth reading.
step "invariants" "$PY" -m tests.test_invariants

# EXP01's headline share (73.6% from termination) is the 100-seed run. const_ttl is
# excluded deliberately: it is bit-identical to lru by construction and guarded by a test,
# so spending 100 seeds re-proving an identity buys nothing.
step "EXP01 share, ${SEEDS_SHARE} seeds" \
  "$PY" -m experiments.exp01_ttl_falsify \
    --sessions "$SESSIONS" --seeds "$SEEDS_SHARE" \
    --concurrency 8,10,12,14,16,18 --pause "" \
    --arms lru,ttl_oracle,oracle_terminal,belady \
    --out "$OUT/exp01_share"

step "EXP01 full arms, ${SEEDS} seeds" \
  "$PY" -m experiments.exp01_ttl_falsify \
    --sessions "$SESSIONS" --seeds "$SEEDS" --out "$OUT/exp01"

step "EXP02 pressure axis" \
  "$PY" -m experiments.exp02_pressure_axis \
    --sessions "$SESSIONS" --seeds "$SEEDS" --out "$OUT/exp02"

step "EXP03 pause isolation" \
  "$PY" -m experiments.exp03_pause_isolation \
    --sessions "$SESSIONS" --seeds "$SEEDS" --out "$OUT/exp03"

step "EXP04 predictor" \
  "$PY" -m experiments.exp04_predictor \
    --sessions "$SESSIONS" --seeds "$SEEDS" --out "$OUT/exp04"

step "EXP05 threshold" \
  "$PY" -m experiments.exp05_threshold \
    --sessions "$SESSIONS" --seeds "$SEEDS" --out "$OUT/exp05"

# Whether the seed counts still suffice is a property of the constants, not a constant of
# the project: changing the timing changes the cost figures and can change how much of the
# interval width is irreducible. Cheap to check, expensive to assume.
for exp in exp01 exp02 exp03; do
  step "seed sufficiency $exp" \
    "$PY" -m experiments.seed_sufficiency --results "$OUT/$exp" --exp "$exp"
done

for n in 01 02 03; do
  step "plot exp$n" "$PY" -m "experiments.plot_exp$n" --results "$OUT/exp$n" || true
done

echo
echo "=============================================================="
echo "total $(( $(date +%s) - START ))s"
echo "=============================================================="
echo "results in $OUT/"
echo
echo "Still to do, by hand:"
echo "  * update docs/*_findings.md and docs/SUMMARY.md -- every number in them is from"
echo "    the previous round, and each superseded run should say so and point here"
echo "  * commit the sim/config.py edit and these results together, so the platform"
echo "    switch is one attributable change"
echo "  * never plot the two rounds in one figure"
