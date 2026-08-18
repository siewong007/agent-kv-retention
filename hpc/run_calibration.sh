#!/usr/bin/env bash
# The calibration + validation round, on whatever node the scheduler gave us.
#
# Scheduler-agnostic on purpose: hpc/calibrate.sbatch is a thin Slurm wrapper around this,
# and a PBS or LSF site needs a new header, not a new script.
#
# WHY THIS EXISTS. Every engine constant in sim/config.py was fitted on a local RTX 5080
# under WSL2. docs/calibration.md already records that some of those numbers are
# WSL-specific and do not transfer -- no nvcc in the container inflates step_overhead_s,
# and a Windows desktop holding ~1.3 GiB changes the KV pool size. The project's first
# rule is that every number in the paper comes from one platform and one round. So either
# the thesis reports 5080/WSL2 numbers, or this script runs and the five experiments are
# re-run on its constants. It does not merge the two, and neither should anything else:
# output goes under results/hpc/ so a stray path cannot mix them.
#
# WHAT IT COSTS. Billing is wall clock, not GPU utilisation, so the only thing that
# matters is that the job exits the moment it stops measuring. It is a single batch job
# with no interactive step, the vLLM server is killed by an EXIT trap on every path
# including crashes, and the elapsed time and its cost are printed at the end. Budget
# about 2.5 hours; on the T4 list price of RM 3.06/h that is roughly RM 8.
#
# WHAT IT PRODUCES.
#   results/hpc/env.json              the platform, asserted rather than assumed
#   results/hpc/calib/                timing and batch fits
#   results/hpc/validate/             behaviour check at pressure ~0.64
#   results/hpc/validate_matched/     behaviour check above pressure 1.0
#   results/hpc/manifest.json         what ran, on what, for how long, at what cost
#
# AFTER IT FINISHES, the constants in results/hpc/calib/ are NOT automatically adopted.
# Copy them into sim/config.py deliberately, in a commit that says so, and re-run every
# experiment. A half-updated config is worse than either platform on its own.
set -uo pipefail

# --------------------------------------------------------------------- site settings
# These have no sensible defaults and the script refuses to guess. A job that silently
# ran on the wrong node or the wrong environment would produce numbers that look fine.
: "${PROJECT:?set PROJECT to the checkout path on the HPC filesystem}"
: "${VENV:?set VENV to the virtualenv holding torch and vllm}"
: "${EXPECT_CAPABILITY:?set EXPECT_CAPABILITY to the node compute capability, e.g. 7.5 for T4 Turing, 8.9 for L40S Ada}"
: "${RM_PER_HOUR:?set RM_PER_HOUR to the list price of the node you are on, e.g. 3.06 for T4}"

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
# A headless node has no desktop holding memory, so this can sit higher than the 0.85 the
# local box needs. The resulting pool is still read back from the log rather than assumed.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
OUT="${OUT:-results/hpc}"
SERVER_LOG="${SERVER_LOG:-$PWD/vllm_hpc_server.log}"

START_EPOCH=$(date +%s)

cd "$PROJECT" || { echo "PROJECT=$PROJECT is not a directory"; exit 1; }
mkdir -p "$OUT"

PY="$VENV/bin/python"
VLLM="$VENV/bin/vllm"
[ -x "$PY" ]   || { echo "no python at $PY"; exit 1; }
[ -x "$VLLM" ] || { echo "no vllm at $VLLM"; exit 1; }

# The expensive failure mode is a crash that leaves vLLM holding the node while the clock
# runs. Kill it on every exit path, not just the happy one.
cleanup() {
  rc=$?
  pkill -f "[v]llm serve" 2>/dev/null
  sleep 5
  elapsed=$(( $(date +%s) - START_EPOCH ))
  cost=$(awk -v s="$elapsed" -v r="$RM_PER_HOUR" 'BEGIN{printf "%.2f", s/3600*r}')
  hours=$(awk -v s="$elapsed" 'BEGIN{printf "%.2f", s/3600}')
  echo
  echo "=============================================================="
  echo "elapsed        : ${elapsed}s (${hours} h)"
  echo "billed at      : RM ${RM_PER_HOUR}/h"
  echo "estimated cost : RM ${cost}"
  echo "exit code      : ${rc}"
  echo "=============================================================="
  if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  # Poll the endpoint. Do NOT grep the log for a readiness string: it carries a benign
  # WARNING containing "Traceback (most recent call last):" from the deep_gemm CUDA_HOME
  # probe, which trips any keyword watcher and caused two false starts locally.
  i=0
  while [ "$i" -lt 180 ]; do
    if curl -sf -o /dev/null "http://localhost:$PORT/v1/models"; then
      echo "server ready after $(( i * 5 ))s"
      return 0
    fi
    if ! pgrep -f "[v]llm serve" > /dev/null; then
      echo "SERVER PROCESS DIED during startup"
      tail -40 "$SERVER_LOG"
      return 1
    fi
    sleep 5
    i=$(( i + 1 ))
  done
  echo "SERVER DID NOT BECOME READY within 900s"
  tail -40 "$SERVER_LOG"
  return 1
}

pool_blocks_from_log() {
  tokens=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$SERVER_LOG" | tail -1 \
           | grep -oE "[0-9,]+" | tr -d ',')
  [ -n "$tokens" ] || return 1
  echo $(( tokens / 16 ))
}

echo "=============================================================="
echo "step 1/5  environment"
echo "=============================================================="
# Hard-fails if the node is not the architecture this round is supposed to be on. That
# assertion is the reason the platform can later be quoted with any confidence at all.
"$PY" bench/check_env.py --expect-capability "$EXPECT_CAPABILITY" --out "$OUT/env.json" \
  || { echo "environment check failed; refusing to measure on an unverified node"; exit 1; }

echo
echo "=============================================================="
echo "step 2/5  calibration server (prefix caching OFF)"
echo "=============================================================="
# OFF is not a detail. fit_timing measures the cost of computing prefill; with prefix
# caching on it would measure the cost of NOT computing it, and the fit would be wrong in
# a way that still produces plausible-looking constants.
setsid "$VLLM" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --no-enable-prefix-caching \
  --uvicorn-log-level warning \
  > "$SERVER_LOG" 2>&1 &
wait_for_server || exit 1

"$PY" bench/read_server_config.py "$SERVER_LOG" | tee "$OUT/server_config_calib.txt"
"$PY" -m bench.fit_timing --base-url "http://localhost:$PORT" --out "$OUT/calib" || exit 1
"$PY" -m bench.fit_batch  --base-url "http://localhost:$PORT" --out "$OUT/calib" || exit 1

pkill -f "[v]llm serve"; sleep 10

echo
echo "=============================================================="
echo "step 3/5  validation server (prefix caching ON), pressure ~0.64"
echo "=============================================================="
setsid "$VLLM" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --uvicorn-log-level warning \
  > "$SERVER_LOG" 2>&1 &
wait_for_server || exit 1

POOL=$(pool_blocks_from_log) || { echo "could not read the KV pool from the log"; exit 1; }
echo "KV pool: $POOL blocks (read from this node's own log, not carried over)"

# Default admission. Nothing preempts at this pressure, so vLLM's per-scheduling counters
# equal a per-request hit rate and the comparison is exact.
"$PY" -m bench.validate_vs_vllm \
  --base-url "http://localhost:$PORT" \
  --sessions 40 --concurrency 10 \
  --pool-blocks "$POOL" \
  --out "$OUT/validate" || exit 1

pkill -f "[v]llm serve"; sleep 10

echo
echo "=============================================================="
echo "step 4/5  validation above pressure 1.0, matched admission"
echo "=============================================================="
# max_num_seqs must be passed to BOTH sides. Above pressure 1.0 vLLM otherwise
# over-commits, its prefix-cache counters inflate (1.694x locally, and the inflation turns
# out to be scheduling attempts rather than preemptions), and the ratio stops being a hit
# rate at all. Capping makes vLLM queue instead, which is what the simulator's whole-prompt
# admission already does, and the comparison becomes exact again. Capping also changes the
# KV pool, so it is re-read rather than reused.
setsid "$VLLM" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs 8 \
  --uvicorn-log-level warning \
  > "$SERVER_LOG" 2>&1 &
wait_for_server || exit 1

POOL_MATCHED=$(pool_blocks_from_log) || { echo "could not read the KV pool"; exit 1; }
echo "KV pool: $POOL_MATCHED blocks (differs from step 3 -- capping max_num_seqs reserves"
echo "more activation memory; locally this moved it 13663 -> 12865)"

"$PY" -m bench.validate_vs_vllm \
  --base-url "http://localhost:$PORT" \
  --sessions 60 --concurrency 16 --max-num-seqs 8 \
  --pool-blocks "$POOL_MATCHED" \
  --out "$OUT/validate_matched" || exit 1

pkill -f "[v]llm serve"; sleep 10

echo
echo "=============================================================="
echo "step 5/5  manifest"
echo "=============================================================="
ELAPSED=$(( $(date +%s) - START_EPOCH ))
"$PY" hpc/write_manifest.py \
  --out "$OUT" \
  --elapsed-s "$ELAPSED" \
  --rm-per-hour "$RM_PER_HOUR" \
  --model "$MODEL" \
  --pool-default "$POOL" \
  --pool-capped "$POOL_MATCHED" || exit 1

echo
echo "done. Next, and deliberately not automatic:"
echo "  1. diff $OUT/calib against the constants in sim/config.py"
echo "  2. if adopting, edit sim/config.py in its own commit and re-run all five experiments"
echo "  3. never plot the two rounds together"
