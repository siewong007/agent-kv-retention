#!/usr/bin/env bash
# Paired seeds at pressure 1.27, at two admission widths.
#
# Everything measured against vLLM so far is one seed per point, and the project has
# already had to retract two single-seed claims. docs/validation_findings.md reports the
# simulator 4.8 pp pessimistic at max_num_seqs=8 and 10.9 pp at 6, and calls that a
# measured but not established disagreement. This puts an interval on it.
#
# The informative quantity is the DIFFERENCE between the two caps, not either one alone.
# A single effective-pressure offset would have to produce the same gap at both, since
# both run at the same pressure on the same pinned pool; that they differ by 2.3x on one
# seed is what ruled out the dull explanation (bench/pressure_sensitivity.py). Whether
# that 2.3x survives seeds is the open question.
#
# Conditions are held identical to results/sweep_admission/ so seed 0 from that round is
# a fourth data point rather than a near-miss:
#   * the pool is pinned with --kv-cache-memory 6.0 GiB, which lands at 10922 blocks;
#     --gpu-memory-utilization must be passed too, or vLLM back-computes a utilization it
#     cannot satisfy and refuses to start;
#   * 60 sessions at concurrency 16;
#   * max_num_seqs passed to BOTH sides, so vLLM queues instead of over-committing and
#     its /metrics counters stay a per-request hit rate.
#
# The server is restarted between seeds. Reusing it would leave the previous seed's blocks
# resident, so the server would begin warm while the simulator begins cold -- a small
# effect, but this whole exercise exists because measurement-apparatus contamination has
# twice produced findings that had to be withdrawn.
#
# About 2.2 hours on the local 5080 for 6 runs. No HPC time.
#
#   bash bench/seeds_at_pressure.sh
set -uo pipefail

VENV="${VENV:-$HOME/venv-vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PROJECT="${PROJECT:-/mnt/c/Users/Paul/Documents/deep-learning-project}"
LOG="$HOME/vllm_seeds_server.log"
CAPS="${CAPS:-6 8}"
SEEDS="${SEEDS:-1 2 3}"
SESSIONS="${SESSIONS:-60}"
CONCURRENCY="${CONCURRENCY:-16}"
KV_BYTES="${KV_BYTES:-6442450944}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
EXPECTED_POOL="${EXPECTED_POOL:-10922}"

export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "$PROJECT" || exit 1
START=$(date +%s)

cleanup() {
  rc=$?
  pkill -f "[v]llm serve" 2>/dev/null
  sleep 5
  echo
  echo "elapsed $(( $(date +%s) - START ))s, exit $rc"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null
}
trap cleanup EXIT INT TERM

for cap in $CAPS; do
  for seed in $SEEDS; do
    echo "=============================================================="
    echo "max_num_seqs=$cap  seed=$seed"
    echo "=============================================================="

    pkill -f "[v]llm serve" 2>/dev/null
    sleep 8

    setsid "$VENV/bin/vllm" serve "$MODEL" \
      --port 8000 \
      --max-model-len 32768 \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --kv-cache-memory "$KV_BYTES" \
      --max-num-seqs "$cap" \
      --uvicorn-log-level warning \
      > "$LOG" 2>&1 &

    ready=0
    i=0
    while [ "$i" -lt 120 ]; do
      if curl -sf -o /dev/null http://localhost:8000/v1/models; then ready=1; break; fi
      pgrep -f "[v]llm serve" > /dev/null || break
      sleep 5
      i=$(( i + 1 ))
    done
    if [ "$ready" -ne 1 ]; then
      echo "SERVER FAILED TO START (cap=$cap seed=$seed)"
      tail -30 "$LOG"
      exit 1
    fi

    tokens=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | tail -1 \
             | grep -oE "[0-9,]+" | tr -d ',')
    blocks=$(( tokens / 16 ))
    if [ "$blocks" != "$EXPECTED_POOL" ]; then
      echo "POOL IS $blocks, EXPECTED $EXPECTED_POOL. These seeds would not be comparable"
      echo "to results/sweep_admission/, which is the whole point of pinning it. Stopping."
      exit 1
    fi
    echo "pool: $blocks blocks (matches the sweep round)"

    "$VENV/bin/python" -m bench.validate_vs_vllm \
      --sessions "$SESSIONS" \
      --concurrency "$CONCURRENCY" \
      --max-num-seqs "$cap" \
      --seed "$seed" \
      --pool-blocks "$blocks" \
      --out "results/seeds_1p27/cap${cap}_seed${seed}"
    rc=$?

    pkill -f "[v]llm serve" 2>/dev/null
    sleep 5
    [ "$rc" -eq 0 ] || { echo "replay failed (cap=$cap seed=$seed)"; exit "$rc"; }
  done
done

echo
echo "all runs complete"
echo "analyse with: python -m bench.analyze_seed_pairs"
