#!/usr/bin/env bash
# Map simulator pressure onto server pressure by sweeping how wide admission is.
#
# docs/validation_findings.md left exactly one thing open. Under MATCHED admission the
# simulator's cache agrees with vLLM to 1.4 pp, but at vLLM's default admission the two
# systems do not behave the same way at all: the simulator admits a request only if its
# whole prompt fits and therefore never preempts, while vLLM admits optimistically and
# re-scheduled 69% of prompt tokens at nominal pressure 1.02. So "pressure 1.0" does not
# mean the same thing on the two systems, and every headroom figure in this project is
# quoted on the simulator's axis with no known conversion to a real one.
#
# This sweeps max_num_seqs on BOTH sides at a fixed workload and a fixed pool, and reads
# off two things at each point: how far the hit rates diverge, and how much vLLM preempts.
# Preemption counts need no denominator, so the second one is valid even where the first
# is not.
#
# Two design points that are not incidental:
#
#   * THE POOL IS PINNED with --kv-cache-memory. Changing max_num_seqs changes how much
#     activation memory vLLM reserves, which changes the KV pool -- it went 13663 -> 12865
#     blocks between the default and 8. A sweep that let the pool move would confound
#     admission width with memory pressure, which is the one thing this is trying to
#     separate. The script asserts every run reported the same pool and stops if not.
#   * THE TOP POINT IS 16, NOT THE SERVER DEFAULT. The workload runs 16 concurrent
#     sessions, so at most 16 requests can ever be runnable and any cap at or above 16 is
#     unbounded for this workload. Using 16 makes the top of the sweep exactly matched on
#     both sides instead of depending on what vLLM's default happens to be in this version.
#
# Runs four servers back to back and shuts down at the end. Roughly 75 minutes on the
# local 5080. No HPC time.
#
#   bash bench/sweep_admission.sh
set -uo pipefail

VENV="${VENV:-$HOME/venv-vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PROJECT="${PROJECT:-/mnt/c/Users/Paul/Documents/deep-learning-project}"
LOG="$HOME/vllm_sweep_server.log"
SEQS="${SEQS:-6 8 12 16}"
SESSIONS="${SESSIONS:-60}"
CONCURRENCY="${CONCURRENCY:-16}"

# 6.0 GiB. Below what every max_num_seqs in the sweep could allocate on its own, so the
# pool is decided by this number rather than by the activation estimate.
#
# --gpu-memory-utilization must be passed as well, and low. Given --kv-cache-memory alone,
# vLLM back-computes the utilization it would need (0.92 = 14.65 GiB at 6.5 GiB of KV) and
# refuses to start, because a Windows desktop holds ~1.3 GiB and only 14.6 GiB is free.
# The two flags are not alternatives here: the utilization bounds the total, the explicit
# KV figure pins the pool inside it.
KV_BYTES="${KV_BYTES:-6442450944}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "$PROJECT" || exit 1
pool_seen=""

for n in $SEQS; do
  echo "=============================================================="
  echo "max_num_seqs = $n"
  echo "=============================================================="

  pkill -f "[v]llm serve" 2>/dev/null
  sleep 5

  setsid "$VENV/bin/vllm" serve "$MODEL" \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-memory "$KV_BYTES" \
    --max-num-seqs "$n" \
    --uvicorn-log-level warning \
    > "$LOG" 2>&1 &

  # Poll the endpoint rather than grepping the log for a readiness string: the log
  # contains a benign WARNING carrying the text "Traceback (most recent call last):"
  # from deep_gemm's CUDA_HOME probe, which trips any keyword-based watcher.
  ready=0
  for _ in $(seq 1 120); do
    if curl -sf -o /dev/null http://localhost:8000/v1/models; then ready=1; break; fi
    if ! pgrep -f "[v]llm serve" > /dev/null; then break; fi
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then
    echo "SERVER FAILED TO START at max_num_seqs=$n"
    tail -30 "$LOG"
    pkill -f "[v]llm serve" 2>/dev/null
    exit 1
  fi

  tokens=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | tail -1 \
           | grep -oE "[0-9,]+" | tr -d ',')
  blocks=$((tokens / 16))
  echo "pool: $tokens tokens -> $blocks blocks"

  if [ -z "$pool_seen" ]; then
    pool_seen="$blocks"
  elif [ "$blocks" != "$pool_seen" ]; then
    echo "POOL MOVED between sweep points ($pool_seen -> $blocks). The sweep would be"
    echo "confounding admission width with pool size, which is the whole point of"
    echo "pinning --kv-cache-memory. Stopping rather than producing a mixed curve."
    pkill -f "[v]llm serve" 2>/dev/null
    exit 1
  fi

  "$VENV/bin/python" -m bench.validate_vs_vllm \
    --sessions "$SESSIONS" \
    --concurrency "$CONCURRENCY" \
    --max-num-seqs "$n" \
    --pool-blocks "$blocks" \
    --out "results/sweep_admission/seqs_$n"
  rc=$?

  pkill -f "[v]llm serve" 2>/dev/null
  sleep 5
  if [ "$rc" -ne 0 ]; then
    echo "replay failed at max_num_seqs=$n (exit $rc)"
    exit "$rc"
  fi
done

pkill -f "[v]llm serve" 2>/dev/null
sleep 5
echo
echo "sweep complete; GPU released:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "analyse with: python -m bench.analyze_admission_sweep"
