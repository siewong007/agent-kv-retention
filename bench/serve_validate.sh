#!/usr/bin/env bash
# Start vLLM the way the VALIDATION needs it: prefix caching ON.
#
# Deliberately a separate script from serve_calib.sh, which turns prefix caching OFF.
# The two experiments need opposite settings and confusing them would silently produce
# a hit rate of zero (calibration config) or a contaminated timing fit (this config),
# in both cases without any error.
set -euo pipefail

PORT="${1:-8000}"
LOG="${2:-$HOME/vllm_validate_server.log}"
VENV="${VENV:-$HOME/venv-vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Cap on concurrently RUNNING sequences. Empty leaves the server's default. Set it to
# match bench/validate_vs_vllm.py --max-num-seqs when validating at pressure above 1.0:
# vLLM then queues instead of over-committing, which is what the simulator's whole-prompt
# admission already does. Without it, vLLM preempts, its /metrics prefix-cache counters
# are inflated by re-queries, and they cannot be compared to the simulator at all.
# NOTE: changing it also changes the KV pool size, because more activation memory is
# reserved -- re-read the pool with bench/read_server_config.py instead of reusing the
# previous number. At 8 it went 13663 -> 12865 blocks.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

# Both required under WSL2; see docs/calibration.md for why.
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "model       : $MODEL"
echo "prefix cache: ENABLED (this is the validation config, not the calibration one)"
echo "max_num_seqs: ${MAX_NUM_SEQS:-server default}"

nohup "$VENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  ${MAX_NUM_SEQS:+--max-num-seqs "$MAX_NUM_SEQS"} \
  --uvicorn-log-level warning \
  > "$LOG" 2>&1 &

echo "pid $! ; tail -f $LOG to watch startup"
