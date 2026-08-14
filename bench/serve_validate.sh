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

# Both required under WSL2; see docs/calibration.md for why.
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "model       : $MODEL"
echo "prefix cache: ENABLED (this is the validation config, not the calibration one)"

nohup "$VENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --uvicorn-log-level warning \
  > "$LOG" 2>&1 &

echo "pid $! ; tail -f $LOG to watch startup"
