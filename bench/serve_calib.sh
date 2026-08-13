#!/usr/bin/env bash
# Start vLLM in the configuration used for timing calibration, and only that.
#
# Prefix caching is OFF on purpose. The constants being fitted describe what a cache
# MISS costs; leaving the cache on would let hits shorten the very prefills being timed
# and the fitted prefill_tps would come out too high.
#
# The startup log is kept because it is the only trustworthy source for the KV pool
# size. docs/calibration.md derives ~16000 blocks from hardware specs; that derivation
# is a sanity check on this log line, not a substitute for it.
#
#   bash bench/serve_calib.sh [PORT] [LOGFILE]
#
# Stop it when the measurement is done. A vLLM server left running holds the GPU.

set -euo pipefail

PORT="${1:-8000}"
LOG="${2:-$HOME/vllm_calib_server.log}"
VENV="${VENV:-$HOME/venv-vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

# REQUIRED under WSL2. vLLM's V1 worker allocates UVA buffers, whose availability check
# is just "is pinned memory available". On WSL2 vLLM gates pinned memory behind this
# env var: supported on kernels >= 4.19.121 but off by default. Without it the engine
# dies at startup with "RuntimeError: UVA is not available", which reads like a hardware
# limitation and is not one -- pinned allocation and async H2D copies both work here
# (verified by bench/probe_uva.py on kernel 6.18.33.2).
#
# Caveat worth carrying into docs/calibration.md: vLLM ships this off by default under
# WSL for a reason, so host-device transfer costs measured here may not match a native
# Linux node. It affects scheduling overhead, not the prefill/decode kernel timings the
# calibration is after, but it is one more reason local numbers stay local.
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"

# FlashInfer JIT-compiles its sampling kernels on first use and needs nvcc, which the
# torch wheel does not ship (it carries the CUDA runtime, not the toolkit). Rather than
# install a 3 GB toolkit to satisfy a component we do not measure, fall back to the
# PyTorch-native sampler. Calibration runs at temperature 0, so the sampler contributes
# a fixed sub-millisecond cost to every step and cannot distort the prefill throughput
# or per-KV-token decode slope being fitted -- but it does land inside the fitted
# step_overhead_s, so a machine WITH FlashInfer will have a slightly smaller one.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# FlashAttention 3 has been unreliable on Blackwell (sm_120). If the server fails to
# start with an attention-backend error, re-run with VLLM_FLASH_ATTN_VERSION=2 exported.
: "${VLLM_FLASH_ATTN_VERSION:=}"
if [[ -n "$VLLM_FLASH_ATTN_VERSION" ]]; then
  export VLLM_FLASH_ATTN_VERSION
  echo "using VLLM_FLASH_ATTN_VERSION=$VLLM_FLASH_ATTN_VERSION"
fi

echo "model      : $MODEL"
echo "port       : $PORT"
echo "log        : $LOG"
echo "prefix cache: DISABLED (calibration measures the cost of a miss)"

# gpu_memory_utilization is set explicitly and low. On a Windows desktop GPU the display
# output holds ~1.3 GiB permanently, so only ~14.6 of the 15.92 GiB are free at startup
# and vLLM 0.26's default of 0.92 cannot be satisfied. This is a property of running on
# a desktop card, not of the model -- a headless HPC node will have the full amount.
# That difference lands directly in the KV pool size, and therefore in every pressure
# ratio, which is one more reason local and HPC numbers must never share a figure.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# --disable-log-requests was removed in vLLM 0.26; quiet the access log instead.
nohup "$VENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --no-enable-prefix-caching \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --uvicorn-log-level warning \
  > "$LOG" 2>&1 &

echo "pid $! ; tail -f $LOG to watch startup"
