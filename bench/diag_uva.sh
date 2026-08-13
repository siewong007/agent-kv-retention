#!/usr/bin/env bash
# Pull the traceback that leads to the UVA failure out of a vLLM server log.
#
# "UVA is not available" is a WSL2 limitation, not a misconfiguration: the WSL GPU
# paravirtualisation layer does not expose unified virtual addressing, so any vLLM code
# path that wants host-mapped device memory fails at startup. Knowing WHICH path asks
# for it decides whether a flag can avoid it or whether this machine cannot run the
# server at all.
set -euo pipefail
LOG="${1:-$HOME/vllm_calib_server.log}"
grep -n "UVA" -B 30 "$LOG" | grep -E 'File "|, in |UVA|Error' | tail -30
