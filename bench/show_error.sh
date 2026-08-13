#!/usr/bin/env bash
# Print the first real exception from a vLLM server log.
#
# vLLM re-raises the engine-core failure in the API server process, so the tail of the
# log is always the same useless "Engine core initialization failed". The cause is the
# FIRST exception, several hundred lines earlier, in the EngineCore process.
# No pipefail here: `head` closing a pipe early makes grep die of SIGPIPE, and with
# pipefail that kills this script before it prints anything useful.
set -u
LOG="${1:-$HOME/vllm_calib_server.log}"

echo "=== exception messages, in order ==="
# The engine's own config dump contains the substring "Error" inside option names and
# is thousands of characters long; drop it or it buries the actual exception.
grep -aE '^\(EngineCore' "$LOG" | grep -av 'Initializing a V1 LLM engine' \
  | grep -aE '(Error|Exception|assert)' \
  | sed 's/^.*core\.py:1330\] //' | cut -c1-200 | sort -u | head -20

echo
echo "=== deepest frames of the EngineCore traceback ==="
grep -aE '^\(EngineCore' "$LOG" | grep -aE 'File "|, in ' \
  | sed 's/^.*core\.py:1330\] //' | tail -12

echo
echo "=== warnings mentioning memory, backend, attention or graph ==="
grep -aiE 'warn.*(memory|backend|attention|flash|graph|compil)' "$LOG" \
  | grep -av 'Initializing a V1 LLM engine' | cut -c1-200 | tail -8
