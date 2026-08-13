"""Find out whether UVA is genuinely unavailable under WSL2, or only reported so.

ANSWER, for vLLM 0.26 on WSL2 kernel 6.18.33.2: only reported so. Pinned allocation and
async host-to-device copies both work. vLLM's `is_uva_available()` is just
`is_pin_memory_available()`, and the CUDA platform gates that behind an opt-in env var
under WSL:

    export VLLM_WSL2_ENABLE_PIN_MEMORY=1

Without it the engine dies with "RuntimeError: UVA is not available", which reads like a
hardware limitation and is not one. bench/serve_calib.sh sets it.

This script is kept as the diagnostic that establishes the distinction: it separates
"the capability is missing" from "the library declined to use it", which lead to
completely different responses -- abandoning the machine versus setting one variable.
"""

from __future__ import annotations

import json
import traceback

import torch


def main() -> int:
    report: dict = {}

    try:
        from vllm.utils.platform_utils import is_uva_available
        import inspect
        report["vllm_is_uva_available"] = bool(is_uva_available())
        report["vllm_check_source"] = inspect.getsource(is_uva_available).strip()
    except Exception as exc:  # noqa: BLE001
        report["vllm_is_uva_available"] = f"error: {exc}"

    # 1. plain pinned host allocation
    try:
        t = torch.zeros(1024, dtype=torch.int32, pin_memory=True)
        report["pinned_alloc_ok"] = bool(t.is_pinned())
    except Exception as exc:  # noqa: BLE001
        report["pinned_alloc_ok"] = False
        report["pinned_alloc_error"] = f"{type(exc).__name__}: {exc}"

    # 2. async host->device copy from pinned memory (what the fast path actually needs)
    try:
        host = torch.arange(4096, dtype=torch.int32, pin_memory=True)
        dev = torch.empty_like(host, device="cuda")
        dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        report["pinned_async_copy_ok"] = bool(int(dev[-1].item()) == 4095)
    except Exception as exc:  # noqa: BLE001
        report["pinned_async_copy_ok"] = False
        report["pinned_async_copy_error"] = f"{type(exc).__name__}: {exc}"

    # 3. the env var that actually decides it, and what vLLM's platform layer says
    try:
        import vllm.envs as envs
        report["VLLM_WSL2_ENABLE_PIN_MEMORY"] = getattr(
            envs, "VLLM_WSL2_ENABLE_PIN_MEMORY", "attribute missing")
        from vllm.platforms import current_platform
        report["platform_is_pin_memory_available"] = bool(
            current_platform.is_pin_memory_available())
    except Exception as exc:  # noqa: BLE001
        report["platform_probe_error"] = f"{type(exc).__name__}: {exc}"
        report["platform_probe_traceback"] = traceback.format_exc().splitlines()[-3:]

    # 4. cudaHostRegister-style path via torch's own attribute, if exposed
    try:
        props = torch.cuda.get_device_properties(0)
        report["unified_addressing_attr"] = getattr(props, "unified_addressing", "n/a")
        report["device"] = props.name
    except Exception as exc:  # noqa: BLE001
        report["unified_addressing_attr"] = f"error: {exc}"

    print(json.dumps(report, indent=2))

    hardware_ok = report.get("pinned_alloc_ok") and report.get("pinned_async_copy_ok")
    library_ok = report.get("platform_is_pin_memory_available")

    if not hardware_ok:
        print("\nVERDICT: pinned host memory does not work on this setup at all. "
              "No vLLM version will run here; the machine is the problem.")
    elif hardware_ok and not library_ok:
        print("\nVERDICT: the capability is present, the library declined to use it.\n"
              "Pinned allocation and async H2D copies both succeeded, but vLLM's "
              "platform layer reports pin memory unavailable.\nUnder WSL this is opt-in:"
              "\n    export VLLM_WSL2_ENABLE_PIN_MEMORY=1\n"
              "Do NOT conclude that WSL2 cannot run vLLM from the 'UVA is not "
              "available' message alone.")
    else:
        print("\nVERDICT: pinned memory available and enabled. UVA buffers will "
              "allocate; this is not the cause of any startup failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
