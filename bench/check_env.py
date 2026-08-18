"""Verify the GPU environment before any measurement is trusted.

Hard requirements for this project (RTX 5080, Blackwell):
  * compute capability (12, 0) -- if this is not what torch reports, the wheels were
    built for a different architecture and every kernel timing is meaningless;
  * torch >= 2.7.0 built against CUDA >= 12.8 -- earlier builds have no sm_120 kernels;
  * a prebuilt wheel, not a source build and not a nightly.

The project's written rule says "CUDA 12.8 wheel (cu128)". That was the *minimum* that
carries sm_120 kernels, so a newer toolkit satisfies the requirement it was protecting.
But newer is not free: a cu130 build needs a newer driver than a cu128 build, and the
HPC nodes are a different machine with a different driver. This script therefore reports
the CUDA build separately from the pass/fail, so a mismatch against the local wheel is
noticed before an HPC run fails on it rather than after.

Run inside WSL:
    ~/venv-vllm/bin/python bench/check_env.py
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys

# The local box is Blackwell (sm_120) and the project's written rule names it. The HPC
# nodes are Turing (7.5) or Ada (8.9), so this cannot be a hardcoded constant without the
# script failing on the machine it exists to check. Pass --expect-capability there.
# What must NEVER be silently accepted is a capability that differs from the one expected,
# because that is exactly how numbers from two platforms end up in one figure.
DEFAULT_CAPABILITY = (12, 0)
MIN_TORCH = (2, 7, 0)
MIN_CUDA = (12, 8)
# What the project's written rule names. Not a failure if the build is newer, but it
# must be surfaced: the wheel's CUDA version sets the minimum driver, and the HPC nodes
# will not necessarily have one that new.
PINNED_CUDA = (12, 8)


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for chunk in text.split("+")[0].split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _parse_capability(text: str) -> tuple[int, int]:
    major, _, minor = text.partition(".")
    return (int(major), int(minor))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expect-capability", default=None,
                    help="compute capability this run is supposed to be on, e.g. 7.5 for "
                         "a Turing HPC node, 8.9 for Ada, 12.0 for the local Blackwell "
                         "box (the default). A mismatch is a hard failure, because the "
                         "whole point is that numbers from two platforms must never be "
                         "mixed. Pass 'any' to record the capability without asserting it")
    ap.add_argument("--out", default=None,
                    help="also write the report as JSON here, so the platform a set of "
                         "results came from is recorded next to the results themselves")
    args = ap.parse_args(argv)

    if args.expect_capability in (None, ""):
        expected: tuple[int, int] | None = DEFAULT_CAPABILITY
    elif args.expect_capability.lower() == "any":
        expected = None
    else:
        expected = _parse_capability(args.expect_capability)

    report: dict = {"platform": platform.platform(), "python": sys.version.split()[0],
                    "hostname": socket.gethostname(),
                    "expected_capability": list(expected) if expected else "any"}
    problems: list[str] = []

    try:
        import torch
    except ImportError as exc:
        print(f"FAIL torch not importable: {exc}")
        return 1

    report["torch"] = torch.__version__
    report["torch_cuda_build"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()

    if _version_tuple(torch.__version__) < MIN_TORCH:
        problems.append(f"torch {torch.__version__} < {'.'.join(map(str, MIN_TORCH))}")
    if "dev" in torch.__version__ or "nightly" in torch.__version__:
        problems.append(f"torch {torch.__version__} looks like a nightly build")
    cuda_build = _version_tuple(torch.version.cuda) if torch.version.cuda else None
    if cuda_build is None or cuda_build < MIN_CUDA:
        problems.append(f"torch built against CUDA {torch.version.cuda}, "
                        f"need >= {'.'.join(map(str, MIN_CUDA))}")
    elif cuda_build[:2] != PINNED_CUDA:
        report["cuda_build_note"] = (
            f"CUDA {torch.version.cuda} wheel, project notes say cu"
            f"{PINNED_CUDA[0]}{PINNED_CUDA[1]}. Satisfies sm_120, but raises the minimum "
            f"driver version -- confirm the HPC nodes before pinning this for the final "
            f"reproduction runs.")

    if not torch.cuda.is_available():
        problems.append("torch.cuda.is_available() is False")
    else:
        cap = torch.cuda.get_device_capability()
        report["capability"] = list(cap)
        report["device"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        report["total_memory_gb"] = round(props.total_memory / 2**30, 2)
        if expected is not None and cap != expected:
            problems.append(f"compute capability {cap}, expected {tuple(expected)}. "
                            f"Either this is the wrong node, or --expect-capability is "
                            f"wrong; do not proceed until it is one of those two")

        # A capability number proves nothing on its own -- the wheel may simply lack
        # sm_120 kernels and fall back or crash. Force an actual matmul.
        try:
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
            b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            c = (a @ b).float().sum().item()
            report["fp16_matmul_ok"] = True
            report["fp16_matmul_checksum_finite"] = c == c  # NaN check
            if c != c:
                problems.append("fp16 matmul produced NaN")
        except Exception as exc:  # noqa: BLE001
            report["fp16_matmul_ok"] = False
            problems.append(f"fp16 matmul on device failed: {type(exc).__name__}: {exc}")

    try:
        import vllm
        report["vllm"] = vllm.__version__
    except ImportError:
        problems.append("vllm not importable")

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False)
        report["nvidia_smi"] = out.stdout.strip() or out.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report["nvidia_smi"] = f"unavailable: {exc}"

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    if "cuda_build_note" in report:
        print(f"\nNOTE  {report['cuda_build_note']}")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: environment satisfies the project's hardware requirements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
