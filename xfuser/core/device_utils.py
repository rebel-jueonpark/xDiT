"""Device-agnostic helpers that dispatch to the active accelerator backend.

xDiT historically called `torch.cuda.synchronize() / .empty_cache() / .Event(...)`
directly in several hot paths. To support non-CUDA backends (NPU, MUSA, RBLN, ...),
those call-sites should go through the helpers here.
"""
from __future__ import annotations

import time
from typing import Optional

import torch

from xfuser.envs import _is_cuda, _is_hip, _is_musa, _is_npu, _is_rbln


def synchronize(device: Optional[torch.device | int | str] = None) -> None:
    if _is_cuda() or _is_hip():
        torch.cuda.synchronize(device)
    elif _is_musa() and hasattr(torch, "musa"):
        torch.musa.synchronize(device)
    elif _is_npu() and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
    elif _is_rbln() and hasattr(torch, "rbln") and hasattr(torch.rbln, "synchronize"):
        torch.rbln.synchronize()


def empty_cache() -> None:
    if _is_cuda() or _is_hip():
        torch.cuda.empty_cache()
    elif _is_musa() and hasattr(torch, "musa") and hasattr(torch.musa, "empty_cache"):
        torch.musa.empty_cache()
    elif _is_npu() and hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()
    elif _is_rbln() and hasattr(torch, "rbln") and hasattr(torch.rbln, "empty_cache"):
        torch.rbln.empty_cache()


def memory_allocated(device: Optional[int] = None) -> int:
    if _is_cuda() or _is_hip():
        return torch.cuda.memory_allocated(device)
    return 0


class _WallClockEvent:
    """Fallback `torch.cuda.Event(enable_timing=True)` analogue for backends
    that don't expose CUDA-style events (RBLN today).
    """

    def __init__(self) -> None:
        self._t: float = 0.0

    def record(self) -> None:
        synchronize()
        self._t = time.perf_counter()

    def elapsed_time(self, other: "_WallClockEvent") -> float:
        return (other._t - self._t) * 1000.0  # ms, matching torch.cuda.Event semantics


def event(enable_timing: bool = True):
    if _is_cuda() or _is_hip():
        return torch.cuda.Event(enable_timing=enable_timing)
    if _is_musa() and hasattr(torch, "musa") and hasattr(torch.musa, "Event"):
        return torch.musa.Event(enable_timing=enable_timing)
    if _is_npu() and hasattr(torch, "npu") and hasattr(torch.npu, "Event"):
        return torch.npu.Event(enable_timing=enable_timing)
    return _WallClockEvent()
