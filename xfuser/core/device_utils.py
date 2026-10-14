"""Accelerator-neutral device helpers.

xDiT grew up on CUDA and called ``torch.cuda.synchronize`` / ``empty_cache`` /
``Event`` directly in a few hot paths. Those call sites go through here so a
backend without a ``torch.cuda`` namespace (MUSA, Ascend NPU, Rebellions RBLN)
runs the same code. Every helper is a no-op on a backend that lacks the
facility rather than an error: these are conveniences around timing and
memory, never correctness.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from xfuser.envs import _is_cuda, _is_hip, _is_musa, _is_npu, _is_rbln, get_device_name


def _device_module():
    """The ``torch.<backend>`` module of the active accelerator, or None on cpu."""
    if _is_cuda() or _is_hip():
        return torch.cuda
    if _is_musa():
        return getattr(torch, "musa", None)
    if _is_npu():
        return getattr(torch, "npu", None)
    if _is_rbln():
        return getattr(torch, "rbln", None)
    return None


def synchronize(device: Optional[torch.device | int | str] = None) -> None:
    mod = _device_module()
    if mod is None or not hasattr(mod, "synchronize"):
        return
    if device is None:
        mod.synchronize()
    else:
        mod.synchronize(device)


def empty_cache() -> None:
    mod = _device_module()
    if mod is not None and hasattr(mod, "empty_cache"):
        mod.empty_cache()


def memory_allocated(device: Optional[int] = None) -> int:
    mod = _device_module()
    if mod is None or not hasattr(mod, "memory_allocated"):
        return 0
    try:
        return int(mod.memory_allocated(device))
    except Exception:  # noqa: BLE001 - reporting only
        return 0


def local_device(local_rank: int) -> torch.device:
    """This rank's accelerator (``cuda:N``, ``rbln:N``, ...) or cpu."""
    name = get_device_name()
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps")
    return torch.device(name, local_rank)


class _WallClockEvent:
    """``torch.cuda.Event(enable_timing=True)`` stand-in for backends without event timing.

    ``record`` synchronizes the device first so the wall-clock stamp marks the
    point where all previously queued work has completed, which is what the
    CUDA event pair measures once both are synchronized.
    """

    def __init__(self) -> None:
        self._t: float = 0.0

    def record(self, stream=None) -> None:  # noqa: ARG002 - signature parity
        synchronize()
        self._t = time.perf_counter()

    def synchronize(self) -> None:
        synchronize()

    def elapsed_time(self, other: "_WallClockEvent") -> float:
        return (other._t - self._t) * 1000.0  # milliseconds, like torch.cuda.Event


def event(enable_timing: bool = True):
    """A timing event for the active accelerator.

    CUDA/HIP, MUSA and Ascend expose CUDA-style events. torch-rbln has an
    ``Event`` type but no ``elapsed_time``, so it gets the wall-clock stand-in,
    as does cpu.
    """
    if _is_cuda() or _is_hip():
        return torch.cuda.Event(enable_timing=enable_timing)
    if _is_musa() and hasattr(torch, "musa") and hasattr(torch.musa, "Event"):
        return torch.musa.Event(enable_timing=enable_timing)
    if _is_npu() and hasattr(torch, "npu") and hasattr(torch.npu, "Event"):
        return torch.npu.Event(enable_timing=enable_timing)
    return _WallClockEvent()
