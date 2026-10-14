"""Per-stage wall-clock timers for a diffusion pipeline run.

Used by the runner when ``XDIT_STAGE_TIMING=1``. Each wrapped call synchronizes the
accelerator before and after, so the stamps bound the work of that stage and nothing
else; the price is lost overlap, which is fine for a benchmark.
"""

from __future__ import annotations

import json
import os
import time

from xfuser.core import device_utils


class StageTimer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.totals: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def _start(self, bucket: str) -> None:
        device_utils.synchronize()
        self._t0 = time.perf_counter()

    def _stop(self, bucket: str) -> None:
        device_utils.synchronize()
        self.totals[bucket] = self.totals.get(bucket, 0.0) + (time.perf_counter() - self._t0)
        self.calls[bucket] = self.calls.get(bucket, 0) + 1

    def wrap(self, obj, attr: str, bucket: str, count: bool = False) -> None:
        """Time ``obj.attr`` calls. An ``nn.Module``'s ``forward`` is timed through forward
        hooks: replacing a module's ``forward`` attribute would make Dynamo treat the module
        as dynamic (its weights become graph inputs), which the compiled-for-NPU transformer
        must not become."""
        import torch

        if isinstance(obj, torch.nn.Module) and attr == "forward":
            if getattr(obj, "_xfuser_stage_timed", False):
                return
            obj.register_forward_pre_hook(lambda m, a, k: self._start(bucket), with_kwargs=True)
            obj.register_forward_hook(lambda m, a, k, o: self._stop(bucket), with_kwargs=True)
            obj._xfuser_stage_timed = True
            return

        fn = getattr(obj, attr)
        if getattr(fn, "_xfuser_stage_timed", False):
            return

        def timed(*args, **kwargs):
            self._start(bucket)
            out = fn(*args, **kwargs)
            self._stop(bucket)
            return out

        timed._xfuser_stage_timed = True
        setattr(obj, attr, timed)

    def summary(self, total_s: float) -> str:
        parts = [f"total={total_s:.3f}s"]
        for name in ("encode", "denoise", "decode"):
            if name in self.totals:
                extra = f" ({self.calls[name]} calls, {self.totals[name] / self.calls[name]:.3f}s/call)" if name == "denoise" else ""
                parts.append(f"{name}={self.totals[name]:.3f}s{extra}")
        return " ".join(parts)

    def record(self, total_s: float, output_directory: str, rank: int) -> None:
        """Append one JSON line per run; the benchmark driver aggregates them."""
        if not output_directory:
            return
        os.makedirs(output_directory, exist_ok=True)
        rec = {"rank": rank, "total_s": total_s}
        for name, value in self.totals.items():
            rec[f"{name}_s"] = value
            rec[f"{name}_calls"] = self.calls[name]
        with open(os.path.join(output_directory, f"stage_timing_rank{rank}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
