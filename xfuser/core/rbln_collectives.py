"""RBLN-only fallbacks for collective ops not yet exposed by `rbln-ccl`.

`rbln-ccl` (torch_rbln 0.2.x) implements allreduce, allgather, broadcast,
send/recv but not `alltoall_base`. xDiT / DeepSpeed-Ulysses sequence
parallelism calls `dist.all_to_all_single` at every attention layer.

This module installs a Python-level fallback for `all_to_all_single` and the
functional `_functional_collectives.all_to_all_single` that decomposes the
all-to-all into allgather + index_select. Bandwidth is `world_size×` of a
native alltoall, which is acceptable for bringup; a native rbln alltoall
should replace this when available.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def _all_to_all_single_via_all_gather(output, input_tensor, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    # Only the uniform-split path is needed for xDiT / Ulysses.
    if output_split_sizes is not None or input_split_sizes is not None:
        raise NotImplementedError(
            "rbln-ccl all_to_all_single polyfill: non-uniform split sizes are not yet supported"
        )

    assert input_tensor.size(0) % world_size == 0, (
        f"all_to_all_single requires dim-0 ({input_tensor.size(0)}) divisible by world_size ({world_size})"
    )

    # 1) Drain pending NPU compute, then allgather. rbln-ccl seems to require
    # the device to be quiescent before collectives (raw allgather inside a
    # busy model forward errors with code 1, while the same op in isolation
    # succeeds). empty_cache is the only public hook torch_rbln 0.2.x gives us
    # that forces a barrier.
    if hasattr(torch, "rbln") and hasattr(torch.rbln, "empty_cache"):
        torch.rbln.empty_cache()
    src_input = input_tensor.detach().clone().contiguous()
    gathered_list = [torch.empty_like(src_input) for _ in range(world_size)]
    dist.all_gather(gathered_list, src_input, group=group)

    # 2) on each rank r, output chunk i = gathered[i] split into `world_size` along dim 0, pick block r
    per_rank_rows = input_tensor.size(0) // world_size
    chunks = []
    for src in range(world_size):
        start = rank * per_rank_rows
        end = start + per_rank_rows
        chunks.append(gathered_list[src][start:end])
    result = torch.cat(chunks, dim=0)
    output.copy_(result)

    class _NullWork:
        def wait(self):
            return None
        def is_completed(self):
            return True
        def get_future(self):
            fut = torch.futures.Future()
            fut.set_result(output)
            return fut
    return _NullWork() if async_op else None


def _functional_all_to_all_single(input_tensor, output_split_sizes=None, input_split_sizes=None, group=None):
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    output = torch.empty_like(input_tensor)
    _all_to_all_single_via_all_gather(output, input_tensor, output_split_sizes, input_split_sizes, group, async_op=False)
    return output


_installed = False


def install_rbln_collective_polyfills() -> None:
    """Monkey-patch `dist.all_to_all_single` and the functional variant for rbln-ccl.

    The patched function checks the backend at call time so that other process
    groups (e.g. CPU gloo subgroups xDiT spawns for control flow) keep their
    native behavior.
    """
    global _installed
    if _installed:
        return
    _installed = True

    _original_all_to_all_single = dist.all_to_all_single

    def _patched_all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
        pg = group if group is not None else dist.distributed_c10d._get_default_group()
        try:
            backend = pg.name().lower()
        except Exception:
            backend = ""
        if backend in {"rbln-ccl", "rbln_ccl"}:
            return _all_to_all_single_via_all_gather(
                output, input, output_split_sizes, input_split_sizes, group, async_op
            )
        return _original_all_to_all_single(output, input, output_split_sizes, input_split_sizes, group=group, async_op=async_op)

    dist.all_to_all_single = _patched_all_to_all_single

    # Also patch the functional-collectives variant used by xDiT's usp.py
    try:
        from torch.distributed import _functional_collectives as ft_c

        _orig_ft_all_to_all = ft_c.all_to_all_single

        def _patched_ft_all_to_all(input, output_split_sizes=None, input_split_sizes=None, group=None):
            pg = group if group is not None else dist.distributed_c10d._get_default_group()
            try:
                backend = pg.name().lower() if hasattr(pg, "name") else ""
            except Exception:
                backend = ""
            # Resolve group-like wrappers used by _functional_collectives
            if isinstance(group, (list, tuple)) and len(group) > 0 and hasattr(group[0], "name"):
                try:
                    backend = group[0].name().lower()
                except Exception:
                    pass
            if backend in {"rbln-ccl", "rbln_ccl"}:
                return _functional_all_to_all_single(input, output_split_sizes, input_split_sizes, group)
            return _orig_ft_all_to_all(input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group)

        ft_c.all_to_all_single = _patched_ft_all_to_all
    except (ImportError, AttributeError):
        pass
