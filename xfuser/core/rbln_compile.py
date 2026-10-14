"""torch.compile through the rebel compiler for Rebellions NPUs (the ``rbln`` backend).

One place builds the compile options so the legacy pipeline wrappers and the runner
agree on them: static shapes, strict mode (a compile failure raises instead of silently
running the graph on the host), a persistent artifact cache, this rank's device, and the
process-group map the compiler needs to lower in-graph collectives onto rbln-ccl.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from xfuser.core.distributed.parallel_state import (
    get_cfg_group,
    get_dp_group,
    get_pp_group,
    get_sp_group,
    get_tp_group,
    get_world_group,
)
from xfuser.logger import init_logger

logger = init_logger(__name__)


def default_cache_dir() -> str:
    return os.environ.get("XDIT_RBLN_CACHE_DIR") or os.path.expanduser("~/.cache/xfuser/rbln_compile_cache")


def process_group_dict() -> dict:
    """c10d process-group name -> global ranks, for every parallel group this rank is in.

    The rebel compiler resolves an in-graph collective's group by its c10d name at trace
    time and the runtime builds the matching rbln-ccl communicator from this map (the
    contract vllm-rbln uses). Ulysses and ring groups are sub-groups of the sequence
    parallel group that yunchang creates, so they are listed too.
    """
    groups: dict = {}
    for getter in (get_tp_group, get_sp_group, get_pp_group, get_dp_group, get_cfg_group):
        try:
            coordinator = getter()
        except Exception:  # noqa: BLE001 - group not initialized for this run
            continue
        if coordinator is None:
            continue
        for attr in ("device_group", "cpu_group", "ulysses_group", "ring_group"):
            pg = getattr(coordinator, attr, None)
            name = getattr(pg, "group_name", None) if pg is not None else None
            if not name:
                continue
            try:
                ranks = list(torch.distributed.get_process_group_ranks(pg))
            except Exception:  # noqa: BLE001
                ranks = list(coordinator.ranks)
            if len(ranks) < 2:
                # A degree-1 group never carries a collective, and its c10d name differs
                # per rank (each rank kept its own singleton), so it has no place in a map
                # every rank must agree on.
                continue
            groups[name] = ranks
    return groups


def _runtime_device(local_rank: int):
    """The device(s) a program for this rank's logical ``rbln:N`` runs on.

    With ``RBLN_NPUS_PER_DEVICE`` / ``RBLN_DEVICE_MAP`` a logical device spans several
    physical NPUs and the compiler shards the program across them (RSD); the runtime then
    needs that physical list, a bare logical index names one NPU.
    """
    try:
        from torch_rbln._internal.rsd_utils import get_physical_device_ids

        physical = get_physical_device_ids(local_rank)
    except Exception:  # noqa: BLE001 - older torch-rbln without the helper
        physical = None
    if physical and len(physical) > 1:
        return list(physical)
    return local_rank


def compile_options(cache_dir: Optional[str] = None) -> dict:
    world = get_world_group()
    options = {
        "mode": ["strict"],
        "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe,
        "cache_dir": cache_dir or default_cache_dir(),
        "device": _runtime_device(world.local_rank),
    }
    if world.world_size > 1:
        options["process_group_dict"] = process_group_dict()
    trace_method = os.environ.get("XDIT_RBLN_TRACE_METHOD")  # export | export_strict | jittrace
    if trace_method:
        options["model_trace_method"] = trace_method
    return options


def _contiguous(obj):
    """Device inputs of a compiled program must be contiguous and must start at the base
    of their allocation: the rbln runtime resolves an input by the device key of its
    allocation, so a view at an offset (``timesteps[i].expand(1)`` is one, 4 bytes in at
    step 2) is an invalid key. diffusers also hands over ``timestep.expand(batch)`` with
    stride 0 and the odd transposed view. Views are copied into their own buffer."""
    if isinstance(obj, torch.Tensor):
        if obj.device.type != "rbln":
            return obj
        if not obj.is_contiguous() or obj._base is not None or obj.storage_offset() != 0:
            return obj.clone(memory_format=torch.contiguous_format)
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_contiguous(o) for o in obj)
    if isinstance(obj, dict):
        return type(obj)((k, _contiguous(v)) for k, v in obj.items())
    return obj


_DEBUG_INPUTS = os.environ.get("XDIT_RBLN_DEBUG_INPUTS", "0") == "1"


def _describe(name, t):
    if not isinstance(t, torch.Tensor):
        return f"{name}={type(t).__name__}"
    base = "view" if t._base is not None else "own"
    return f"{name}: {tuple(t.shape)} {t.dtype} {t.device} {base} contig={t.is_contiguous()} ptr={t.data_ptr():#x}"


def _contiguous_inputs_hook(module, args, kwargs):
    args, kwargs = _contiguous(args), _contiguous(kwargs)
    if _DEBUG_INPUTS:
        calls = getattr(module, "_xfuser_rbln_calls", 0)
        if calls < 3:
            lines = [_describe(f"arg{i}", a) for i, a in enumerate(args)]
            lines += [_describe(k, v) for k, v in kwargs.items()]
            logger.info(f"rbln compiled call #{calls} of {type(module).__name__}: " + " | ".join(lines))
        module._xfuser_rbln_calls = calls + 1
    return args, kwargs


def compile_forward(module: torch.nn.Module, cache_dir: Optional[str] = None) -> torch.nn.Module:
    """Compile ``module.forward`` in place for the NPU. Idempotent.

    Exactly ``module.forward = torch.compile(module.forward)``: the rebel backend
    recognises that form and keeps the module specialized (its weights stay graph
    constants). Input fix-ups therefore go in a pre-hook, not in a wrapper.
    """
    if getattr(module, "_xfuser_rbln_compiled", False):
        return module
    options = compile_options(cache_dir)
    logger.info(
        f"rbln: compiling {type(module).__name__}.forward with the rebel compiler "
        f"(groups={list(options.get('process_group_dict', {}).keys())}, cache_dir={options['cache_dir']})"
    )
    module.register_forward_pre_hook(_contiguous_inputs_hook, with_kwargs=True, prepend=True)
    module.forward = torch.compile(module.forward, backend="rbln", dynamic=False, options=options)
    module._xfuser_rbln_compiled = True
    return module


def compile_vae_decode(vae: torch.nn.Module, cache_dir: Optional[str] = None) -> None:
    """Compile the VAE's decode path for the NPU and route ``vae.decode`` through it.

    diffusers' ``AutoencoderKL._decode`` is ``post_quant_conv`` then ``decoder``; both are
    convolutions, which torch-rbln's eager mode does not run, so the pair becomes one
    compiled graph. Slicing is left to the module: it calls ``_decode`` per slice, which
    is still this graph. Tiling calls the decoder directly and is not supported here.
    """
    from diffusers.models.autoencoders.vae import DecoderOutput

    if getattr(vae, "_xfuser_rbln_decode_compiled", False):
        return

    class _Decode(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.post_quant_conv = vae.post_quant_conv
            self.decoder = vae.decoder

        def forward(self, z):
            if self.post_quant_conv is not None:
                z = self.post_quant_conv(z)
            return self.decoder(z)

    compiled = torch.compile(_Decode(vae), backend="rbln", dynamic=False, options=compile_options(cache_dir))

    def _decode(z, *args, **kwargs):
        return DecoderOutput(sample=compiled(_contiguous(z)))

    vae._decode = _decode
    vae._xfuser_rbln_decode_compiled = True
    logger.info("rbln: VAE decode path compiled with the rebel compiler")
