"""Rebellions all-to-all for Ulysses sequence parallelism under torch.compile.

rebel-compiler lowers ``rbln_custom_ops::ccl_all2all_x_kernel`` to the relay
``all_to_all_x`` op and on to ``rcclAllToAllX`` (librbln-ccl). This module makes
that path usable from xDiT's Ulysses attention when the transformer is compiled
with ``torch.compile(backend="rbln")``:

  1. declares the ``ccl_all2all_x_kernel`` torch.library custom op (so Dynamo
     traces it as an opaque node the rebel converter can match),
  2. declares a ``a2a_cast_u16`` helper used to build ``send_sizes`` as an
     in-graph uint16 *intermediate* (never a model I/O),
  3. injects the two rebel converters when they are not already registered (the
     converters live on a separate rebel-compiler branch; the runtime + MLIR
     ``all_to_all_x`` are already present on the build we target).

Constraints validated empirically on ATOM (2 NPUs); they shape the helper below:

  * **No input_desc** — a CCL op may not consume a raw graph input. Inside the
    transformer the all-to-all input is always an activation (a compute slot), so
    this holds naturally; ``send_sizes`` must likewise be produced by ops.
  * **Uniform device slots** — every CCL-op input must be a compute-op output.
    ``cast`` alone yields a non-slot; ``cat`` of a tensor yields a slot, so
    ``cast(fp32)->cat`` materialises a uint16 slot.
  * **uint16 only in-graph** — memory export (required for the transfer) has no
    ``uint16`` dtype, so ``send_sizes`` must be an intermediate, fed from fp32.
  * **Stay on the export frontend** — torch ``.to(uint16)`` drops compilation to
    the TorchScript frontend (which lacks these converters); the cast must be a
    relay-level op, hence the ``a2a_cast_u16`` custom op + converter.
"""
from __future__ import annotations

import torch
from torch import Tensor

import xfuser.envs as envs
from xfuser.core.utils.runner_utils import log

_OPS_DECLARED = False
_CONVERTERS_REGISTERED = False


def _declare_ops() -> None:
    """Declare the torch.library custom ops (idempotent, process-wide)."""
    global _OPS_DECLARED
    if _OPS_DECLARED:
        return

    try:
        @torch.library.custom_op("rbln_custom_ops::ccl_all2all_x_kernel", mutates_args=())
        def ccl_all2all_x_kernel(
            send_buffer: Tensor, send_sizes: Tensor, ccl_world_size: int, group_name: str
        ) -> Tensor:
            # CPU stub; under backend="rbln" the converter replaces this with the
            # real collective. Returns the same (R, t, H) shape it receives.
            R = ccl_world_size
            t = send_buffer.shape[1]
            H = send_buffer.shape[2]
            return torch.zeros(R, t, H, dtype=send_buffer.dtype)

        @ccl_all2all_x_kernel.register_fake
        def _ccl_all2all_x_kernel_fake(send_buffer, send_sizes, ccl_world_size, group_name):
            R = ccl_world_size
            t = send_buffer.shape[1]
            H = send_buffer.shape[2]
            return torch.empty(R, t, H, dtype=send_buffer.dtype)

        @torch.library.custom_op("rbln_custom_ops::a2a_cast_u16", mutates_args=())
        def a2a_cast_u16(x: Tensor) -> Tensor:
            return x.to(torch.uint16)

        @a2a_cast_u16.register_fake
        def _a2a_cast_u16_fake(x):
            return torch.empty_like(x, dtype=torch.uint16)
    except Exception as exc:  # already declared in this process
        log(f"rbln_alltoall: custom op declaration skipped ({exc})")

    _OPS_DECLARED = True


def _register_converters() -> None:
    """Inject the torch->relay converters into rebel-compiler if absent.

    Mirrors the proven ``extend_c10d_all_reduce`` pattern (string group passthrough,
    resolved at runtime via ``process_group_dict``). Uses ``setdefault`` so a rebel
    build that already ships ``_pt_ccl`` keeps its own converters.
    """
    global _CONVERTERS_REGISTERED
    if _CONVERTERS_REGISTERED:
        return
    try:
        import rebel.core.compilation.torch_to_relay as _ttr
        from rebel.core.custom_converter._pt_utils import wrapped_custom_op
        from rebel.core.custom_converter.pytorch import (
            custom_pytorch_convert_map as _orig_map,
        )
    except Exception:
        return  # rebel not importable (e.g. non-RBLN); nothing to inject

    def _extend_ccl_all2all_x_kernel(infer_shape_func, inputs, input_types):
        from rebel.core.relay.op import c10d
        return c10d.all_to_all_x(inputs[0], inputs[1], int(inputs[2]), str(inputs[3]))

    def _extend_a2a_cast_u16(infer_shape_func, inputs, input_types):
        from tvm.relay import op as _op
        return _op.cast(inputs[0], "uint16")

    def _patched_map(infer_shape_func):
        m = _orig_map(infer_shape_func)
        m.setdefault(
            "rbln_custom_ops::ccl_all2all_x_kernel",
            wrapped_custom_op(_extend_ccl_all2all_x_kernel, infer_shape_func),
        )
        m.setdefault(
            "rbln_custom_ops::a2a_cast_u16",
            wrapped_custom_op(_extend_a2a_cast_u16, infer_shape_func),
        )
        return m

    _ttr.custom_pytorch_convert_map = _patched_map
    _CONVERTERS_REGISTERED = True
    log("rbln_alltoall: injected ccl_all2all_x_kernel + a2a_cast_u16 converters")


def ensure_registered() -> None:
    """Declare the custom ops and inject the rebel converters (idempotent).

    Call once before ``torch.compile(backend="rbln")`` on any graph that uses the
    Ulysses all-to-all.
    """
    _declare_ops()
    if envs._is_rbln():
        _register_converters()


def _build_send_sizes(send_buffer: Tensor, world_size: int, rows: int) -> Tensor:
    """Build the ``(R, 64)`` uint16 ``send_sizes`` as an in-graph device slot.

    ``send_sizes[d, 0] = rows`` for every destination ``d`` (symmetric all-to-all);
    other columns are zero. Produced from fp32 via ``a2a_cast_u16`` then ``cat`` so
    the result is a genuine uint16 slot (see module docstring, constraint 2/3).
    """
    # fp32 (R, 64), col 0 = rows. Derived from send_buffer's device/options so it
    # lives on the NPU. (This constant->slot materialisation is the spot the
    # rebel-compiler may need to bless for symmetric collectives.)
    sizes_f = send_buffer.new_zeros((world_size, 64), dtype=torch.float32)
    sizes_f[:, 0] = float(rows)
    sizes_u = torch.ops.rbln_custom_ops.a2a_cast_u16(sizes_f)
    return torch.cat([sizes_u, sizes_u], dim=0)[:world_size]


def rbln_all_to_all_flat(x_flat: Tensor, world_size: int, group_name: str) -> Tensor:
    """All-to-all of a flat tensor across ``world_size`` ranks, lowered to
    ``rcclAllToAllX`` under ``torch.compile(backend="rbln")``.

    Matches ``dist.all_to_all_single`` semantics on a 1-D tensor whose length is
    ``world_size * chunk``: chunk ``d`` is sent to rank ``d`` and the result is the
    concatenation of the chunk received from each source rank.
    """
    assert x_flat.dim() == 1, f"expected flat tensor, got {x_flat.shape}"
    n = x_flat.shape[0]
    assert n % world_size == 0, f"{n} not divisible by world_size {world_size}"
    chunk = n // world_size
    # (R, t=1, H=chunk): one row of `chunk` elements per destination rank.
    send = x_flat.reshape(world_size, 1, chunk).contiguous()
    send_sizes = _build_send_sizes(send, world_size, rows=1)
    recv = torch.ops.rbln_custom_ops.ccl_all2all_x_kernel(
        send, send_sizes, world_size, group_name
    )
    return recv.reshape(n)
