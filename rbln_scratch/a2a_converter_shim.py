"""Runtime injection of the `ccl_all2all_x_kernel` rebel converter.

The current rebel-compiler branch (jueonpark/update-aten-ops-decompose) already
ships the MLIR relay op `all_to_all_x` and the runtime `rcclAllToAllX`
(librbln-ccl 3.2.2) — only the pure-Python torch->relay converter registration
is missing (it lives on origin/pr/ccl-kernels-converters-rebase).

Rather than edit the user's rebel source tree, we monkeypatch
`custom_pytorch_convert_map` (consumed fresh at every compile in
rebel/core/compilation/torch_to_relay.py:550) to add the one converter we need.

The converter mirrors the *proven* `extend_c10d_all_reduce` pattern: the group
identifier is passed straight through as a string attr (resolved at runtime via
`process_group_dict`); only the relay `all_to_all_x` op differs.
"""
from __future__ import annotations

import rebel.core.compilation.torch_to_relay as _ttr
from rebel.core.custom_converter._pt_utils import wrapped_custom_op
from rebel.core.custom_converter.pytorch import (
    custom_pytorch_convert_map as _orig_map,
)


def _extend_ccl_all2all_x_kernel(infer_shape_func, inputs, input_types):
    """rbln_custom_ops::ccl_all2all_x_kernel(send_buffer, send_sizes, world, group)
    -> relay all_to_all_x -> rcclAllToAllX."""
    from rebel.core.relay.op import c10d  # local import (matches upstream style)

    send_buffer = inputs[0]
    send_sizes = inputs[1]
    world = int(inputs[2])
    group = str(inputs[3])  # string passthrough, like extend_c10d_all_reduce
    return c10d.all_to_all_x(send_buffer, send_sizes, world, group)


def _extend_a2a_cast_u16(infer_shape_func, inputs, input_types):
    """rbln_custom_ops::a2a_cast_u16(x) -> relay cast to uint16.

    Produces send_sizes as an in-graph uint16 *intermediate* (an RBLN slot),
    so it is never a model I/O — the runtime's memory-export dtype enum has no
    'uint16' entry, and torch's own `.to(uint16)` drops the compile to the
    TorchScript frontend (which lacks our injected converters). A relay-level
    cast keeps everything on the export path.
    """
    from tvm.relay import op as _op  # local import

    return _op.cast(inputs[0], "uint16")


def _patched_map(infer_shape_func):
    m = _orig_map(infer_shape_func)
    m["rbln_custom_ops::ccl_all2all_x_kernel"] = wrapped_custom_op(
        _extend_ccl_all2all_x_kernel, infer_shape_func
    )
    m["rbln_custom_ops::a2a_cast_u16"] = wrapped_custom_op(
        _extend_a2a_cast_u16, infer_shape_func
    )
    return m


# Patch the consumer's bound name (it did `from ... import custom_pytorch_convert_map`).
_ttr.custom_pytorch_convert_map = _patched_map
print("[a2a-shim] injected rbln_custom_ops::ccl_all2all_x_kernel -> relay all_to_all_x")
