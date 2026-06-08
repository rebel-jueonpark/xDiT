"""Runtime patches against `rebel-compiler` for ops that ship missing in
torch_to_relay but DO have working built-in PyTorch decompositions.

Why this exists
---------------
The current rebel-compiler wheel raises
``NotImplementedError: The following operators are not implemented:
['aten::outer', 'aten::rms_norm']`` when ``torch.compile(backend="rbln")`` lowers
FLUX / Qwen-Image graphs. Both ops have ``torch.export.default_decompositions()``
entries that reduce them to primitives the rebel-compiler frontend already
handles:

- ``aten.outer.default``      → ``aten.view`` + ``aten.mul``
- ``aten.rms_norm.default``   → ``aten._fused_rms_norm.default``
- ``aten._fused_rms_norm.default`` → ``aten.pow / mean.dim / add.Scalar / rsqrt /
                                       mul / type_as``

All of the lowered ops are already in the rebel-compiler convert map, so the
fix is simply to extend the ``ATEN_OPS_TO_DECOMPOSE`` set so that
``exported.run_decompositions(...)`` rewrites these ops before relay conversion.

Once the editable rebel-compiler build is unbroken (currently fails with an
undefined ``rbln::pyvmem::debug::Free`` symbol when rebuilding ``_C.so``), the
same change lives in the source at
``rebel_compiler/rebel/python/rebel/core/compilation/torch_to_relay_aten_ops.py``
and this runtime patch becomes a no-op.
"""
from __future__ import annotations

import logging
import os

_patched = False


def install_rbln_compiler_aten_decomp_patches() -> None:
    """Add aten.outer / aten.rms_norm / aten._fused_rms_norm to the
    rebel-compiler decomposition set. Safe to call multiple times."""
    global _patched
    if _patched:
        return
    try:
        import torch
        from rebel.core.compilation.torch_to_relay_aten_ops import (
            ATEN_OPS_TO_DECOMPOSE,
        )
    except ImportError:
        # rebel-compiler is optional from xfuser's perspective.
        return

    missing_ops = [
        torch.ops.aten.outer.default,
        torch.ops.aten.rms_norm.default,
        torch.ops.aten._fused_rms_norm.default,
    ]
    added = [op for op in missing_ops if op not in ATEN_OPS_TO_DECOMPOSE]
    ATEN_OPS_TO_DECOMPOSE.update(missing_ops)
    if added:
        logging.getLogger(__name__).info(
            "rebel-compiler decomposition set extended with: %s",
            ", ".join(str(op) for op in added),
        )

    _install_compose_expr_arity_fix()

    _patched = True


def _install_compose_expr_arity_fix() -> None:
    """Repair an arity-mismatch regression in rebel-compiler 0.10.4.post1.

    The Pyarmor-obfuscated wheel's frozen ``get_preprocess_passes`` body
    calls Pass factories like ``rebel_transform.ComposeExpr(1)``,
    ``rebel_transform.LowerExpr(1)``, etc. — but the matching C++
    factories in this wheel's ``_C.so`` are still the 0-arg form
    (``Pass ComposeExpr()``, ``Pass LowerExpr()``, …). The TVM PackedFunc
    layer therefore raises::

        TVMError: Function relay._transform.<X>() -> transform.Pass
        expects 0 arguments, but N were provided.

    for every affected pass, then the diagnostic is re-emitted as a fatal
    diagnostic and ``torch.compile(backend="rbln")`` silently falls back
    to CPU execution.

    Mitigation: wrap each ``_ffi_api`` PackedFunc that exposes this gap
    with a thin proxy that retries with 0 args when the underlying C++
    PackedFunc complains about an unexpected positional. Wheels with a
    matching C++ signature (which would accept the extra arg) take the
    fast path and are unaffected. Once the rebel-compiler wheel is
    rebuilt with consistent Python+C++ signatures this is a no-op.
    """
    try:
        from tvm.relay.transform import _ffi_api  # type: ignore
    except ImportError:
        return

    if getattr(_ffi_api, "_xfuser_arity_fix_installed", False):
        return

    _ARITY_ERR_NEEDLE = "expects 0 arguments, but"

    def _make_adaptive(name, original):
        def _adaptive(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if _ARITY_ERR_NEEDLE in msg and (args or kwargs):
                    logging.getLogger(__name__).debug(
                        "rebel-compiler %s: stripping %d args / %d kwargs "
                        "due to C++ side expecting 0 args (wheel/C++ "
                        "signature mismatch).",
                        name, len(args), len(kwargs),
                    )
                    return original()
                raise

        _adaptive.__wrapped_by_xfuser__ = True  # type: ignore[attr-defined]
        return _adaptive

    # Pass factories observed mismatched in the wheel. Adding more here is
    # cheap — the adaptive proxy is a no-op when the C++ side accepts the arg.
    candidate_names = [
        "ComposeExpr",
        "LowerExpr",
        "SimplifyConstant",
        "SimplifyExpr",
        "SimplifyExprRebel",
        "CanonicalizeOpsRebel",
        "OptimizeTargetOpsRebel",
        "SimplifyReshapeOps",
    ]

    wrapped = []
    for name in candidate_names:
        original = getattr(_ffi_api, name, None)
        if original is None or getattr(original, "__wrapped_by_xfuser__", False):
            continue
        setattr(_ffi_api, name, _make_adaptive(name, original))
        wrapped.append(name)

    _ffi_api._xfuser_arity_fix_installed = True

    if wrapped:
        logging.getLogger(__name__).info(
            "rebel-compiler Pass-factory arity guard installed on: %s "
            "(wheel calls factories with extra args; C++ side takes 0). "
            "Falls back to 0-arg call when needed.",
            ", ".join(wrapped),
        )

    if os.environ.get("XFUSER_RBLN_STUB_MISSING_PASSES", "0") == "1":
        _install_missing_pass_stubs(_ffi_api)


def _install_missing_pass_stubs(_ffi_api) -> None:
    """Register identity Pass factories for names that the wheel's
    ``get_preprocess_passes`` references but the wheel's ``_C.so`` does
    not actually expose. Without this the Python pipeline construction
    raises ``AttributeError`` before any C++ pass runs.
    """
    try:
        import tvm
        from tvm import IRModule
        from tvm.ir.transform import module_pass
    except ImportError:
        return

    # Identity module-pass factory: returns the IRModule unchanged.
    def _make_identity_pass(name):
        def _factory(*args, **kwargs):
            @module_pass(opt_level=0, name=name)
            def _identity(mod: IRModule, ctx) -> IRModule:
                return mod

            return _identity

        _factory.__wrapped_by_xfuser__ = True  # type: ignore[attr-defined]
        return _factory

    # Pass names the wheel build_module pipeline references but that the
    # wheel's compiled C++ side does not expose. Stub each as a no-op so
    # the compile sequence runs to completion.
    stub_names = [
        "SimplifyTransposeOps",
    ]
    stubbed = []
    for name in stub_names:
        if hasattr(_ffi_api, name):
            continue
        setattr(_ffi_api, name, _make_identity_pass(name))
        stubbed.append(name)

    if stubbed:
        logging.getLogger(__name__).info(
            "rebel-compiler missing-pass stubs installed (identity passes): %s",
            ", ".join(stubbed),
        )
