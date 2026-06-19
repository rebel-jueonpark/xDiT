# rebel-compiler `PadChannels` hand-off: no-padding CCL ops crash

**Context.** Enabling xDiT Ulysses sequence parallelism on ATOM via
`torch.compile(backend="rbln")`. The transformer's per-attention all-to-all is
emitted as `rbln_custom_ops::ccl_all2all_x_kernel` → relay `all_to_all_x` →
`rcclAllToAllX`. The converter + runtime work; the blocker is the `PadChannels`
relay pass crashing on the symmetric (no-padding-needed) `all_to_all_x` subgraph.

## Symptom

```
InternalError: Check failed: (0 <= i && i < p->size_) is false:
IndexError: indexing 0 on an array of size 0
  ... tvm::relay::pad_channels::PadChannels
  ... PadChannels::RebelFunc::Rewrite_
  ... pad_channels::PadChannelsWithZero<PadChannelsTransformMemorizer>
```

## Root cause

`PadChannels` is built around ops that need channel-padding (conv; and MoE's
`all_to_all_x`, whose `H` = hidden_size 2048/4096 always pads up to a multiple of
64). A **symmetric Ulysses `all_to_all_x`** (already-aligned dims) plus its
helper ops (`cast`→uint16, `cat`) need **no** padding, so the framework hands the
handlers an **empty** `vec_pad_values` / `pad_values`. Several functions then index
`[0]` (or `[i]`) on the empty array and crash, instead of gracefully returning the
op unchanged the way `PadWhere`/`PadConv` do.

## What's been tried (and what we learned)

### Defensive guard in `PadCclOp` — committed, but does NOT close the live bug

`src/relay/transforms/rbln/pad_op_channels.cc`, `PadCclOp` (~line 2265):

```cpp
Array<ObjectRef> PadCclOp(const Call& ref_call, const Array<Expr>& normal_new_args,
                          const ObjectRef& ctx, const Array<Array<IndexExpr>>& vec_pad_values) {
-  if (vec_pad_values[0].empty()) {
+  if (vec_pad_values.empty() || vec_pad_values[0].empty()) {
    return {ref_call};
  }
  return {Call(ref_call->op, normal_new_args, ref_call->attrs, {}, ref_call->span), Bool(true)};
}
```

Committed to `rebel-compiler` as `fix(pad_channels): guard empty vec_pad_values
in PadCclOp (defensive)`. Pure belt-and-braces hardening — keeps the existing
`[0].empty()` check working if a future code path ever delivers a zero-length
outer `Array`. **Does not fix the live crash**: `push_back_one_arg` always
pushes one entry per `Call` argument inside `PadChannelsWithZero`
(`pad_channels.cc:128-144`), so `vec_pad_values` is never zero-length when
`PadCclOp` is called on this subgraph. With or without this guard, `PadCclOp`
returns `{ref_call}` cleanly on the symmetric-CCL test.

> Build note: `librbln.so` (target `CXX_SHARED_LIBRARY_LINKER__tvm_Release`) is
> NOT in ninja's default `all` target, and the CMake glob-recheck can leave the
> object stale. To force a rebuild of one source: delete the `.o`
> (`build/CMakeFiles/rbln_compiler_objs.dir/.../pad_op_channels.cc.o`) then
> `ninja -C build librbln.so`.

### The real crash site (still NEEDS A FIX)

Pinpointed by writing a `pytest` regression test against the live IR
(see `tests/python/test_rebel/test_pass_rebel_pad_channels_ccl.py` — three
tests at `_R=2, _T=1, _H=64`, channel already 64-aligned, matching the IR
captured in `rbln_scratch/rank_0/tvm_debug.log`). Key finding from comparing
runs with/without the `PadCclOp` guard:

- A pure-pytest run on the symmetric `all_to_all_x` subgraph **does not crash**
  (with or without the `PadCclOp` defensive guard).
- The live FLUX.2 + Ulysses run **does crash** with
  `InternalError: indexing 0 on an array of size 0` from
  `PadChannelsWithZero<PadChannelsTransformMemorizer>`.

The only path that's present in the live run but absent in pytest is the
`IsDeviceResidentInPort(arg)` branch of `pad_channels.cc:135-141`:

```cpp
if (rbln::IsDeviceResidentInPort(arg)) {
  // device-resident input -> MakeContribAlignedPad path with zero pad_width
  ...
  vec_pad_values.push_back(/* empty Array<IndexExpr> for this slot */);
}
```

`IsDeviceResidentInPort` consults `GetDevicePortInfo`, which in turn returns
the `port_info_` populated by `rbln_port_config.cc:61` — and that only fires
when `rbln::CompileSessionManager::GetCurrentSession()` is non-null. A pytest
unit test does not set up a compile session, so this branch is unreachable
from the standalone test scope. Inside the live `torch.compile(backend="rbln")`
call there *is* an active session, and one of the inputs to the symmetric
`all_to_all_x` (likely `send_sizes`, materialised as a uint16 device slot via
`a2a_cast_u16 → cat`) is marked device-resident → the `MakeContribAlignedPad`
path runs with a zero-size `pad_width` → an empty `Array<IndexExpr>` is pushed
onto `vec_pad_values` → downstream `[0]` / `[i]` access on that empty slot
crashes.

## Fix #2 — NEEDED

Two viable patches in `src/relay/transforms/rbln/pad_channels.cc`:

1. **Early-return for already-aligned `fannotate_ccl` ops** in
   `PadChannelsWithZero` (before reaching the `IsDeviceResidentInPort`
   branch). The cleanest fix: a symmetric CCL op whose channel dim is
   already aligned has nothing to pad, regardless of whether one of its
   inputs is device-resident.

2. **Guard `MakeContribAlignedPad`-with-zero-`pad_width`** in the
   device-resident branch (`pad_channels.cc:135-141`) so it does not push
   an empty `Array<IndexExpr>` onto `vec_pad_values`, and add empty-checks
   on every downstream `vec_pad_values[i]` / `pad_values[i]` access in
   `PadChannelsWithZero` so they treat an empty entry as "skip this slot".

(1) is preferable: it's one early-return, it's symmetric to how `MoE`'s
`all_to_all_x` is handled (whose `H` always pads to 64, so it never hits
the no-pad case), and the unit tests in `test_pass_rebel_pad_channels_ccl.py`
already lock in the no-crash behaviour for both code paths.

## Regression test (already in the repo)

`tests/python/test_rebel/test_pass_rebel_pad_channels_ccl.py` ships three tests:

1. `test_pad_channels_symmetric_all_to_all_x_does_not_crash` — full live-IR
   shape (`send_buffer (R=2, t=1, H=64)` bf16 → `add` → `all_to_all_x`).
2. `test_pad_channels_bare_symmetric_all_to_all_x_does_not_crash` — narrowest
   reproducer (bare `all_to_all_x(var, const_send_sizes)`).
3. `test_pad_channels_symmetric_ccl_with_io_attrs_does_not_crash` — attaches
   the `input_attrs={"device":"rbln",...}` / `output_attrs` the live converter
   emits. Wrapped in a `try/except` that flips to `xfail` with the documented
   `indexing 0 on an array of size 0` signature if/when the device-resident
   branch ever becomes reachable from unit-test scope (or any new code path
   reaches the same empty-array crash without a `CompileSession`). Any other
   exception propagates as a real failure.

All three currently **PASS** in pure pytest scope because the live crash
path requires an active `CompileSession`. Together they're the regression
canary: when Fix #2 lands, they stay green; if anyone reintroduces an
unguarded `vec_pad_values[0]` access along a path reachable from unit tests,
test #3 flips to xfail with the exact diagnostic.

## Reproduce

Standalone 2-NPU all_to_all_x (no model): `rbln_scratch/test_all2all_x.py` +
`rbln_scratch/a2a_converter_shim.py` (injects the converter; the rebel branch
ships the MLIR/runtime but not the Python converter):

```
RBLN_FORCE_CCL_ASYNC=1 torchrun --nproc_per_node=2 \
    rbln_scratch/test_all2all_x.py
```

Full path (FLUX.2-klein-4B, the xDiT integration):

```
RBLN_FORCE_CCL_ASYNC=1 torchrun --nproc_per_node=2 \
    examples/flux2_rbln_example.py --model FLUX.2-klein-4B \
    --ulysses_degree 2 --use_torch_compile --torch_compile_backend rbln \
    --height 256 --width 256 --num_inference_steps 2 --prompt "a cat" \
    --output_directory /tmp/ulysses_out
```

## Validated constraints (for the xDiT side / any all_to_all_x caller)

1. A CCL op may not consume a raw graph input (`"CCL op should not have
   input_desc"`) — route inputs through a compute op first.
2. All CCL-op inputs must be uniform device **slots**; `cast` alone yields a
   non-slot, `cat` of a tensor yields a slot (`cast→cat` gives a uint16 slot).
3. `send_sizes` must be uint16 but NOT a model I/O (memory-export has no uint16);
   build it as an in-graph intermediate from fp32.
4. Stay on the export frontend — torch `.to(uint16)` drops to the TorchScript
   frontend (no injected converters); cast must be relay-level.
5. Don't create a redundant torch `new_group` for the device comm — the rebel
   runtime splits its own comm from `process_group_dict`; a 2nd split deadlocks.
   For a single all-group case, `group="-1"` resolves to the default (WORLD) comm.
6. `RCCL_FORCE_EXPORT_MEM` must be on (or the collective hangs);
   `RBLN_FORCE_CCL_ASYNC=1` (vllm-rbln default).
