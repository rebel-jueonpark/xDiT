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

## Fix #1 — APPLIED (advanced the compile past this spot)

`src/relay/transforms/rbln/pad_op_channels.cc`, `PadCclOp` (~line 2265):

```cpp
Array<ObjectRef> PadCclOp(const Call& ref_call, const Array<Expr>& normal_new_args,
                          const ObjectRef& ctx, const Array<Array<IndexExpr>>& vec_pad_values) {
-  if (vec_pad_values[0].empty()) {
+  // Symmetric CCL ops (e.g. all_to_all_x with an already-aligned channel dim)
+  // need no padding -> the framework hands us an empty vec_pad_values; guard the
+  // [0] access (mirrors PadWhere/PadConv) instead of crashing with IndexError.
+  if (vec_pad_values.empty() || vec_pad_values[0].empty()) {
    return {ref_call};
  }
  return {Call(ref_call->op, normal_new_args, ref_call->attrs, {}, ref_call->span), Bool(true)};
}
```

Applied as an **uncommitted** change in `~/rebel_compiler`; `librbln.so` rebuilt
(`ninja -C build librbln.so`). Verified: the crash moved off `PadCclOp` to the
next spot, confirming the guard is effective.

> Build note: `librbln.so` (target `CXX_SHARED_LIBRARY_LINKER__tvm_Release`) is
> NOT in ninja's default `all` target, and the CMake glob-recheck can leave the
> object stale. To force a rebuild of one source: delete the `.o`
> (`build/CMakeFiles/rbln_compiler_objs.dir/.../pad_op_channels.cc.o`) then
> `ninja -C build librbln.so`.

## Fix #2 — NEEDED (next empty-array spot)

After Fix #1 the crash is inside the `PadChannelsWithZero` machinery
(`src/relay/transforms/rbln/pad_channels.cc:108`) — an inlined `Array::operator[]`
on an empty pad array. The C++ backtrace has **no line numbers**, so pinpointing
needs a local debug build (`-g`, break on `tvm::runtime::ArrayNode` bounds-check,
or bisect the `vec_pad_values`/`pad_values` accesses). Candidate sites: the
`push_back_one_arg` device-resident branch (`MakeContribAlignedPad` with a
zero-size `pad_width`), `AddOpPaddingIfForced`, and any `[0]`/`[i]` on
`pad_values` reached when all entries are empty.

## Recommended proper fix

Harden the pass for **no-padding CCL ops** rather than per-spot band-aids:
- treat `fannotate_ccl` ops whose channel dims are already aligned as no-pad
  (return unchanged / terminator) early in `PadChannelsWithZero`, **or**
- guard every `vec_pad_values`/`pad_values` `[0]`/`[i]` access against empty.

MoE is unaffected because its `all_to_all_x` H always pads to 64 (non-empty
pad values). A unit test with an already-64-aligned `all_to_all_x` would lock
this in.

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
