# Running xDiT on Rebellions ATOM™ NPU — `torch.compile` + rebel-compiler

This branch (`adopt_atom_npu_compile`) runs xDiT diffusion transformers on
**Rebellions ATOM NPUs** with **pure PyTorch + rebel-compiler via
`torch.compile(backend="rbln")`**, and targets **multiple NPUs** via tensor
parallelism.

> **Why this branch is different.** The sibling `adopt_atom_npu` branch runs the
> transformer in **eager** mode on the `torch-rbln` backend — which is single-NPU
> and cannot run convolution (`convolution_overrideable not implemented`). This
> branch instead **compiles** the transformer with the **rebel-compiler**
> torch.compile backend, which **supports conv2d/conv3d** and can **lower in-graph
> collectives (`all_reduce`/`all_gather`) onto `rbln-ccl`** for multi-NPU tensor
> parallelism — the same technique `vllm-rbln` uses for LLMs.

> **Status — Stage 1a done, Stage 1b actively unblocking.** Backend wiring is done
> and verified on representative micro-graphs (conv2d/conv3d, bf16 SDPA, `outer`,
> `rms_norm`). The Flux/Qwen-Image RoPE + QK-RMSNorm path now lowers cleanly
> after the `aten::outer` / `aten::rms_norm` decomposition fix landed
> ([§6](#6-status--roadmap), [§7](#7-troubleshooting)). Full end-to-end model
> bring-up and multi-NPU TP are still in progress.

---

## 1. Prerequisites

### Hardware
One or more Rebellions ATOM NPUs:

```bash
rbln-stat            # lists NPUs (15.7 GiB each), utilization
```

### Software stack (verified set)
| Package          | Version                          | Role |
| ---              | ---                              | ---  |
| Python           | 3.12                             | |
| `torch`          | `2.10.0+cpu`                     | CPU build; NPU support comes from the rebel/torch-rbln stack |
| `torch-rbln`     | `0.2.1`                          | registers the `rbln` device + `rbln-ccl` distributed backend |
| `rebel-compiler` | **`0.10.5.dev145+` (source build, see [§2.1](#21-build-rebel-compiler-from-source))** | provides the `torch.compile` backend `"rbln"` |
| `diffusers`      | `0.37.1`                         | |
| `transformers`   | `5.9.0`                          | |
| `torchvision`    | `0.25.0+cpu`                     | optional; only some processors need it (must match the torch version) |

The conv + collective lowering all happen inside `rebel-compiler`'s
`torch.compile` backend; `torch-rbln` supplies the `rbln` device and the
`rbln-ccl` process-group backend.

> **About the rebel-compiler version.** The published wheel
> `rebel-compiler==0.10.4.post1` has a partial-refactor leak: its Pyarmor-built
> Python pipeline references `SimplifyTransposeOps` and calls several Pass
> factories with `(batch_size=1)` while the same wheel's `_C.so` exposes the
> 0-arg form and never registered `SimplifyTransposeOps`. Symptoms include
> `TVMError: ... expects 0 arguments, but 1 were provided` and
> `AttributeError: module 'tvm.relay.transform._ffi_api' has no attribute
> 'SimplifyTransposeOps'`. For now, **build rebel-compiler from source**
> (instructions below). xfuser ships a runtime safety net for the wheel
> (`xfuser/core/rbln_compiler_patches.py`) that wraps the affected factories;
> it is a no-op on the source-built compiler.

---

## 2. Installation

### 2.1 Build rebel-compiler from source

```bash
# Prereqs: conan (>=2.1), cmake, ninja, an x86_64 Linux box, an existing
# Rebellions SDK install for the dynamic libraries (rblnthunk / rbln-ccl).
git clone <rebel-compiler-repo> rebel_compiler && cd rebel_compiler
git submodule update --init
chmod a+x rebel_install.sh && ./rebel_install.sh       # configures + builds the C++ libs
source ./dynamic_linking.env                            # exports LD_LIBRARY_PATH for rblnthunk/ccl

# Editable Python install (picks up librbln.so from the build dir)
pip install scikit-build-core pybind11 cython "cmake>=3.26" ninja setuptools_scm
REBEL_BUILD_DIR=$PWD/build pip install --no-build-isolation -e python/
```

Verify the install picked up the source tree, not the wheel:

```bash
python -c "import rebel; print(rebel.__file__)"
# expect a path under rebel_compiler/rebel/python/, not site-packages
python -c "
import inspect, rebel.core.transform as rt
print('ComposeExpr sig:', inspect.signature(rt.ComposeExpr))   # expect ()
print('SimplifyTransposeOps present:', hasattr(rt, 'SimplifyTransposeOps'))  # expect False
from rebel.core.compilation.torch_to_relay_aten_ops import ATEN_OPS_TO_DECOMPOSE
import torch
print('outer registered:', torch.ops.aten.outer.default in ATEN_OPS_TO_DECOMPOSE)
print('rms_norm registered:', torch.ops.aten.rms_norm.default in ATEN_OPS_TO_DECOMPOSE)
"
# expect True for the last two
```

### 2.2 Install torch + torch-rbln

```bash
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-rbln==0.2.1
```

Sanity-check both the device and the compile backend are present:

```bash
python - <<'PY'
import torch, torch_rbln
from rebel.core.torch_compile import rbln_backend   # the torch.compile backend
print("rbln device:", torch.rbln.is_available())
print("device count:", torch.rbln.device_count())
PY
```

### 2.3 Install xDiT (this branch)

```bash
git clone https://github.com/rebel-jueonpark/xDiT.git
cd xDiT
git checkout adopt_atom_npu_compile
pip install -e .
```

### 2.4 (Optional) torchvision

Only if a processor raises `requires the Torchvision library`:

```bash
pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torchvision==0.25.0
```

---

## 3. Environment setup

Set the RBLN bootstrap variables **before** `torch_rbln` is imported (the bundled
launchers do this for you):

```bash
export RCCL_FORCE_EXPORT_MEM=1
export RCCL_PORT_GEN=1                 # multi-NPU rbln-ccl port generation
export RBLN_ROOT_IP=127.0.0.1
export RBLN_LOCAL_IP=127.0.0.1
export MASTER_ADDR=127.0.0.1
export HF_HOME=/path/to/huggingface_cache   # + token for gated models
```

---

## 4. Enabling the rebel `torch.compile` path

Compilation is opt-in via `--use_torch_compile`, and the backend is selected by
`--torch_compile_backend`:

| `--torch_compile_backend` | Effect |
| --- | --- |
| `auto` (default) | `rbln` (rebel-compiler) when an ATOM NPU is present, else `inductor` |
| `rbln` | force the rebel-compiler backend |
| `inductor` | force PyTorch Inductor (CPU/GPU) |

On the `rbln` backend xDiT compiles **`pipe.transformer`** with
`torch.compile(..., backend="rbln", dynamic=False, options={mode:["strict"],
compile_context, cache_dir, tensor_parallel_size, process_group_dict})`. Compiled
artifacts are cached under `<output_directory>/rbln_compile_cache/` and reused on
re-runs.

### Single NPU

```bash
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29500 \
    examples/flux2_rbln_example.py \
    --model SD3.5 \
    --use_torch_compile --torch_compile_backend rbln \
    --ulysses_degree 1 --tensor_parallel_degree 1 \
    --height 512 --width 512 --num_inference_steps 4 \
    --prompt "a futuristic city at dusk" \
    --output_directory ./outputs
```

### Multiple NPUs (tensor parallelism)

Multi-NPU uses **tensor parallelism** (the rebel backend lowers the TP
`all_reduce`/`all_gather` onto `rbln-ccl`). Launch one process per NPU and set
`--tensor_parallel_degree` = number of NPUs:

```bash
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29500 \
    examples/flux2_rbln_example.py \
    --model <model> \
    --use_torch_compile --torch_compile_backend rbln \
    --tensor_parallel_degree 4 --ulysses_degree 1 \
    --height 512 --width 512 --num_inference_steps 4 \
    --prompt "..." --output_directory ./outputs
```

> Use **tensor parallelism**, not sequence parallelism, on this path:
> rebel-compiler lowers `all_reduce`/`all_gather` in-graph but **not `all_to_all`**,
> so xDiT's USP/Ulysses (`--ulysses_degree > 1`) cannot be compiled here. Keep
> `--ulysses_degree 1`.

---

## 5. Static shapes

The rebel backend requires `dynamic=False` (static shapes). A separate compiled
artifact is produced/cached per distinct `(height, width, batch, sequence)`. Keep
resolution, batch, and CFG mode fixed for a given run; changing them recompiles.

---

## 6. Status & roadmap

The work is staged. **Stage 1a (backend wiring) is done and verified at the op
level; full models are being brought up.**

| Stage | Goal | Status |
| --- | --- | --- |
| **1a — backend wiring** | route `pipe.transformer` through `torch.compile(backend="rbln")` | ✅ done; conv2d/conv3d + bf16 SDPA verified compiling on one NPU |
| **1a.5 — RoPE + QK-RMSNorm ops** | `aten::outer` and `aten::rms_norm` (Flux/Qwen family) | ✅ done via decomposition through existing primitives; mini-graph compiles and runs on `rbln:0` |
| **1b — single-NPU full model** | compile a whole DiT transformer end-to-end | 🚧 in progress |
| **2 — multi-NPU FFN-TP** | FeedForward tensor parallelism, collectives via rbln-ccl | ⏳ planned |
| **3 — attention TP** | Megatron-style column/row-parallel attention | ⏳ planned |

**Bring-up history (Stage 1a.5).** The original Flux/Qwen-Image blocker —
`NotImplementedError: ['aten::outer', 'aten::rms_norm']` — is resolved by
extending rebel-compiler's `ATEN_OPS_TO_DECOMPOSE` set so PyTorch's
`torch.export.default_decompositions()` rewrites both ops into primitives the
frontend already supports:

- `aten.outer.default` → `aten.view` + `aten.mul`
- `aten.rms_norm.default` → `aten._fused_rms_norm.default` → `aten.pow` +
  `aten.mean.dim` + `aten.add.Scalar` + `aten.rsqrt.default` + `aten.mul.Tensor` +
  `aten.type_as.default`

The change lives in
`rebel_compiler/rebel/python/rebel/core/compilation/torch_to_relay_aten_ops.py`
(in the source build) and is also re-applied at runtime by
`xfuser/core/rbln_compiler_patches.py` as a wheel-compat safety net. Verified by
a FLUX-style mini-graph (`torch.outer` + `torch.nn.RMSNorm` + RoPE-style fusion)
that compiles through `RunEarlyCompilation` → `RunLateCompilation` →
`genCommand: finished` and produces correct output on `rbln:0` (no CPU
fallback).

**Current Stage 1b targets:**
- **Flux / Flux2 / Qwen-Image** — `outer` + `rms_norm` no longer block; bring-up
  now focuses on remaining whole-model patterns surfacing during full forward
  trace (FFN activations, attention masks, residual paths).
- **SD3.5** — its transformer forward currently routes to Inductor instead of
  the rebel backend; that routing needs fixing before it compiles on the NPU.

Because rebel supports conv, the eventual model coverage on this path is broader
than the eager branch (it should include the conv-patchify DiTs and VAEs that
eager cannot run) — pending the items above.

---

## 7. Troubleshooting

| Error / symptom | Cause & fix |
| --- | --- |
| `NotImplementedError: The following operators are not implemented: ['aten::outer', 'aten::rms_norm']` | **Fixed** in source builds of rebel-compiler — the ops are now decomposed via `ATEN_OPS_TO_DECOMPOSE`. xfuser's `rbln_compiler_patches.py` also re-applies the fix at runtime for the wheel. If you still see this, you're on an older site-packages install — re-do [§2.1](#21-build-rebel-compiler-from-source) and confirm `python -c "import rebel; print(rebel.__file__)"` points at the source tree. |
| `TVMError: Function relay._transform.ComposeExpr() -> transform.Pass expects 0 arguments, but 1 were provided.` (or the same for `LowerExpr`/`SimplifyExprRebel`/…) | Wheel-only regression in `rebel-compiler==0.10.4.post1`: its frozen `get_preprocess_passes` calls Pass factories with `(batch_size=1)` while the same wheel's `_C.so` only registers the 0-arg form. **Fix:** build rebel-compiler from source ([§2.1](#21-build-rebel-compiler-from-source)). Runtime safety net in `xfuser/core/rbln_compiler_patches.py` wraps the affected factories and retries with 0 args on this specific error. |
| `AttributeError: module 'tvm.relay.transform._ffi_api' has no attribute 'SimplifyTransposeOps'` | Same wheel-vs-C++ partial refactor: wheel's Python `get_main_passes` references a Pass the wheel's `_C.so` never registered. **Fix:** build rebel-compiler from source. Opt-in runtime fallback `XFUSER_RBLN_STUB_MISSING_PASSES=1` registers an identity stub (lets the pipeline progress but typically segfaults later, since downstream codegen depends on the canonical form that pass would produce). |
| `ImportError: _C.cpython-…so: undefined symbol: _ZN4rbln6pyvmem5debug4FreeEm` (or another `rbln::...` symbol) | The `librbln.so` in the build dir is **older** than the `vmem.cc`/headers the editable install is building `_C.so` from. Re-run `ninja librbln.so` in the rebel-compiler build dir (the symbol gets added to `librbln.so`), then re-install editable. |
| `torch._inductor.exc.InductorError` on an `rbln` tensor | The compile fell to **Inductor** instead of the rebel backend. Ensure `--torch_compile_backend rbln` (or `auto` on an NPU) and that the model's `_compile_model` routes through `_compile_transformer_rbln`. |
| `DiagnosticError: one or more error diagnostics were emitted` (generic, in PROD mode) | `rebel.core._env.ENV == "PROD"` suppresses the underlying message. Set it to `"DEV"` temporarily to surface the real error: `python -c "import rebel.core._env as e; e.ENV='DEV'; import rebel.core.build_module; rebel.core.build_module.ENV='DEV'; ..."`. |
| `RCCL ... failed` / multi-NPU hangs | Ensure `RCCL_PORT_GEN=1` and the bootstrap env are set; use **tensor parallelism** (`--tensor_parallel_degree N`), not `--ulysses_degree` (`all_to_all` isn't compilable). |
| `requires the Torchvision library` | Install the matching torchvision ([§2.4](#24-optional-torchvision)). |
| First run is slow | Compilation runs once and is cached under `<output_directory>/rbln_compile_cache/`; subsequent runs reuse the `.rbln` artifact. |

---

## 8. What this branch adds (vs the eager `adopt_atom_npu` branch)

- `--torch_compile_backend {auto,inductor,rbln}` (`xfuser/config/args.py`).
- `base_model._compile_model` dispatches on the backend; `_compile_transformer_rbln`
  builds the rebel options + a `process_group_dict` from the `GroupCoordinator`s so
  the backend can lower tensor-parallel collectives onto `rbln-ccl`
  (`xfuser/model_executor/models/runner_models/base_model.py`).
- `flux.py` / `stable_diffusion.py` `_compile_model` overrides route through the
  rebel path (transformer only; text encoders stay eager).
- `RCCL_PORT_GEN` added to the example launcher for multi-NPU.
- **`xfuser/core/rbln_compiler_patches.py`** (new): at xfuser import time when
  `_is_rbln()`, (a) extends `rebel-compiler`'s `ATEN_OPS_TO_DECOMPOSE` set to
  include `aten.outer.default`, `aten.rms_norm.default`, and
  `aten._fused_rms_norm.default` so the Flux/Qwen RoPE+QK-RMSNorm graph lowers
  through existing primitives; (b) installs an adaptive arity guard on the
  affected `_ffi_api` Pass factories so the broken wheel doesn't trip on
  `expects 0 arguments, but 1 were provided`; (c) provides an opt-in
  (`XFUSER_RBLN_STUB_MISSING_PASSES=1`) identity stub for `SimplifyTransposeOps`
  on the broken wheel. **All three are no-ops on a source-built rebel-compiler.**

It builds on the RBLN device/backend enablement from `adopt_atom_npu` (the `rbln`
device, `rbln-ccl` selection, and the scheduler-dtype/VAE-on-CPU compat hooks).
