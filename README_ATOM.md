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
- **SD3.5** — base `_compile_model` correctly routes to rebel on NPU and its
  override calls `_compile_transformer_rbln` (`runner_models/stable_diffusion.py:63`).
  Remaining work is downstream op coverage during the full transformer compile.
- **CausalWan / LTX / Hunyuan** — these runners' `_compile_model` overrides
  hard-code `torch.compile(..., mode="default")` (Inductor) and will fall to
  CPU/GPU paths on an NPU. Routing them through `_compile_transformer_rbln`
  is a small follow-up (see [§7](#7-troubleshooting) `InductorError` row).

Because rebel supports conv, the eventual model coverage on this path is broader
than the eager branch (it should include the conv-patchify DiTs and VAEs that
eager cannot run) — pending the items above.

---

## 7. Troubleshooting

Each row below is tagged with where it currently applies. Items marked
**[wheel-only]** do not reproduce on a source-built `rebel-compiler` from
[§2.1](#21-build-rebel-compiler-from-source) — they are kept here as a paper
trail for anyone still running `0.10.4.post1` from PyPI.

| Error / symptom | Where it applies | Cause & fix |
| --- | --- | --- |
| `NotImplementedError: The following operators are not implemented: ['aten::outer', 'aten::rms_norm']` | **resolved on source build** (also resolved on the wheel by the runtime patch) | The Flux/Qwen-Image RoPE + QK-RMSNorm path used to fail rebel-compiler conversion. Both ops are now decomposed via `torch.export.default_decompositions()` into primitives the frontend already supports — `outer → view+mul`, `rms_norm → _fused_rms_norm → pow+mean.dim+add+rsqrt+mul+type_as`. The source build's `ATEN_OPS_TO_DECOMPOSE` set includes all three overloads; `xfuser/core/rbln_compiler_patches.py` re-applies the same change at runtime for users still on the wheel. Verify with `python -c "import rebel; print(rebel.__file__)"` — should point under `rebel_compiler/rebel/python/`. |
| `TVMError: Function relay._transform.ComposeExpr() -> transform.Pass expects 0 arguments, but 1 were provided.` (and the same for `LowerExpr` etc.) | **[wheel-only]** `rebel-compiler==0.10.4.post1` | The wheel's frozen `get_preprocess_passes` calls Pass factories with `(batch_size=1)` while the same wheel's `_C.so` only registers the 0-arg form. **Source-built rebel-compiler does not exhibit this** — `inspect.signature(rebel.core.transform.ComposeExpr) == ()` and the C++ side matches. Wheel safety net: `xfuser/core/rbln_compiler_patches.py` wraps the affected `_ffi_api` factories and retries with 0 args on the specific TVMError. (The guard is installed at xfuser import; logs show `Pass-factory arity guard installed on: ComposeExpr, LowerExpr, …`. On a source build it never trips — verify by absence of `stripping N args` debug lines.) |
| `AttributeError: module 'tvm.relay.transform._ffi_api' has no attribute 'SimplifyTransposeOps'` | **[wheel-only]** `rebel-compiler==0.10.4.post1` | Same wheel-vs-C++ partial-refactor leak: wheel's `get_main_passes` calls `rebel_transform.SimplifyTransposeOps()` but the wheel's `_C.so` never registered it under any name. **Source build does not reference this pass at all** (`hasattr(rebel.core.transform, 'SimplifyTransposeOps') == False`). Fix on the wheel: build from source ([§2.1](#21-build-rebel-compiler-from-source)). The opt-in `XFUSER_RBLN_STUB_MISSING_PASSES=1` identity-Pass fallback lets the pipeline progress but typically segfaults at runtime — downstream codegen depends on the canonical form that pass was supposed to produce. |
| `ImportError: _C.cpython-…so: undefined symbol: _ZN4rbln6pyvmem5debug4FreeEm` (or another `rbln::...` symbol) | source-build, when `librbln.so` is older than the Python C extension | The build's `librbln.so` predates the addition of the symbol that `pyrbln/vmem.cc` references. Re-run `ninja librbln.so` in the rebel-compiler build dir (`source dynamic_linking.env` first so the conan-installed `rbln-ccl` libs are on `LD_LIBRARY_PATH`), then re-do the editable install. Verify with `nm -D build/librbln.so \| grep _ZN4rbln6pyvmem5debug4FreeEm`. |
| `torch._inductor.exc.InductorError` on an `rbln` tensor | source-build, on models whose runner override `_compile_model` without routing through the rebel path | The base `_compile_model` correctly dispatches `auto → rbln` on an NPU and `rbln → _compile_transformer_rbln`. However, several runner-model overrides still hard-code `torch.compile(..., mode="default")` (Inductor): `runner_models/causal_wan.py`, `runner_models/ltx.py`, `runner_models/hunyuan.py`. Those will fall to Inductor on NPU and fail; fixing requires routing each through `_compile_transformer_rbln`. `flux.py`, `stable_diffusion.py`, and the base path already do the right thing. |
| `DiagnosticError: one or more error diagnostics were emitted, please check diagnostic render for output.` (no underlying message visible) | source-build only when manually forced, **[wheel-default]** otherwise | The source build defaults `rebel.core._env.ENV = "DEV"` and surfaces the real underlying TVMError. The wheel ships `ENV = "PROD"` which routes any pipeline error through `raise_diagnostic_error_if_prod`, masking it. To surface the actual message on the wheel: `import rebel.core._env as e; e.ENV='DEV'; import rebel.core.build_module; rebel.core.build_module.ENV='DEV'` **before** `torch.compile(backend="rbln")` runs. |
| `RCCL ... failed` / multi-NPU hangs | source-build and wheel | Make sure the bootstrap env is set (`RCCL_FORCE_EXPORT_MEM=1`, `RCCL_PORT_GEN=1`, `RBLN_ROOT_IP=127.0.0.1`, `RBLN_LOCAL_IP=127.0.0.1`, `MASTER_ADDR=127.0.0.1`) in every rank before `torch_rbln` is imported. Use **tensor parallelism** (`--tensor_parallel_degree N`), not `--ulysses_degree` — `all_to_all` is not compilable into rbln-ccl yet. |
| `requires the Torchvision library` | both | Install the matching torchvision ([§2.4](#24-optional-torchvision)). |
| First run is slow | both | Compilation runs once and is cached under `<output_directory>/rbln_compile_cache/`; subsequent runs reuse the `.rbln` artifact. |

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
