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

> **Status — work in progress (Stage 1a).** The compile backend is wired into xDiT
> and the foundational ops are verified compiling on one NPU (conv2d/conv3d and
> bf16 SDPA). Full end-to-end model bring-up and multi-NPU TP are still in
> progress — see [§6 Status & roadmap](#6-status--roadmap) for exactly what works
> and what's pending.

---

## 1. Prerequisites

### Hardware
One or more Rebellions ATOM NPUs:

```bash
rbln-stat            # lists NPUs (15.7 GiB each), utilization
```

### Software stack (verified set)
| Package          | Version       | Role |
| ---              | ---           | ---  |
| Python           | 3.12          | |
| `torch`          | `2.10.0+cpu`  | CPU build; NPU support comes from the rebel/torch-rbln stack |
| `torch-rbln`     | `0.2.1`       | registers the `rbln` device + `rbln-ccl` distributed backend |
| `rebel-compiler` | `0.10.5`      | **provides the `torch.compile` backend `"rbln"`** (this branch's compute path) |
| `diffusers`      | `0.37.1`      | |
| `transformers`   | `5.9.0`       | |
| `torchvision`    | `0.25.0+cpu`  | optional; only some processors need it (must match the torch version) |

The conv + collective lowering all happen inside `rebel-compiler`'s
`torch.compile` backend; `torch-rbln` supplies the `rbln` device and the
`rbln-ccl` process-group backend.

---

## 2. Installation

1. **Install the Rebellions SDK** (`rebel-compiler` + `torch-rbln`) per Rebellions'
   instructions, then confirm both the device and the compile backend are present:

   ```bash
   python - <<'PY'
   import torch, torch_rbln
   from rebel.core.torch_compile import rbln_backend   # the torch.compile backend
   print("rbln device:", torch.rbln.is_available())
   PY
   ```

2. **Install xDiT (this branch):**

   ```bash
   git clone https://github.com/rebel-jueonpark/xDiT.git
   cd xDiT
   git checkout adopt_atom_npu_compile
   pip install -e .
   ```

3. **(Optional) torchvision** — only if a processor raises
   `requires the Torchvision library`:

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
| **1b — single-NPU full model** | compile a whole DiT transformer end-to-end | 🚧 in progress (see blockers) |
| **2 — multi-NPU FFN-TP** | FeedForward tensor parallelism, collectives via rbln-ccl | ⏳ planned |
| **3 — attention TP** | Megatron-style column/row-parallel attention | ⏳ planned |

**Current bring-up blockers (Stage 1b):**
- **Flux / Flux2 / Qwen-Image** — rebel traces the full DiT but conversion fails on
  two unimplemented ops: **`aten::outer`** (RoPE) and **`aten::rms_norm`** (QK-norm).
  These need rebel-compiler converters/decompositions, or model-level decomposition.
- **SD3.5** — uses neither of those ops (only conv2d), but its transformer forward
  currently routes to Inductor instead of the rebel backend; that routing needs
  fixing before it compiles on the NPU.

Because rebel supports conv, the eventual model coverage on this path is broader
than the eager branch (it should include the conv-patchify DiTs and VAEs that
eager cannot run) — pending the items above.

---

## 7. Troubleshooting

| Error / symptom | Cause & fix |
| --- | --- |
| `NotImplementedError: The following operators are not implemented: ['aten::outer', 'aten::rms_norm']` | The DiT uses RoPE/QK-RMSNorm ops the rebel converter lacks (Flux/Qwen family). Pending op support — see [§6](#6-status--roadmap). |
| `torch._inductor.exc.InductorError` on an `rbln` tensor | The compile fell to **Inductor** instead of the rebel backend. Ensure `--torch_compile_backend rbln` (or `auto` on an NPU) and that the model's `_compile_model` routes through `_compile_transformer_rbln`. |
| `DiagnosticError: one or more error diagnostics were emitted` | A rebel-compiler lowering error for some op/shape. Run with rebel debug logging to see the op; report it as a missing converter. |
| `RCCL ... failed` / multi-NPU hangs | Ensure `RCCL_PORT_GEN=1` and the bootstrap env are set; use **tensor parallelism** (`--tensor_parallel_degree N`), not `--ulysses_degree` (`all_to_all` isn't compilable). |
| `requires the Torchvision library` | Install the matching torchvision ([§2](#2-installation) step 3). |
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

It builds on the RBLN device/backend enablement from `adopt_atom_npu` (the `rbln`
device, `rbln-ccl` selection, and the scheduler-dtype/VAE-on-CPU compat hooks).
