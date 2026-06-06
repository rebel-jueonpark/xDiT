# Running xDiT on Rebellions ATOM™ NPU

This guide explains, step by step, how to run xDiT diffusion-transformer inference
on **Rebellions ATOM NPUs** using the `torch-rbln` backend.

> **Status:** ATOM support runs **single-NPU** today and covers the
> linear-patchify text→image DiTs (Flux, Z-Image, Qwen-Image) plus CogVideoX.
> See the [Supported models](#5-supported-models-on-atom) matrix for exactly what
> works, what is partial, and what is not yet supported — and
> [Limitations](#7-limitations) for multi-NPU status.

---

## 1. Prerequisites

### Hardware
One or more Rebellions ATOM NPUs. Verify they are visible:

```bash
rbln-stat            # lists NPUs, memory (15.7 GiB each), utilization
# rbln-stat -j       # JSON output
```

### Software stack (verified working set)
| Package        | Verified version  | Notes |
| ---            | ---               | ---   |
| Python         | 3.12              | |
| `torch`        | `2.10.0+cpu`      | CPU build; the NPU backend is provided by `torch-rbln` |
| `torch-rbln`   | `0.2.1`           | registers the `rbln` device + `rbln-ccl` distributed backend |
| `rebel-compiler` | `0.10.5`        | RBLN SDK / compiler |
| `diffusers`    | `0.37.1`          | |
| `transformers` | `5.9.0`           | |
| `yunchang`     | `0.6.4`           | required for the USP / ring attention import path |
| `torchvision`  | `0.25.0+cpu`      | *optional*, only needed by some processors (e.g. Qwen-Image-Edit). Must match the torch version. |

---

## 2. Installation

1. **Install the Rebellions SDK** (`rebel-compiler` + `torch-rbln`) following
   Rebellions' official instructions for your environment. After installation,
   confirm the backend loads:

   ```bash
   python -c "import torch, torch_rbln; print(torch.rbln.is_available())"   # -> True
   ```

2. **Install xDiT (this repository)** and its dependencies into the same
   environment:

   ```bash
   git clone https://github.com/rebel-jueonpark/xDiT.git
   cd xDiT
   git checkout adopt_atom_npu
   pip install -e .          # installs xfuser (editable) + deps
   ```

3. **(Optional) torchvision** — only if you hit a
   `requires the Torchvision library` error (some image/video processors).
   Install the version matching your torch, with no deps so it cannot disturb
   the RBLN torch build:

   ```bash
   pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torchvision==0.25.0
   ```

---

## 3. Environment setup

The RBLN runtime needs a few environment variables set **before** `torch_rbln`
is imported. The provided example scripts set these for you via
`os.environ.setdefault(...)`, but if you write your own launcher, set them first:

```bash
export RCCL_FORCE_EXPORT_MEM=1        # required by the rbln-ccl runtime
export RBLN_ROOT_IP=127.0.0.1
export RBLN_LOCAL_IP=127.0.0.1
export MASTER_ADDR=127.0.0.1

# Point at a Hugging Face cache (and token for gated models, e.g. FLUX):
export HF_HOME=/path/to/your/huggingface_cache
```

> **Import order matters.** In any custom script, set the variables above, then
> `import torch`, then `import torch_rbln` (which registers the `rbln` device and
> the `rbln-ccl` backend), **before** initializing `torch.distributed`. The
> bundled examples already do this correctly — use them as a template.

---

## 4. Quick start (text→image)

The fastest path is the **model-agnostic runner launcher**,
`examples/flux2_rbln_example.py`. It builds an `xFuserModelRunner` from the
`--model` flag and runs it. Launch it with `torchrun` (one process per NPU).

Run **FLUX.2-klein-4B** on a single NPU:

```bash
torchrun --nproc_per_node=1 --master_port=29500 \
    examples/flux2_rbln_example.py \
    --model FLUX.2-klein-4B \
    --ulysses_degree 1 \
    --height 512 --width 512 \
    --num_inference_steps 4 \
    --prompt "a futuristic city at dusk" \
    --output_directory ./outputs
```

A `.png` is written to `./outputs/`. The same launcher works for any of the
runner models in [§5](#5-supported-models-on-atom); just change `--model`:

```bash
# Flux.1-dev (12B — loads via host-shared memory, runs slower on one NPU)
torchrun --nproc_per_node=1 examples/flux2_rbln_example.py \
    --model FLUX.1-dev --ulysses_degree 1 \
    --height 512 --width 512 --num_inference_steps 4 \
    --prompt "a red panda astronaut" --output_directory ./outputs

# Z-Image
torchrun --nproc_per_node=1 examples/flux2_rbln_example.py \
    --model Z-Image --ulysses_degree 1 \
    --height 512 --width 512 --num_inference_steps 8 \
    --prompt "a watercolor fox" --output_directory ./outputs

# Qwen-Image (20B)
torchrun --nproc_per_node=1 examples/flux2_rbln_example.py \
    --model Qwen-Image --ulysses_degree 1 \
    --height 512 --width 512 --num_inference_steps 8 \
    --prompt "a bowl of ramen, studio photo" --output_directory ./outputs
```

### Common arguments
| Flag | Meaning |
| --- | --- |
| `--model` | Registry name or HF id (e.g. `FLUX.2-klein-4B`, `FLUX.1-dev`, `Z-Image`, `Qwen-Image`). |
| `--ulysses_degree` | Sequence-parallel degree. **Keep at `1`** on ATOM (see [Limitations](#7-limitations)). |
| `--height` / `--width` | Output resolution. |
| `--num_inference_steps` | Denoising steps. |
| `--num_frames` | Video frame count (video models). |
| `--prompt` | Text prompt (quote it). |
| `--seed` | RNG seed (default 42). |
| `--output_directory` | Where outputs (`.png`/`.mp4`) are written. |
| `--input_images` | Input image(s) for image-conditioned / edit models. |
| `--task` | Task selector for multi-task models (e.g. `t2v`). |

---

## 5. Supported models on ATOM

Empirically validated on this branch (single NPU). The dividing line is the
**patch-embed type**: models that patchify with a `Linear` run today; models
that patchify with a `Conv` do not yet (the NPU lacks an eager `convolution`
kernel — see [Troubleshooting](#8-troubleshooting)).

| Model | Status | Notes |
| --- | --- | --- |
| **Flux.1-dev** | ✅ works | single NPU |
| **Flux.2-klein (4B)** | ✅ works | single NPU |
| **Z-Image / Z-Image-Turbo** | ✅ works | single NPU |
| **Qwen-Image** | ✅ works | 20B; loads via host-shared memory |
| **CogVideoX (-2b)** | ✅ works | use `examples/cogvideox_rbln_example.py` (see §6) |
| **Stable Diffusion 3.5** | ❌ not yet | conv2d patch-embed (`convolution_overrideable`) |
| **Qwen-Image-Edit** | ❌ not yet | conv3d patch-embed |
| **Wan2.x / video runner models** | ❌ not yet | conv3d patch-embed |
| **Flux Kontext** | ❌ not yet | CPU/NPU device mismatch on edit conditioning |
| **PixArt, SANA, SDXL, HunyuanDiT, Latte, …** | ❌ not yet | legacy pipeline path has no RBLN adapter |

---

## 6. Running video (CogVideoX)

CogVideoX uses the legacy pipeline path with a dedicated RBLN adapter,
`examples/cogvideox_rbln_example.py`. Run on a single NPU **without** the
CPU-offload flag (manual placement keeps the conv patch-embed and VAE on CPU and
the transformer on the NPU):

```bash
export XDIT_RBLN_OUTPUT=./outputs        # where the .mp4 is written
python examples/cogvideox_rbln_example.py \
    --model THUDM/CogVideoX-2b \
    --height 480 --width 720 --num_frames 9 \
    --num_inference_steps 4 \
    --prompt "a tiny astronaut riding a horse on the moon" \
    --seed 42
```

The `.mp4` is written to `$XDIT_RBLN_OUTPUT`. Avoid `--enable_model_cpu_offload`
here — it moves the conv patch-embed back onto the NPU and conflicts with the
adapter's manual CPU placement.

---

## 7. Limitations

- **Single NPU only (for now).**
  - **Sequence Parallelism** (`--ulysses_degree > 1`) currently fails — the
    all-to-all polyfill's underlying `rbln-ccl` AllGather errors. Keep
    `--ulysses_degree 1` and `--nproc_per_node 1`.
  - **FSDP weight sharding** (`--fully_shard_degree > 1`) is not yet RBLN-ready
    (its placement path uses CUDA device literals).
- **Memory is not the wall.** RBLN backs weights with host shared memory, so
  large models (12B Flux.1-dev, 20B Qwen-Image) *load and run* on a single
  15.7 GiB NPU — but unsharded, so latency for big models is high.
- **Conv-based models are not yet supported** (SD3.5, Qwen-Image-Edit, the Wan
  video family, and the standalone VAE conv path) — see Troubleshooting.
- **bf16** is the working dtype.

---

## 8. Troubleshooting

| Error / symptom | Cause & fix |
| --- | --- |
| `convolution_overrideable not implemented` | The model uses a **Conv** patch-embed/VAE; `torch-rbln` has no eager conv kernel yet. Affects SD3.5, Qwen-Image-Edit, Wan/video. Not user-fixable from xDiT — needs a backend `convolution_overrideable` registration in `torch-rbln`. Use a linear-patchify model meanwhile. |
| `RCCL AllGather failed with error code: 1` | You set `--ulysses_degree > 1`. Multi-NPU sequence parallelism is not working yet — use `--ulysses_degree 1` and `--nproc_per_node 1`. |
| `Cannot get CUDA generator without ATen_cuda library` | An older build without the CPU-generator fix. Update to this branch (`adopt_atom_npu`), which routes RNG generators through a device-aware helper. |
| `Unsafe cast: input has dtype torch.float32 but ... torch.bfloat16` | Missing the scheduler-dtype fix. Update to this branch (the fix is centralized for all runner models). |
| `requires the Torchvision library` | Install the matching torchvision (see [§2](#2-installation), step 3). |
| `cat: all inputs must be on the same device, got rbln:0 and cpu` | Image-conditioned/edit model (e.g. Kontext): the CPU-VAE conditioning latents need to be moved back to the NPU before concatenation. Not yet handled. |
| `Address already in use` / rendezvous errors | A previous `torchrun` is still holding the port. Pick a new `--master_port`, or kill stale processes. |
| Model weights won't download / `gated` / `401`,`403` | Set `HF_HOME` to a cache with a valid token, and accept the model's license on Hugging Face. |

---

## 9. Writing your own launcher

If you need a custom entry point, mirror `examples/flux2_rbln_example.py`:

```python
import os
os.environ.setdefault("RCCL_FORCE_EXPORT_MEM", "1")
os.environ.setdefault("RBLN_ROOT_IP", "127.0.0.1")
os.environ.setdefault("RBLN_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

import torch            # noqa: E402
import torch_rbln       # noqa: F401,E402  (registers torch.rbln + rbln-ccl)

from xfuser.config.args import FlexibleArgumentParser, xFuserArgs
from xfuser.runner import xFuserModelRunner, setup_logging

def main():
    setup_logging()
    parser = FlexibleArgumentParser(description="xDiT on Rebellions NPU")
    args = vars(xFuserArgs.add_runner_args(parser).parse_args())
    runner = xFuserModelRunner(args)
    input_args = runner.preprocess_args(args)
    runner.initialize(input_args)
    output, timings = runner.run(input_args)
    runner.save(output=output, timings=timings)
    runner.cleanup()

if __name__ == "__main__":
    main()
```

Launch it with `torchrun --nproc_per_node=1 your_launcher.py --model ... --ulysses_degree 1 ...`.

---

## 10. What the branch changes (for reference)

The `adopt_atom_npu` branch adds, on top of upstream xDiT:
- `rbln` device + `rbln-ccl` backend detection (`xfuser/envs.py`,
  `parallel_state.py`), device-agnostic utilities (`xfuser/core/device_utils.py`),
  and a collective-ops polyfill (`xfuser/core/rbln_collectives.py`).
- Runner-model fixes: device-aware RNG generators and centralized
  scheduler-dtype / VAE-on-CPU handling so every runner model gets the RBLN
  treatment (`xfuser/model_executor/models/runner_models/`).
- RBLN examples: `examples/flux2_rbln_example.py`, `examples/cogvideox_rbln_example.py`.
