# xDiT on Rebellions ATOM NPUs

xDiT runs diffusion transformers on Rebellions ATOM NPUs through two pieces of the
Rebellions stack:

- **torch-rbln** registers the `rbln` device (`torch.rbln`, tensors on `rbln:N`) and the
  `rbln-ccl` `torch.distributed` backend (all-reduce, all-gather, broadcast, send/recv,
  reduce-scatter over the NPU interconnect).
- **rebel-compiler** serves `torch.compile(backend="rbln")`: the transformer graph is
  compiled ahead of time for the NPU with static shapes, and the in-graph c10d
  collectives (`all_reduce`, `all_gather_into_tensor`, `reduce_scatter_tensor`) are
  lowered onto rbln-ccl, so a parallel transformer runs as one device program per step.

Eager execution of a whole DiT on torch-rbln is not the goal here: eager mode has no
convolution (the patch-embed and the VAE are convolutions) and runs each op as its own
kernel. With `--use_torch_compile` the runner compiles the transformer and the VAE decode
path; everything else (schedulers, CFG arithmetic, text encoders) stays eager.

## Install

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install rebel-compiler torch-rbln --extra-index-url <rebellions package index>
uv pip install -e .            # xDiT
```

`torch-rbln` pins the `torch` and `rebel-compiler` versions it was built against; let
pip resolve them together. Do not export `TVM_HOME` / `REBEL_HOME` from a compiler
source checkout in the same shell: the wheel honours `TVM_HOME` and would load that
tree's `librbln.so`. `examples/rbln/env.sh` unsets them. With a source-built compiler
installed editable into the venv (`uv pip install --no-deps -e <rebel_compiler>/python`)
those variables are the point: set `XDIT_RBLN_SOURCE_COMPILER=1` and env.sh keeps them.

## Run

```bash
source examples/rbln/env.sh            # rbln-ccl bootstrap variables, HF cache
NPROC=1 examples/rbln/run_sd3.sh                                  # one NPU
NPROC=2 examples/rbln/run_sd3.sh --use_cfg_parallel               # CFG parallel
NPROC=2 examples/rbln/run_sd3.sh --ulysses_degree 2               # Ulysses sequence parallel
NPROC=4 examples/rbln/run_sd3.sh --use_cfg_parallel --ulysses_degree 2
TEXT_ENCODER_DEVICE=spare NPROC=4 examples/rbln/run_sd3.sh --use_cfg_parallel --ulysses_degree 2   # T5 on a 5th NPU
```

The launcher is `torchrun -m xfuser.runner --use_torch_compile --torch_compile_backend
rbln ...`; `RBLN_DEVICES` picks the NPUs (torch-rbln's `CUDA_VISIBLE_DEVICES`).
`--torch_compile_backend auto` (the default) already selects `rbln` when an NPU is
present.

Runner flags specific to this platform:

| flag | default | meaning |
|---|---|---|
| `--torch_compile_backend {auto,inductor,rbln}` | `auto` | `rbln` on an NPU, `inductor` elsewhere |
| `--rbln_text_encoder_device {cpu,rbln,spare}` | `cpu` | SD3's T5-XXL (~9.5 GB bf16) does not fit next to the transformer in one ATOM's 16 GB, so the text encoders run on the host by default. `spare` compiles them for one NPU beyond the ranks' own: rank 0 encodes there and broadcasts the embeddings. The launcher exposes that NPU (`RBLN_DEVICES=0,...,NPROC`) when `TEXT_ENCODER_DEVICE=spare` |
| `--rbln_spare_device N` | local world size | the visible NPU index `spare` uses |
| `--rbln_vae_device {rbln,cpu}` | `rbln` | compile the VAE decoder for the NPU, or decode on the host |

Set `XDIT_STAGE_TIMING=1` to log the time spent in prompt encoding, the transformer
calls and VAE decode of every run (written to `stage_timing_rank<N>.jsonl` in the
output directory).

With `spare`, T5 is compiled as its 24 blocks only (`xfuser/core/rbln_text_encoders.py`):
the token embedding is looked up on the host and the relative-position bias, which depends
only on the static sequence length, is a constant buffer of the program. Compiling the whole
`T5EncoderModel` traces, but the program failed at its first run in the rebel runtime
("Transpose axes size mismatch") on the bias' gather-then-permute. The CLIP encoders compile
whole, hidden states included. On the NPU the encoders take 0.12 s (T5) and under 0.01 s
(each CLIP) per prompt where the host took 0.7 s and 0.03 to 0.1 s.

## Parallelism on ATOM

| xDiT parallelism | status | how it runs |
|---|---|---|
| CFG parallel (`--use_cfg_parallel`) | supported | the two guidance branches run on two NPUs; the noise predictions are gathered eagerly through rbln-ccl |
| Ulysses sequence parallel (`--ulysses_degree`) | supported | the attention all-to-all is expressed as an in-graph all-gather plus a static slice (`xfuser/model_executor/layers/usp.py`); the compiler lowers it onto rbln-ccl and splits the transformer program into device segments around each collective (SD3: 4 exchanges per block, 96 per step) |
| Data parallel (`--data_parallel_degree`) | supported | independent replicas |
| Ring attention (`--ring_degree`) | not supported | needs point-to-point exchange inside the attention kernel loop; not compilable |
| PipeFusion (`--pipefusion_parallel_degree`) | not supported yet | its patch-level KV cache and per-patch stage loop are dynamic; no static-shape program |
| Tensor parallel (`--tensor_parallel_degree`) | not supported | xDiT's TP layers are CUDA-only; use the compiler's own sharding instead (below) |

**Compiler-side tensor parallelism (RSD).** The rebel compiler can shard one compiled
program across several NPUs itself. torch-rbln exposes it per process: with
`RBLN_NPUS_PER_DEVICE=2` every logical `rbln:N` device is two physical NPUs and the
backend compiles each graph with `num_devices=2`; `rbln_compile.py` then hands the
runtime the rank's physical device list. It composes with xDiT's multi-process
parallelism (two CFG ranks, each over two NPUs), but measured on SD3-medium the sharded
batch-1 transformer was slower than one NPU (0.62 vs 0.56 s per call) and the eager glue
between programs slowed down a lot, so prefer xDiT's own degrees first.

## Writing code the rbln backend can compile

These rules came out of bringing SD3 up; each one cost a run.

- No untraceable call inside the compiled forward. `envs._is_rbln()` reaches torch-rbln's
  C-level `is_available()`, and `get_cache_manager().has_cache_entry(layer)` is a dict
  lookup keyed by a module: Dynamo cannot trace either, and a graph break raised inside a
  nested call cannot be resumed, so Dynamo silently drops the *whole* forward to eager
  (where torch-rbln then runs it op by op and fails on the first convolution). Evaluate
  platform checks once at import time; gate PipeFusion-only lookups on the pipeline degree.
- `torch.compiler.disable` deep inside the attention path has the same effect; a collective
  that must run eagerly has to be at the top level of the compiled function.
- No `Tensor.view(dtype)`: the rebel converter handles `view` as a shape view and fails on
  the dtype argument ("expected Object but got int" from `InferTypeLocal`).
- A compiled program's device inputs must be contiguous and start at the base of their
  allocation; an offset view (`timesteps[i].expand(1)` at step 2) is an unknown device key
  at run time (`RUN_INTERNAL (Invalid key_vaddr=...)`). `xfuser/core/rbln_compile.py`
  clones such inputs in a forward pre-hook.
- Keep `module.forward = torch.compile(module.forward)` exactly: wrapping the compiled
  forward in another function (for timing, say) makes Dynamo treat the module as dynamic,
  and its weights stop being graph constants. Time with forward hooks instead.
- No module-attribute side effects in the traced forward: the Conv2d wrapper's
  `self.activation_cache = x` made the graph input an output alias.

## Measured: SD3-medium, 1024×1024, 28 steps, guidance 7

Seconds per transformer call (batch 2 for CFG) and per image, mean of two timed images
after a warm-up, one host with RBLN-CA25 NPUs, rebel-compiler 0.12.1.dev79+gc6c3fb6d
built from source at that commit (the next dev commit, 0.12.1.dev80+g040db57c, was
rebuilt and re-checked on one NPU: same per-step time, bit-identical image). The optimum-rbln rows are the compiler's own RSD
tensor parallelism for comparison (its T5 sits on one extra NPU; xDiT's text encoders
run on the host, one rank encodes and broadcasts). The "+ spare" rows put xDiT's text
encoders on one extra NPU too (`TEXT_ENCODER_DEVICE=spare`, section above): the same
per-step time, and 0.3 to 0.4 s of prompt encoding per image instead of 1.7 to 2.4 s.

| | NPUs | s / step | vs 1 NPU | s / image |
|---|---|---|---|---|
| optimum-rbln RSD tp1 | 1 | 1.089 | 1.00× | 31.4 |
| xDiT, 1 NPU | 1 | 1.089 | 1.00× | 33.4 |
| optimum-rbln RSD tp2 | 2 | 0.582 | 1.87× | 17.2 |
| xDiT CFG parallel ×2 | 2 | 0.556 | 1.96× | 19.1 |
| xDiT Ulysses ×2 | 2 | 0.656 | 1.66× | 21.2 |
| optimum-rbln RSD tp4 | 4 | 0.434 | 2.51× | 13.1 |
| xDiT CFG ×2 · Ulysses ×2 | 4 | 0.392 | 2.78× | 13.7 |
| xDiT Ulysses ×4 | 4 | 0.399 | 2.73× | 14.2 |
| xDiT, 1 NPU + spare | 1 + 1 | 1.088 | 1.00× | 31.6 |
| xDiT CFG parallel ×2 + spare | 2 + 1 | 0.557 | 1.96× | 17.1 |
| xDiT Ulysses ×2 + spare | 2 + 1 | 0.657 | 1.66× | 19.8 |
| xDiT CFG ×2 · Ulysses ×2 + spare | 4 + 1 | 0.392 | 2.78× | 12.4 |
| xDiT Ulysses ×4 + spare | 4 + 1 | 0.400 | 2.72× | 12.4 |

## Known limits

- Static shapes: a compiled program is specific to the resolution, batch (CFG doubles
  it) and prompt length. Each new combination compiles once and is cached under
  `~/.cache/xfuser/rbln_compile_cache` (`XDIT_RBLN_CACHE_DIR`).
- `--enable_tiling` on the VAE bypasses the compiled decode path and fails (no eager
  convolution); use `--rbln_vae_device cpu` with tiling.
- Event timing: torch-rbln has no `Event.elapsed_time`, so iteration times are wall
  clock around device syncs.
