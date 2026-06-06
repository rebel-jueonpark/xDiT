"""Run CogVideoX-2b / CogVideoX-5b on Rebellions NPU via xDiT (legacy pipeline path).

Launch examples:

  # Single-NPU smoke (model with text encoder + VAE offloaded to CPU)
  python examples/cogvideox_rbln_example.py \
      --model THUDM/CogVideoX-2b --enable_model_cpu_offload \
      --height 480 --width 720 --num_frames 9 --num_inference_steps 4 \
      --prompt "a tiny astronaut riding a horse on the moon" --seed 42 \
      --output_directory /home/jueonpark/xdit-rbln/outputs

  # Multi-NPU data-parallel (one video per prompt per NPU)
  torchrun --nproc_per_node=2 examples/cogvideox_rbln_example.py \
      --model THUDM/CogVideoX-2b --data_parallel_degree 2 \
      --enable_model_cpu_offload \
      --height 480 --width 720 --num_frames 9 --num_inference_steps 4 \
      --prompt "..." "..." --seed 42 \
      --output_directory /home/jueonpark/xdit-rbln/outputs
"""

import os
import sys

# RBLN distributed bootstrap (must precede torch_rbln C++ loading)
os.environ.setdefault("RCCL_FORCE_EXPORT_MEM", "1")
os.environ.setdefault("RBLN_ROOT_IP", "127.0.0.1")
os.environ.setdefault("RBLN_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("HF_HOME", "/mnt/shared_data/groups/sw_dev/.cache/huggingface")

import time
import logging
import functools
import torch
import torch.distributed
import torch_rbln  # noqa: F401  (registers torch.rbln + rbln-ccl)

from diffusers.utils import export_to_video
from xfuser import xFuserCogVideoXPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    is_dp_last_group,
)
from xfuser.envs import get_device_name


def _wrap_scheduler_step_inputs(scheduler, device) -> None:
    """Move all tensor inputs of the underlying scheduler.step to `device`.
    rbln 0.2.x's strict same-device check fires if even one tensor is CPU."""
    if scheduler is None:
        return
    # xfuser wraps the scheduler with a delegating `__setattr__`; the actual
    # diffusers scheduler lives at `.module`.
    target = getattr(scheduler, "module", scheduler)
    if getattr(target, "_xfuser_step_wrapped", False):
        return
    original = target.step.__func__ if hasattr(target.step, "__func__") else target.step

    def _move_to(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    def patched(self, *args, **kwargs):
        args = tuple(_move_to(a) for a in args)
        kwargs = {k: _move_to(v) for k, v in kwargs.items()}
        return original(self, *args, **kwargs)

    # Bind as a method on the underlying scheduler.
    import types
    object.__setattr__(target, "step", types.MethodType(patched, target))
    target._xfuser_step_wrapped = True


def _match_scheduler_dtype_and_device(pipe, dtype, device) -> None:
    """Cast scheduler buffers to `dtype` and move them to `device`, and wrap
    `set_timesteps` so the cast survives re-initialization inside `pipe(...)`."""
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None:
        return

    def _recast() -> None:
        for attr in (
            "sigmas", "timesteps", "betas", "alphas",
            "alphas_cumprod", "final_alpha_cumprod",
            "sigmas_interpol", "log_sigmas",
        ):
            buf = getattr(scheduler, attr, None)
            if isinstance(buf, torch.Tensor):
                if buf.is_floating_point():
                    setattr(scheduler, attr, buf.to(dtype).to(device))
                else:
                    setattr(scheduler, attr, buf.to(device))

    _recast()
    original = scheduler.set_timesteps

    @functools.wraps(original)
    def patched(*a, **kw):
        out = original(*a, **kw)
        _recast()
        return out

    scheduler.set_timesteps = patched


def _vae_to_cpu(pipe) -> None:
    """torch-rbln 0.2.x eager mode lacks convolution_overrideable, so the
    decode/encode of the VAE has to land on CPU. The VAE is the bottleneck
    but happens once per video (not per step) so the wall-clock impact is
    relatively small."""
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    vae.to("cpu")

    if not getattr(vae, "_xfuser_decode_wrapped", False):
        original_decode = vae.decode

        @functools.wraps(original_decode)
        def patched_decode(z, *a, **kw):
            return original_decode(z.to("cpu").to(vae.dtype), *a, **kw)

        vae.decode = patched_decode
        vae._xfuser_decode_wrapped = True


def main():
    parser = FlexibleArgumentParser(description="CogVideoX on Rebellions NPU")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for CogVideo"

    pipe = xFuserCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    dev_name = get_device_name()
    dev_str = f"{dev_name}:{local_rank}" if dev_name != "cpu" else "cpu"
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(device=dev_str)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(device=dev_str)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        # Manual placement: T5 + VAE stay on CPU (T5 is huge; VAE uses conv
        # which isn't in torch-rbln eager). Only the DiT transformer lands on
        # NPU. This trades T5 latency for not blowing past 15.7 GB.
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder.to("cpu")
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to("cpu")
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            pipe.transformer.to(dev_str)

        # diffusers' `_execution_device` walks the pipeline modules and picks
        # the one device; with text_encoder on CPU it would pick "cpu" and
        # then create latents/timesteps on CPU. Force it to the NPU since
        # that's where the transformer (the bottleneck) lives.
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline as _DP

        def _fixed_execution_device(self):
            return torch.device(dev_str)

        type(pipe)._execution_device = property(_fixed_execution_device)

    _match_scheduler_dtype_and_device(pipe, torch.bfloat16, dev_str)
    _wrap_scheduler_step_inputs(pipe.scheduler, dev_str)
    _vae_to_cpu(pipe)

    # Wrap transformer.forward so its inputs (which may come from a CPU-resident
    # text encoder) are all moved to the transformer's device.
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None and not getattr(transformer, "_xfuser_inputs_wrapped", False):
        original_t_forward = transformer.forward
        t_device = next(transformer.parameters()).device

        def _move_to(x, dev):
            if isinstance(x, torch.Tensor):
                return x.to(dev)
            if isinstance(x, (list, tuple)):
                return type(x)(_move_to(e, dev) for e in x)
            if isinstance(x, dict):
                return {k: _move_to(v, dev) for k, v in x.items()}
            return x

        @functools.wraps(original_t_forward)
        def patched_transformer_forward(*args, **kwargs):
            args = tuple(_move_to(a, t_device) for a in args)
            kwargs = {k: _move_to(v, t_device) for k, v in kwargs.items()}
            return original_t_forward(*args, **kwargs)

        transformer.forward = patched_transformer_forward
        transformer._xfuser_inputs_wrapped = True

    # CogVideoX transformer's `patch_embed` uses Conv2d. torch-rbln eager
    # does not implement `convolution_overrideable` yet, so keep that one
    # submodule on CPU and round-trip the activations.
    if transformer is not None and hasattr(transformer, "patch_embed"):
        pe = transformer.patch_embed
        pe.to("cpu")
        if not getattr(pe, "_xfuser_patch_embed_wrapped", False):
            original_forward = pe.forward
            tdtype = pe.proj.weight.dtype if hasattr(pe, "proj") else torch.bfloat16

            @functools.wraps(original_forward)
            def patched_forward(*args, **kwargs):
                target_dev = None
                cpu_args = []
                for a in args:
                    if isinstance(a, torch.Tensor):
                        if target_dev is None:
                            target_dev = a.device
                        cpu_args.append(a.to("cpu").to(tdtype))
                    else:
                        cpu_args.append(a)
                cpu_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        if target_dev is None:
                            target_dev = v.device
                        cpu_kwargs[k] = v.to("cpu").to(tdtype)
                    else:
                        cpu_kwargs[k] = v
                out = original_forward(*cpu_args, **cpu_kwargs)
                if target_dev is not None and isinstance(out, torch.Tensor):
                    out = out.to(target_dev)
                return out

            pe.forward = patched_forward
            pe._xfuser_patch_embed_wrapped = True

    if args.enable_tiling:
        pipe.vae.enable_tiling()
    if args.enable_slicing:
        pipe.vae.enable_slicing()

    # warmup
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=1,
        generator=torch.Generator(device="cpu").manual_seed(input_config.seed),
    ).frames[0]

    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(input_config.seed),
    ).frames[0]
    elapsed_time = time.time() - start_time

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}"
    )
    output_dir = os.environ.get("XDIT_RBLN_OUTPUT", "/home/jueonpark/xdit-rbln/outputs")
    os.makedirs(output_dir, exist_ok=True)
    if is_dp_last_group():
        resolution = f"{input_config.width}x{input_config.height}"
        output_filename = f"{output_dir}/cogvideox_{parallel_info}_{resolution}_rank{local_rank}.mp4"
        export_to_video(output, output_filename, fps=8)
        print(f"[rank {local_rank}] output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        peak = torch.rbln.max_memory_allocated() if hasattr(torch, "rbln") else 0
        print(f"epoch time: {elapsed_time:.2f} sec, peak rbln mem: {peak/1e9:.2f} GB")

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
