import types
import weakref

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

from xfuser.core.distributed import get_cfg_group


def _cfg_shard(value, batch_size: int, rank: int):
    """Return this CFG rank's half of every tensor that carries the UNet batch."""
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == batch_size:
            half = batch_size // 2
            return value[rank * half : (rank + 1) * half]
        return value
    if isinstance(value, dict):
        return {key: _cfg_shard(item, batch_size, rank) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cfg_shard(item, batch_size, rank) for item in value)
    return value


def unet_cfg_parallel_monkey_patch_forward(
    self,
    sample: torch.Tensor,
    timestep,
    encoder_hidden_states: torch.Tensor,
    *args,
    return_dict: bool = True,
    **kwargs,
):
    """Run the UNet on one CFG half of the batch and gather both halves.

    Diffusers stacks the batch as ``[unconditional, conditional]`` when
    classifier-free guidance is on, so each of the two CFG ranks runs one half
    and the halves are gathered back in that order. Without CFG the batch holds
    only conditional samples and every rank runs the whole batch.
    """
    original_forward = type(self).forward
    batch_size = sample.shape[0]
    pipeline = self._xfuser_cfg_pipeline()
    split = (
        pipeline is not None
        and getattr(pipeline, "_guidance_scale", None) is not None
        and pipeline.do_classifier_free_guidance
    )
    if not split:
        return original_forward(self, sample, timestep, encoder_hidden_states, *args, return_dict=return_dict, **kwargs)

    cfg_group = get_cfg_group()
    rank = cfg_group.rank_in_group
    output = original_forward(
        self,
        _cfg_shard(sample, batch_size, rank),
        _cfg_shard(timestep, batch_size, rank),
        _cfg_shard(encoder_hidden_states, batch_size, rank),
        *_cfg_shard(args, batch_size, rank),
        return_dict=False,
        **_cfg_shard(kwargs, batch_size, rank),
    )[0]
    output = cfg_group.all_gather(output.contiguous(), dim=0)

    if return_dict:
        return UNet2DConditionOutput(sample=output)
    return (output,)


def _cfg_split_active(unet) -> bool:
    pipeline = unet._xfuser_cfg_pipeline()
    return (
        pipeline is not None
        and getattr(pipeline, "_guidance_scale", None) is not None
        and pipeline.do_classifier_free_guidance
    )


def _cfg_shard_pre_hook(unet, args, kwargs):
    """Forward pre-hook: hand the UNet this CFG rank's half of every batched input."""
    unet._xfuser_cfg_split = _cfg_split_active(unet)
    if not unet._xfuser_cfg_split:
        return None
    from xfuser.core.rbln_compile import own_buffers

    sample = args[0] if args else kwargs["sample"]
    batch_size = sample.shape[0]
    rank = get_cfg_group().rank_in_group
    # A half is a view into the full batch; a compiled NPU program needs each input in a
    # buffer of its own (rank 1's half starts mid-allocation).
    return own_buffers(_cfg_shard(args, batch_size, rank)), own_buffers(_cfg_shard(kwargs, batch_size, rank))


def _cfg_gather_hook(unet, args, kwargs, output):
    """Forward hook: gather both ranks' halves back into the full CFG batch."""
    if not getattr(unet, "_xfuser_cfg_split", False):
        return None
    sample = output[0] if isinstance(output, tuple) else output.sample
    gathered = get_cfg_group().all_gather(sample.contiguous(), dim=0)
    if isinstance(output, tuple):
        return (gathered,) + tuple(output[1:])
    return UNet2DConditionOutput(sample=gathered)


def apply_unet_cfg_parallel_hooks(pipe):
    """CFG parallel through module hooks instead of a replaced ``forward``.

    Same split and gather as the monkey patch, but ``unet.forward`` stays the module's own:
    the RBLN backend compiles exactly ``module.forward = torch.compile(module.forward)`` and
    would otherwise have to trace the collective and the pipeline lookup with it.
    """
    pipe.unet._xfuser_cfg_pipeline = weakref.ref(pipe)
    pipe.unet.register_forward_pre_hook(_cfg_shard_pre_hook, with_kwargs=True)
    pipe.unet.register_forward_hook(_cfg_gather_hook, with_kwargs=True)
    return pipe


def apply_unet_cfg_parallel_monkey_patch(pipe):
    """Split the UNet batch across the two ranks of the CFG parallel group."""
    import xfuser.envs as envs

    if envs._is_rbln():
        return apply_unet_cfg_parallel_hooks(pipe)
    pipe.unet._xfuser_cfg_pipeline = weakref.ref(pipe)
    pipe.unet.forward = types.MethodType(unet_cfg_parallel_monkey_patch_forward, pipe.unet)
    return pipe
