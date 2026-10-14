"""Text encoders of a DiT pipeline on a spare Rebellions NPU.

A DiT's text encoders do not fit next to the transformer on one ATOM (SD3's T5-XXL alone
is 9.5 GB in bf16), so the runner keeps them on the host by default and pays 1.7 to 2.7 s
of CPU time per image. optimum-rbln instead gives T5 a whole NPU of its own. This module
does the same for xDiT: rank 0 compiles the encoder for one NPU beyond the ones the ranks
use and the embeddings are broadcast from there (``_share_prompt_encoding`` in the runner).

T5 is compiled as its 24 blocks only. The token embedding is a gather and the relative
position bias is a gather followed by a permute; the rebel runtime rejected the latter at
run time ("Transpose axes size mismatch"), and neither needs the NPU: the embedding lookup
is a 256-row read on the host and the bias depends only on the (static) sequence length, so
it becomes a constant buffer of the compiled program.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
from torch import nn

from xfuser.core.rbln_compile import compile_forward
from xfuser.logger import init_logger

logger = init_logger(__name__)


def move(obj, device: torch.device):
    """``obj.to(device)`` over tensors and containers; NPU-to-NPU copies go through the host."""
    if isinstance(obj, torch.Tensor):
        if obj.device == device:
            return obj
        if obj.device.type == "rbln" and device.type == "rbln":
            obj = obj.to("cpu")
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        moved = [move(o, device) for o in obj]
        return type(obj)(*moved) if hasattr(obj, "_fields") else type(obj)(moved)
    if isinstance(obj, dict) and hasattr(obj, "to_tuple"):
        # transformers / diffusers ModelOutput: a dict subclass that also indexes by
        # position, so it has to come back as the same class, not a plain dict.
        return type(obj)(**{k: move(v, device) for k, v in obj.items()})
    if isinstance(obj, dict):
        return type(obj)((k, move(v, device)) for k, v in obj.items())
    return obj


def first_tensor_device(args, kwargs) -> Optional[torch.device]:
    for obj in list(args) + list(kwargs.values()):
        if isinstance(obj, torch.Tensor):
            return obj.device
    return None


def pin_module_to_device(module: nn.Module, device) -> None:
    """Run ``module`` on ``device`` no matter where its inputs come from.

    Wraps ``forward`` once: every tensor argument is moved to ``device`` and the tensors in
    the result go back to the device of the first tensor argument, which is what diffusers'
    pipelines expect (they pass inputs on ``_execution_device`` and use the result there).
    """
    device = torch.device(device)
    if getattr(module, "_xfuser_pinned_device", None) == device:
        return
    module.to(device)
    original_forward = module.forward

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        caller_device = first_tensor_device(args, kwargs)
        out = original_forward(*move(args, device), **move(kwargs, device))
        if caller_device is None or caller_device == device:
            return out
        return move(out, caller_device)

    module.forward = forward
    module._xfuser_pinned_device = device


class T5EncoderBlocks(nn.Module):
    """A T5 encoder stack minus its embedding lookup, for a fixed sequence length.

    ``forward`` takes the token embeddings and returns the final hidden state; the
    relative-position bias every block shares is precomputed once for ``seq_len`` and
    stored as a buffer, so the compiled program holds no gather.
    """

    def __init__(self, encoder: nn.Module, seq_len: int):
        super().__init__()
        self.block = encoder.block
        self.final_layer_norm = encoder.final_layer_norm
        attention = encoder.block[0].layer[0].SelfAttention
        with torch.no_grad():
            bias = attention.compute_bias(seq_len, seq_len).to(encoder.final_layer_norm.weight.dtype)
        self.register_buffer("position_bias", bias.contiguous())

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden = inputs_embeds
        for block in self.block:
            hidden = block.layer[0](hidden, position_bias=self.position_bias)[0]
            hidden = block.layer[1](hidden)
        return self.final_layer_norm(hidden)


def place_t5_encoder_on_npu(t5: nn.Module, device_index: int, seq_len: int, cache_dir: Optional[str] = None) -> None:
    """Compile ``t5`` (a ``T5EncoderModel``) for NPU ``device_index`` and route its ``forward`` there.

    The embedding table stays on the host; the blocks, the final norm and the position bias
    move to the NPU and are compiled with the rebel compiler on first use. The result comes
    back on the caller's device as a ``BaseModelOutput`` so ``text_encoder_3(ids)[0]`` keeps
    working. Only ``input_ids`` of exactly ``seq_len`` tokens are served: that is the static
    shape the program is compiled for, and the only call a diffusers pipeline makes.
    """
    from transformers.modeling_outputs import BaseModelOutput

    if getattr(t5, "_xfuser_rbln_spare_device", None) is not None:
        return
    device = torch.device("rbln", device_index)
    t5.to("cpu")
    blocks = T5EncoderBlocks(t5.encoder, seq_len).to(device)
    compile_forward(blocks, cache_dir, device=device_index)
    embed_tokens = t5.encoder.embed_tokens  # shared with t5.shared; stays on the host

    def forward(input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("The spare-NPU T5 encoder needs input_ids.")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("The spare-NPU T5 encoder is compiled without a padding mask (as SD3 calls it).")
        if input_ids.shape[-1] != seq_len:
            raise ValueError(f"The spare-NPU T5 encoder is compiled for {seq_len} tokens, got {input_ids.shape[-1]}.")
        caller_device = input_ids.device
        with torch.no_grad():
            embeds = embed_tokens(input_ids.to("cpu")).to(device)
            hidden = blocks(embeds)
        return BaseModelOutput(last_hidden_state=move(hidden, caller_device))

    t5.forward = forward
    t5._xfuser_rbln_spare_blocks = blocks
    t5._xfuser_rbln_spare_device = device
    logger.info(f"rbln: T5 encoder blocks placed on {device} (embedding on the host, {seq_len} tokens)")
