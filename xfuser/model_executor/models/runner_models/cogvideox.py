import os

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.core.distributed import get_world_group
from xfuser.core.utils.runner_utils import log
from xfuser.envs import _is_rbln
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
    ModelSettings,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    LoadRoute,
)


def _float_timestep_hook(module, args, kwargs):
    """CogVideoX's DDIM/DPM schedulers hand the transformer int64 timesteps. The NPU has no
    cast from a 64-bit integer input to the compute dtype, so the compiled program takes them
    as float32 (the sinusoidal projection casts to float first thing anyway)."""
    timestep = kwargs.get("timestep")
    if torch.is_tensor(timestep) and not timestep.is_floating_point():
        kwargs = dict(kwargs)
        kwargs["timestep"] = timestep.to("cpu").to(torch.float32).to(timestep.device)
    return args, kwargs


@register_model("zai-org/CogVideoX-2b")
@register_model("THUDM/CogVideoX-2b")
@register_model("CogVideoX-2b")
class xFuserCogVideoXModel(xFuserModel):
    """CogVideoX-2b text-to-video through the composition-style pipeline wrapper.

    CFG parallel puts the two guidance branches on two ranks; sequence parallel gives every
    rank a band of latent rows and runs attention through USP. The VAE is CogVideoX's causal
    3D VAE.
    """

    # From the checkpoint's transformer config (30 heads of 64).
    attention_heads = 30

    load_support = LoadSupport(
        meta_transformers=(),
        meta_text_encoders=(),
        replicated_meta=False,
        routes=LoadRoute.NONE,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_cfg_parallel=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=480,
        width=720,
        num_frames=49,
        num_inference_steps=50,
        guidance_scale=6.0,
        max_sequence_length=226,
    )
    settings = ModelSettings(
        model_name="zai-org/CogVideoX-2b",
        output_name="cogvideox_2b",
        model_output_type="video",
        fps=8,
    )

    def _load_model(self) -> DiffusionPipeline:
        from xfuser import xFuserCogVideoXPipeline

        pipe = xFuserCogVideoXPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            engine_config=self.engine_config,
            torch_dtype=torch.bfloat16,
        )
        if _is_rbln():
            # Ahead of the compiled forward's own input hook, which then sees the float tensor.
            pipe.transformer.register_forward_pre_hook(_float_timestep_hook, with_kwargs=True, prepend=True)
        return pipe

    def _get_compiled_pipe_components(self):
        return ["transformer"]

    def _rbln_text_seq_len(self) -> int:
        # The pipeline pads to its own default (226 tokens), not to --max_sequence_length.
        return self.default_input_values.max_sequence_length

    def _get_runtime_state_pipeline(self):
        # The runtime state picks its CogVideoX patch layout (3D VAE scale factors, rows of
        # every latent frame per rank) by the diffusers pipeline's class name.
        return getattr(self.pipe, "module", self.pipe)

    # ------------------------------------------------------------------ #
    # RBLN: decode with a VAE compiled by optimum-rbln (--rbln_vae_path)
    # ------------------------------------------------------------------ #
    def _external_vae_npu(self) -> int:
        """Index (within ``RBLN_DEVICES``, like the ranks' own ``rbln:N``) of the VAE's NPU.

        At 49 frames of 480x720 the two decoder programs hold about 12 GB of a 15.7 GB ATOM,
        so the VAE gets an NPU to itself: ``--rbln_vae_npu``, else the first one beyond the
        ranks', or the one after that when the text encoders sit there.
        """
        explicit = getattr(self.config, "rbln_vae_npu", None)
        if explicit is not None:
            return int(explicit)
        if self._rbln_text_encoder_mode() == "spare":
            index = self._rbln_spare_device_index() + 1
        else:
            index = int(os.environ.get("LOCAL_WORLD_SIZE", get_world_group().world_size))
        count = torch.rbln.device_count()
        if index >= count:
            raise ValueError(
                f"--rbln_vae_path needs NPU index {index} for the VAE, but only {count} NPUs are visible: "
                "expose one more through RBLN_DEVICES or name one with --rbln_vae_npu."
            )
        return index

    def _external_vae(self):
        """The optimum-rbln VAE on its own NPU, loaded on first use.

        xDiT compiles ``AutoencoderKL`` decoders itself; CogVideoX's VAE decodes the video in
        chunks of latent frames with convolution caches carried across chunks, which
        optimum-rbln compiles as two device programs with the caches kept on the NPU.
        """
        vae = getattr(self, "_rbln_external_vae", None)
        if vae is None:
            from optimum.rbln import RBLNAutoencoderKLCogVideoX

            path = self.config.rbln_vae_path
            if os.path.isdir(os.path.join(path, "vae")):
                path = os.path.join(path, "vae")
            device = self._external_vae_npu()
            log(f"RBLN: loading the optimum-rbln VAE from {path} on NPU index {device}")
            vae = RBLNAutoencoderKLCogVideoX.from_pretrained(path, export=False, rbln_config={"device": device})
            self._rbln_external_vae = vae
        return vae

    def _decode_external(self, latents: torch.Tensor):
        """``CogVideoXPipeline.decode_latents`` and post-processing, on the external VAE."""
        pipe = getattr(self.pipe, "module", self.pipe)
        vae = self._external_vae()
        timer = getattr(self, "_stage_timer", None)
        if timer is not None:
            timer._start("decode")
        latents = latents.to("cpu", torch.float32).permute(0, 2, 1, 3, 4)  # (batch, channels, frames, h, w)
        dump = os.environ.get("XDIT_RBLN_DUMP_LATENTS")
        if dump:  # debugging aid: the final latents of this run, to compare layouts numerically
            torch.save(latents, dump)
        frames = vae.decode(latents / pipe.vae_scaling_factor_image).sample
        if timer is not None:
            timer._stop("decode")
        return pipe.video_processor.postprocess_video(video=frames.float(), output_type="pil")

    def _fit_positional_embedding(self, height: int, width: int, num_frames: int) -> None:
        """Give the patch embedding a positional table of this video size in the model's dtype.

        The table is a non-persistent buffer: loading leaves it in float64, and Dynamo hands
        it to the compiled graph as an input, which the NPU cannot cast from 64 bits. It is
        also sized for 49 frames of 480x720; for any other size the patch embedding builds a
        new one inside ``forward``, a tensor created on the device at trace time that the
        rebel compiler cannot lift into the graph.
        """
        pipe = getattr(self.pipe, "module", self.pipe)
        patch_embed = getattr(pipe.transformer, "patch_embed", None)
        patch_embed = getattr(patch_embed, "module", patch_embed)  # behind xDiT's layer wrapper
        if patch_embed is None or not getattr(patch_embed, "use_positional_embeddings", False):
            return
        latent_height = height // pipe.vae_scale_factor_spatial
        latent_width = width // pipe.vae_scale_factor_spatial
        size = (latent_height, latent_width, num_frames)
        table = patch_embed.pos_embedding
        dtype = patch_embed.proj.weight.dtype
        fits = size == (patch_embed.sample_height, patch_embed.sample_width, patch_embed.sample_frames)
        if fits and table.dtype == dtype:
            return
        if not fits:
            device = table.device
            table = patch_embed._get_positional_embeddings(*size).to(device)
            patch_embed.sample_height, patch_embed.sample_width, patch_embed.sample_frames = size
        patch_embed.register_buffer("pos_embedding", table.to(dtype=dtype), persistent=False)
        log(f"RBLN: positional table set for {num_frames} frames of {height}x{width} in {dtype}")

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        external = _is_rbln() and self._rbln_vae_mode() == "external"
        if _is_rbln():
            self._fit_positional_embedding(input_args["height"], input_args["width"], input_args["num_frames"])
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            num_frames=input_args["num_frames"],
            prompt=input_args["prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=self._make_generator(input_args["seed"]),
            output_type="latent" if external else "pil",
        )
        frames = output.frames if output else None
        if external:
            world = get_world_group()
            # Every rank holds the gathered latents; one decode is enough, on the rank whose
            # output the runner saves.
            frames = self._decode_external(frames) if world.rank == world.world_size - 1 else None
        videos = [frames[0]] if frames is not None and frames[0] is not None else []
        return DiffusionOutput(videos=videos, pipe_args=input_args)
