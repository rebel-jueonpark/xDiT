import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

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


@register_model("stabilityai/stable-diffusion-xl-base-1.0")
@register_model("stable-diffusion-xl-base-1.0")
@register_model("SDXL")
class xFuserStableDiffusionXLModel(xFuserModel):
    """Stable Diffusion XL: a UNet, so the only intra-image parallelism is CFG parallel.

    The pipeline wrapper splits the UNet's CFG batch across the two ranks of the CFG group
    and gathers the halves after every call; sequence parallelism and PipeFusion need a
    transformer backbone.
    """

    load_support = LoadSupport(
        meta_transformers=(),
        meta_text_encoders=(),
        replicated_meta=False,
        routes=LoadRoute.NONE,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=False,
        ring_degree=False,
        use_cfg_parallel=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=5.0,
    )
    settings = ModelSettings(
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        output_name="stable_diffusion_xl",
        model_output_type="image",
    )

    def _load_model(self) -> DiffusionPipeline:
        from xfuser import xFuserStableDiffusionXLPipeline

        return xFuserStableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            engine_config=self.engine_config,
            torch_dtype=torch.bfloat16,
        )

    def _get_compiled_pipe_components(self):
        return ["unet"]

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=input_args["prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=self._make_generator(input_args["seed"]),
        )
        images = output.images if output else []
        return DiffusionOutput(images=images, pipe_args=input_args)
