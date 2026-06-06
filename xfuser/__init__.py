# RBLN backend: install collective polyfills before xfuser internals import process groups.
try:
    from xfuser.envs import _is_rbln  # noqa: E402
    if _is_rbln():
        from xfuser.core.rbln_collectives import install_rbln_collective_polyfills  # noqa: E402
        install_rbln_collective_polyfills()
except Exception:
    pass

from xfuser.model_executor.pipelines import (
    xFuserPixArtAlphaPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserStableDiffusion3Pipeline,
    xFuserFluxPipeline,
    xFuserLattePipeline,
    xFuserHunyuanDiTPipeline,
    xFuserCogVideoXPipeline,
    xFuserConsisIDPipeline,
    xFuserStableDiffusionXLPipeline,
    xFuserSanaPipeline,
    xFuserSanaSprintPipeline,
)
from xfuser.config import xFuserArgs, EngineConfig
from xfuser.parallel import xDiTParallel

__all__ = [
    "xFuserPixArtAlphaPipeline",
    "xFuserPixArtSigmaPipeline",
    "xFuserStableDiffusion3Pipeline",
    "xFuserFluxPipeline",
    "xFuserLattePipeline",
    "xFuserHunyuanDiTPipeline",
    "xFuserCogVideoXPipeline",
    "xFuserConsisIDPipeline",
    "xFuserStableDiffusionXLPipeline",
    "xFuserSanaPipeline",
    "xFuserSanaSprintPipeline",
    "xFuserArgs",
    "EngineConfig",
    "xDiTParallel",
]
