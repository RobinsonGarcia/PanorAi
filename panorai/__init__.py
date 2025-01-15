# panorai/__init__.py

# Import core pipeline components
from .pipeline.pipeline import ProjectionPipeline, PipelineConfig
from .pipeline.pipeline_data import PipelineData
from .pipeline.utils.resizer import ResizerConfig
from .common import PipelineFullConfig

# Import sampler components
from .sampler.sampler import SamplerConfig, CubeSampler, IcosahedronSampler, FibonacciSampler

# Import projection components
#from .projection_deprecated.projector import ProjectorConfig, GnomonicProjector
#from .projection_deprecated.utils.remapper import RemapConfig
#from .projection_deprecated.utils.unsharp import UnsharpMaskConfig

# Define the public API for panorai
__all__ = [
    # Pipeline
    "ProjectionPipeline", "PipelineConfig", "ProjectionData", "ResizerConfig", "PipelineFullConfig",
    # Sampler
    "SamplerConfig", "CubeSampler", "IcosahedronSampler", "FibonacciSampler",
    # Projection
    #"ProjectorConfig", "GnomonicProjector", "RemapConfig", "UnsharpMaskConfig"
]