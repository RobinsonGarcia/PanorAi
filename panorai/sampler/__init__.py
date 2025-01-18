# sampler/__init__.py

from .registry import SamplerRegistry, SamplerRegistryError
from .default_samplers import register_default_samplers

# Added custom exception handling # Updated exception
try:
    register_default_samplers()
except Exception as e:
    raise SamplerRegistryError("Cannot register default samplers") from e

__all__ = ["SamplerRegistry", "SamplerRegistryError"]