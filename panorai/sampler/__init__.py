# sampler/__init__.py

from .registry import SamplerRegistry
from .default_samplers import register_default_samplers

try:
    register_default_samplers()
except:
    raise "Cant register default samplers"

__all__ = ["SamplerRegistry"]