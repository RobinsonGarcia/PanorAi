# panorai/blender/__init__.py

from .registry import BlenderRegistry
from .default_blenders import register_default_blenders

try:
    register_default_blenders()
except Exception as e:
    raise RuntimeError(f"Failed to register default blenders: {e}")

__all__ = ["BlenderRegistry"]