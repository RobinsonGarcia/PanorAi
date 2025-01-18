# panorai/sampler/default_samplers.py

import logging
from .registry import SamplerRegistry
from .base_samplers import SAMPLER_CLASSES

logger = logging.getLogger('sampler.default_samplers')


def register_default_samplers() -> None:
    """
    Registers default sampler classes into the SamplerRegistry.

    Raises:
        Exception: If sampler registration fails for any reason.
    """
    logger.debug("Registering default samplers.")
    for k, v in SAMPLER_CLASSES.items():
        # Instantiate and register the default instance
        sampler_instance = v()
        SamplerRegistry.register(k, sampler_instance)
    logger.debug("All default samplers registered.")