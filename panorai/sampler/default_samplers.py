# /Users/robinsongarcia/projects/gnomonic/projection/default_projections.py

from .registry import SamplerRegistry

from .base_samplers import SAMPLER_CLASSES
import logging

# Initialize logger for this module
logger = logging.getLogger('sampler.default_samplers')


def register_default_samplers():

    logger.debug("Registering default samplers.")
    for k, v in SAMPLER_CLASSES.items():
        SamplerRegistry.register(k, v())
    
    logger.debug("All default samplers registered.")