# /Users/robinsongarcia/projects/gnomonic/projection/registry.py

from typing import Any, Dict, Optional, Type, Union

import logging

# Initialize logger for this module
logger = logging.getLogger('sampler.registry')

class SamplerRegistry:
    """
    Registry for managing projection configurations and their components.
    """
    _registry: Dict[str, Dict[str, Type[Any]]] = {}

    @classmethod
    def register(cls, name: str, sampler: Dict[str, Type[Any]]) -> None:


        cls._registry[name] = sampler
        logger.info(f"Sampler '{name}' registered successfully.")

    @classmethod
    def get_sampler(
        cls, 
        name: str, 
        **kwargs: Any
    ):
        
        logger.debug(f"Retrieving sampler '{name}' with override parameters: {kwargs}")
        if name not in cls._registry:
            error_msg = f"Sampler '{name}' not found in the registry."
            logger.error(error_msg)
            raise "Registration error"

        sampler = cls._registry[name]
        
        sampler.update(**kwargs)
        
        return sampler

    @classmethod
    def list_samplers(cls) -> list:
        """
        List all registered projections.

        Returns:
            list: A list of projection names.
        """
        logger.debug("Listing all registered samplers.")
        samplers = list(cls._registry.keys())
        logger.info(f"Registered projections: {samplers}")
        return samplers