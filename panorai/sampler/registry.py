# panorai/sampler/registry.py

import logging
from typing import Any, Dict, Type
from .base_samplers import Sampler

logger = logging.getLogger('sampler.registry')


class SamplerRegistryError(Exception):
    """
    Custom exception for SamplerRegistry errors.
    """
    pass


class SamplerNotFoundError(SamplerRegistryError):
    """
    Raised when a requested sampler is not found in the registry.
    """
    pass


class SamplerRegistry:
    """
    Registry for managing projection sampler configurations.
    """
    _registry: Dict[str, Sampler] = {}

    @classmethod
    def register(cls, name: str, sampler: Sampler) -> None:
        """
        Register a Sampler instance under a given name.

        Args:
            name (str): Sampler name.
            sampler (Sampler): An instance of a Sampler.

        Raises:
            SamplerRegistryError: If there's any error during registration.
        """
        cls._registry[name] = sampler
        logger.info(f"Sampler '{name}' registered successfully.")

    @classmethod
    def get_sampler(cls, name: str, **kwargs: Any) -> Sampler:
        """
        Retrieve a sampler from the registry and update its parameters.

        Args:
            name (str): Name of the registered sampler.
            **kwargs (Any): Additional parameters to override the sampler's internal params.

        Returns:
            Sampler: The requested sampler instance.

        Raises:
            SamplerNotFoundError: If the sampler name is not found in the registry.
        """
        logger.debug(f"Retrieving sampler '{name}' with override parameters: {kwargs}")
        if name not in cls._registry:
            error_msg = f"Sampler '{name}' not found in the registry."
            logger.error(error_msg)
            raise SamplerNotFoundError(error_msg)

        sampler = cls._registry[name]
        sampler.update(**kwargs)
        return sampler

    @classmethod
    def list_samplers(cls) -> list:
        """
        List all registered sampler names.

        Returns:
            list: A list of sampler names.
        """
        logger.debug("Listing all registered samplers.")
        samplers = list(cls._registry.keys())
        logger.info(f"Registered samplers: {samplers}")
        return samplers