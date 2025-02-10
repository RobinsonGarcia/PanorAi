# panorai/pipeline/blender/registry.py

from typing import Any, Dict, Type
import logging

# Initialize logger for this module
logger = logging.getLogger('blender.registry')

class BlenderRegistry:
    """
    Registry for managing blending strategies.
    """
    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str, blender: Type[Any]) -> None:
        cls._registry[name] = blender
        logger.info(f"Blender '{name}' registered successfully.")

    @classmethod
    def get_blender(cls, name: str, **kwargs: Any):
        """
        Retrieve a registered blending strategy by name.

        :param name: Name of the blending strategy.
        :param kwargs: Optional parameters to update the blender.
        :return: An instance of the requested blender.
        """
        logger.debug(f"Retrieving blender '{name}' with override parameters: {kwargs}")
        if name not in cls._registry:
            error_msg = f"Blender '{name}' not found in the registry."
            logger.error(error_msg)
            raise ValueError(error_msg)

        blender = cls._registry[name]()
        blender.update(**kwargs)
        return blender

    @classmethod
    def list_blenders(cls) -> list:
        """
        List all registered blending strategies.

        Returns:
            list: A list of registered blender names.
        """
        logger.debug("Listing all registered blenders.")
        blenders = list(cls._registry.keys())
        logger.info(f"Registered blenders: {blenders}")
        return blenders