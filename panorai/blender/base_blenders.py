# panorai/pipeline/blender/base_blenders.py

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseBlender(ABC):
    def __init__(self, **kwargs: Any) -> None:
        """
        Base Blender initialization.

        Args:
            **kwargs (Any): Additional parameters for sampler configuration.
        """
        self.params: Dict[str, Any] = kwargs
        
    @abstractmethod
    def blend(self, images, masks):
        """
        Perform blending on a set of images.

        :param images: List of image arrays to be blended.
        :param masks: List of corresponding masks for weighting.
        :return: Blended image.
        """
        pass

    def update(self, **kwargs):
        """Update the blending strategy with new parameters."""
        self.params.update(kwargs)