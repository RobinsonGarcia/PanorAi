import numpy as np
from typing import Dict, List, Tuple, Union, Optional

from .utils import PreprocessEquirectangularImage


class PipelineData:
    """
    A container for paired data (e.g., RGB image, depth map, and additional arrays) for projection.
    """

    def __init__(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None, **kwargs: np.ndarray) -> None:
        """
        Initialize PipelineData with RGB, optional depth, and additional data arrays.

        Args:
            rgb (np.ndarray): RGB image as a NumPy array (H, W, 3).
            depth (Optional[np.ndarray]): Depth map as a NumPy array (H, W) or (H, W, 1).
            **kwargs (np.ndarray): Additional data arrays, e.g. "xyz_depth" -> (H, W, 3).

        Notes:
            You can store arbitrary data arrays, but each must share the same (H, W) dimensions.
        """
        self.data: Dict[str, np.ndarray] = {}

        if rgb is not None:
            self.data["rgb"] = rgb
        if depth is not None:
            self.data["depth"] = depth

        # Add any additional data arrays
        for k, v in kwargs.items():
            self.data[k] = v
        
        for k, v in self.data.items():
            self.H, self.W = v.shape[:2]
            break

    def as_dict(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary of all stored arrays.

        Returns:
            Dict[str, np.ndarray]: A dictionary of data arrays by name.
        """
        return self.data

    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> "PipelineData":
        """
        Create a PipelineData instance from a dictionary.

        Args:
            data (Dict[str, np.ndarray]): Dictionary with keys as data names and values as NumPy arrays.
                                          Must contain at least "rgb" or handle the case if missing.

        Returns:
            PipelineData: A new PipelineData instance.

        Raises:
            ValueError: If 'rgb' is not found in the data.
        """
        if "rgb" not in data:
            raise ValueError("The 'rgb' key is required to create PipelineData.")

        data_copy = data.copy()
        rgb = data_copy.pop("rgb")
        depth = data_copy.pop("depth", None)
        return cls(rgb=rgb, depth=depth, **data_copy)

    def stack_all(self) -> Tuple[np.ndarray, List[str]]:
        """
        Stacks all channels into a single multi-channel array along the last dimension.
        Returns (H, W, total_channels).

        Returns:
            (np.ndarray, List[str]): A tuple of (stacked_array, keys_order).
        """
        sorted_keys = sorted(self.data.keys())
        stacked_list = []

        for k in sorted_keys:
            arr = self.data[k]
            if arr.ndim == 2:
                # e.g. (H, W) -> expand to (H, W, 1)
                arr = arr[..., np.newaxis]
            stacked_list.append(arr)

        stacked = np.concatenate(stacked_list, axis=-1)
        return stacked, sorted_keys

    def unstack_all(self, stacked_array: np.ndarray, keys_order: List[str]) -> Dict[str, np.ndarray]:
        """
        Unstacks a single multi-channel array back into separate entries.

        Args:
            stacked_array (np.ndarray): (H, W, total_channels)
            keys_order (List[str]): The list of keys that was used in stack_all().

        Returns:
            Dict[str, np.ndarray]: A dictionary of unstacked arrays keyed by the original keys.
        """
        unstacked = {}
        start_c = 0
        for k in keys_order:
            orig = self.data[k]
            if orig.ndim == 2:
                num_c = 1
            else:
                num_c = orig.shape[-1]

            end_c = start_c + num_c
            chunk = stacked_array[..., start_c:end_c]
            if orig.ndim == 2:
                chunk = chunk[..., 0]
            unstacked[k] = chunk
            start_c = end_c
        return unstacked

    def unstack_new_instance(self, stacked_array: np.ndarray, keys_order: List[str]) -> "PipelineData":
        """
        Create a new PipelineData instance with data split from stacked_array.

        Args:
            stacked_array (np.ndarray): (H, W, total_channels)
            keys_order (List[str]): The list of keys that was used in stack_all().

        Returns:
            PipelineData: A new instance with unstacked data.
        """
        new_data = {}
        start_c = 0
        for k in keys_order:
            orig = self.data[k]
            if orig.ndim == 2:
                num_c = 1
            else:
                num_c = orig.shape[-1]

            end_c = start_c + num_c
            chunk = stacked_array[..., start_c:end_c]
            if orig.ndim == 2:
                chunk = chunk[..., 0]
            new_data[k] = chunk
            start_c = end_c

        return PipelineData.from_dict(new_data)

    def preprocess(self, shadow_angle: float = 0, delta_lat: float = 0, delta_lon: float = 0) -> None:
        """
        Optionally preprocess each stored array by extending and/or rotating the equirectangular image.

        Args:
            shadow_angle (float): Additional field of view in degrees to extend. Default is 0.
            delta_lat (float): Latitude rotation in degrees. Default is 0.
            delta_lon (float): Longitude rotation in degrees. Default is 0.
        """
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = PreprocessEquirectangularImage.preprocess(
                v,
                shadow_angle=shadow_angle,
                delta_lat=delta_lat,
                delta_lon=delta_lon
            )
        self._cached_data = self.data.copy()
        self.data = new_data