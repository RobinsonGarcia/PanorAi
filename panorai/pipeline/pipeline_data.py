import numpy as np

class _PipelineData:
    """
    A container for paired data (e.g., RGB image, depth map, and additional arrays) for projection.
    """
    def __init__(self, rgb: np.ndarray, depth: np.ndarray = None, **kwargs):
        """
        Initialize PipelineData with RGB, optional depth map, and additional data arrays.

        :param rgb: RGB image as a NumPy array.
        :param depth: Depth map as a NumPy array (optional).
        :param kwargs: Additional data as keyword arguments, where the key is the name of the data, and the value is the data array.
        """
        self.data = {"rgb": rgb / 255.0}  # Normalize RGB by default
        if depth is not None:
            self.data["depth"] = depth
        self.data.update(kwargs)  # Add any additional data arrays

    def as_dict(self):
        """
        Returns the data as a dictionary for projection processing.

        :return: Dictionary with keys as data names and values as NumPy arrays.
        """
        return self.data

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a PipelineData instance from a dictionary.

        :param data: Dictionary with keys as data names and values as NumPy arrays.
        :return: PipelineData instance.
        """
        if "rgb" not in data:
            raise ValueError("The 'rgb' key is required in the dictionary to create PipelineData.")
        rgb = data.pop("rgb")
        depth = data.pop("depth", None)
        return cls(rgb=rgb, depth=depth, **data)
    

import numpy as np

class PipelineData:
    """
    A container for paired data (e.g., RGB image, depth map, and additional arrays) for projection.
    """
    def __init__(self, rgb: np.ndarray, depth: np.ndarray = None, **kwargs):
        """
        Initialize PipelineData with RGB, optional depth, and additional data arrays.

        :param rgb: RGB image as a NumPy array (H, W, 3) typically in [0..255].
        :param depth: Depth map as a NumPy array (H, W) or (H, W, 1).
        :param kwargs: Additional data arrays, e.g., "xyz_depth" -> (H, W, 3).
        """
        # Normalize RGB by default to [0..1].
        self.data = {}
        if rgb is not None:
            # Ensure 3rd dim is 3 if it's truly RGB. Some code might pass (H, W). 
            # We'll assume shape is (H, W, 3).
            self.data["rgb"] = rgb / 255.0  
        if depth is not None:
            # shape might be (H, W) or (H, W, 1)
            self.data["depth"] = depth
        # Add any additional data arrays
        self.data.update(kwargs)

    def as_dict(self):
        """Return a dictionary of all stored arrays."""
        return self.data

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a PipelineData instance from a dictionary.

        :param data: Dictionary with keys as data names and values as NumPy arrays.
                     Must contain at least "rgb" or handle the case if missing.
        :return: PipelineData instance.
        """
        if "rgb" not in data:
            raise ValueError("The 'rgb' key is required to create PipelineData.")

        rgb = data.pop("rgb")
        depth = data.pop("depth", None)

        return cls(rgb=rgb, depth=depth, **data)

    def stack_all(self):
        """
        Stacks all channels into a single multi-channel array along the last dimension.
        Returns (H, W, total_channels).

        We assume:
          - 'rgb' is (H, W, 3)
          - 'depth' is (H, W) or (H, W, 1)
          - any additional keys, e.g. 'xyz_depth' is (H, W, 3)
        """
        # Sort keys if you want a consistent channel ordering 
        # (e.g., always stack in [rgb -> depth -> xyz_depth] order).
        # For simplicity, let's gather them in alphabetical order or define your own.
        # We'll do a stable sort by key:
        sorted_keys = sorted(self.data.keys())

        stacked_list = []
        for k in sorted_keys:
            arr = self.data[k]
            if arr.ndim == 2:
                # e.g. (H, W) -> expand to (H, W, 1)
                arr = arr[..., np.newaxis]
            stacked_list.append(arr)

        # Now we can np.concatenate along channels
        stacked = np.concatenate(stacked_list, axis=-1)
        return stacked, sorted_keys

    def unstack_all(self, stacked_array, keys_order):
        """
        Unstacks a single multi-channel array back into separate entries in self.data.

        :param stacked_array: (H, W, total_channels)
        :param keys_order: the list of keys that was used in stack_all (sorted_keys).
        :return: None (updates self.data in-place).
        """
        # We need to split the channels back in the same proportions as before.
        # We'll figure out each shape from the original arrays in self.data.

        # But note: after a forward/backward pass, shape might remain the same (H, W, C).
        # We'll rely on the original shapes from self.data to see how many channels each had.

        start_c = 0
        for k in keys_order:
            orig = self.data[k]
            orig_shape = orig.shape  # e.g. (H, W), (H, W, 3), etc.
            if orig.ndim == 2:
                num_c = 1
            else:
                num_c = orig_shape[-1]

            # Extract the slice
            end_c = start_c + num_c
            chunk = stacked_array[..., start_c:end_c]

            # If the original was 2D, squeeze
            if orig.ndim == 2:
                chunk = chunk[..., 0]  # remove that last dimension
            self.data[k] = chunk
            start_c = end_c

    def unstack_new_instance(self, stacked_array, keys_order):
        """
        Optionally create a new PipelineData instance with data splitted from stacked_array.
        Sometimes we don't want to mutate self.data in place.
        """
        # We'll create a new dictionary
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

        # Then build a new PipelineData from the dictionary
        # we have to pass them properly (rgb, depth, or kwargs)
        # We'll do a simple approach: if 'rgb' in new_data, that's for 'rgb'; 
        # if 'depth' in new_data, for 'depth'. The rest -> kwargs
        return PipelineData.from_dict(new_data)