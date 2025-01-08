from ..projection.projector import deg_to_rad
import numpy as np
from .pipeline_data import PipelineData
from skimage.transform import resize
        
import logging
import os
# Configure logging for Jupyter notebooks or scripts
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('DEBUG', 'False').lower() in ('true', '1') else logging.INFO)

import sys
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logger.level)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.handlers = [stream_handler]  # Replace existing handlers

from .utils.resizer import ResizerConfig
from ..sampler import SamplerConfig
from ..projection import ProjectorConfig

class PipelineConfig:
    """Configuration for the pipeline."""
    def __init__(self, resizer_cfg=None, resize_factor=1.0):
        """
        Initialize pipeline-level configuration.

        :param resize_factor: Factor by which to resize input images before projection.
        """
        
        self.resizer_cfg = resizer_cfg or ResizerConfig(resize_factor=resize_factor)
        
class ProjectionPipeline:
    """
    Manages sampling and projection strategies using modular configuration objects.
    """
    def __init__(self, projector_cfg: ProjectorConfig = None, pipeline_cfg: PipelineConfig = None, sampler_cfg: SamplerConfig = None):
        """
        Initialize the pipeline with configuration objects.

        :param sampler_cfg: Configuration for the sampler (optional).
        :param projector_cfg: Configuration for the projector (optional).
        :param pipeline_cfg: Configuration for the pipeline (optional).
        """
        # Use default configurations if not provided
        self.sampler_cfg = sampler_cfg or SamplerConfig(sampler_cls="CubeSampler")
        self.projector_cfg = projector_cfg or ProjectorConfig(dims=(1024, 1024), shadow_angle_deg=30, unsharp=False)
        self.pipeline_cfg = pipeline_cfg or PipelineConfig(resize_factor=1.)

        # Initialize sampler and projector
        self.sampler = self.sampler_cfg.create_sampler()
        self.projector = self.projector_cfg.create_projector()
        self.resizer = self.pipeline_cfg.resizer_cfg.create_resizer()

    def _resize_image(self, img, upsample=True):
        """
        Resize the input image using the ImageResizer.

        :param img: Input image as a NumPy array.
        :return: Resized image.
        """
        return self.resizer.resize_image(img, upsample)
    
    def _prepare_data(self, data):
        """
        Prepare data for projection. Converts single images into a dictionary format.

        :param data: ProjectionData or a single NumPy image.
        :return: Dictionary with keys as data names and values as NumPy arrays.
        """
        if isinstance(data, PipelineData):
            return {k: self._resize_image(v) for k,v in data.as_dict().items()}
        elif isinstance(data, np.ndarray):
            return {"rgb": self._resize_image(data)}
        else:
            raise TypeError("Data must be either a PipelineData instance or a NumPy array.")

    def set_sampler(self, sampler):
        """Set the sphere sampler."""
        self.sampler = sampler

    def set_projector(self, projector):
        """Set the projection strategy."""
        self.projector = projector

    # === Forward Projections ===
    def project_with_sampler(self, data, fov=(1, 1)):
        """
        Perform forward projection for all tangent points in the sampler.

        :param data: ProjectionData or a single NumPy image.
        :param fov: Field of view (height, width).
        :return: Dictionary with projections for each data type and tangent point.
        """
        if not self.sampler:
            raise ValueError("Sampler is not set.")
        tangent_points = self.sampler.get_tangent_points()
        prepared_data = self._prepare_data(data)
        projections = {name: {} for name in prepared_data.keys()}

        for idx, (lat, lon) in enumerate(tangent_points):
            lat = deg_to_rad(lat)
            lon = deg_to_rad(lon)
            for name, img in prepared_data.items():
                logger.debug(f"Projecting {name} for tangent point {idx + 1} at (lat={lat}, lon={lon}).")
                import matplotlib.pyplot as plt

                projections[name][f"point_{idx + 1}"] = self.projector.forward(img, lat, lon, fov)
                # For debuging
                # plt.title(f"Projecting {name} for tangent point {idx + 1} at (lat={lat}, lon={lon}).")
                # plt.hist(projections[name][f"point_{idx + 1}"].flatten())
                # plt.show()
        return projections

    def single_projection(self, data, lat_center, lon_center, fov=(1, 1)):
        """
        Perform a single forward projection for multiple inputs.

        :param data: ProjectionData or a single NumPy image.
        :param lat_center: Latitude center for the projection.
        :param lon_center: Longitude center for the projection.
        :param fov: Field of view (height, width).
        :return: Dictionary with projections for each data type or a single NumPy array.
        """
        lat_center = deg_to_rad(lat_center)
        lon_center = deg_to_rad(lon_center)
        prepared_data = self._prepare_data(data)
        projections = {name: self.projector.forward(img, lat_center, lon_center, fov) for name, img in prepared_data.items()}
        
        # If the input was a single image, return only the image result
        if isinstance(data, np.ndarray):
            return list(projections.values())[0]
        return projections

    # === Backward Projections ===

    def backward_with_sampler(self, rect_data, img_shape, fov=(1, 1)):
        """
        Perform backward projection for all tangent points in the sampler and combine results into a single image.
        
        :param rect_data: Dictionary of rectilinear images (outputs of forward projections).
        :param img_shape: Shape of the original spherical image (H, W, C).
        :param fov: Field of view (height, width).
        :return: Combined equirectangular image as a NumPy array.
        """
        if not self.sampler:
            raise ValueError("Sampler is not set.")
        
        tangent_points = self.sampler.get_tangent_points()
        combined_images = {name: np.zeros(img_shape, dtype=np.float32) for name in rect_data.keys()}
        weight_map = {name: np.zeros(img_shape[:2], dtype=np.float32) for name in rect_data.keys()} # Weight map for blending overlaps

        for idx, (lat, lon) in enumerate(tangent_points):
            lat = deg_to_rad(lat)
            lon = deg_to_rad(lon)
            for name, images in rect_data.items():
                rect_img = images.get(f"point_{idx + 1}")  # Extract NumPy array for this tangent point
                if rect_img is None:
                    raise ValueError(f"Missing projection for point_{idx + 1} in rect_data[{name}].")
                
                # Perform backward projection
                logger.debug(f"Backward projecting {name} for tangent point {idx + 1} at (lat={lat}, lon={lon}).")
                equirect_img, mask = self.projector.backward(rect_img, img_shape, lat, lon, fov, return_mask=True)

                # Combine results with weighting
                #mask = (equirect_img.sum(axis=-1) > 0).astype(np.float32)  # Non-zero pixel mask
                combined_images[name] += equirect_img * mask[..., None]
                weight_map[name] += mask

        # Normalize combined images by weight map
          
        #for name in combined_images.keys():
        #    w = np.maximum(weight_map[name], 1) # Avoid division by zero
        #    combined_images[name] /= w[..., None]

        # If there are multiple outputs (e.g., 'rgb', 'depth'), return as a dictionary
        return {name: img for name, img in combined_images.items()} #.astype(np.uint8)

    def single_backward(self, rect_data, img_shape, lat_center, lon_center, fov=(1, 1)):
        """
        Perform a single backward projection for multiple inputs.

        :param rect_data: Dictionary of rectilinear images or a single image.
        :param img_shape: Shape of the original spherical image (H, W, C).
        :param lat_center: Latitude center for the projection.
        :param lon_center: Longitude center for the projection.
        :param fov: Field of view (height, width).
        :return: Dictionary with projections for each data type or a single NumPy array.
        """
        lat_center = deg_to_rad(lat_center)
        lon_center = deg_to_rad(lon_center)

        if isinstance(rect_data, np.ndarray):
            rect_data = {"rgb": rect_data}

        projections = {name: self.projector.backward(img, img_shape, lat_center, lon_center, fov) for name, img in rect_data.items()}

        if len(rect_data) == 1 and "rgb" in rect_data:
            return list(projections.values())[0]
        return projections