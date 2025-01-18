from ..projection_deprecated.projector import deg_to_rad, rad_to_deg
import numpy as np

import numpy as np
from .pipeline_data import PipelineData
from skimage.transform import resize
        
import logging
import os
import sys

# For parallelization
from joblib import Parallel, delayed

from .utils.resizer import ResizerConfig

from ..sampler import SamplerRegistry
from ..submodules.projections import ProjectionRegistry

from ..projection_deprecated.projector import deg_to_rad


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('DEBUG', 'False').lower() in ('true', '1') else logging.INFO)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logger.level)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.handlers = [stream_handler]  # Replace existing handlers


class PipelineConfig:
    """Configuration for the pipeline."""
    def __init__(self, resizer_cfg=None, resize_factor=1.0, n_jobs=1):
        """
        Initialize pipeline-level configuration.

        :param resize_factor: Factor by which to resize input images before projection.
        """
        self.resizer_cfg = resizer_cfg or ResizerConfig(resize_factor=resize_factor)
        self.n_jobs = n_jobs

    def update(self, **kwargs):
        """
        Update configuration using keyword arguments.

        :param kwargs: Dictionary of attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                continue


class ProjectionPipeline:
    """
    Manages sampling and projection strategies using modular configuration objects.
    Stacks all data channels into one multi-channel array for forward/backward operations,
    automatically un-stacks after backward if input was PipelineData.

    Additionally, if stacking is used in forward pass, we override `img_shape` in 
    backward pass to the actual stacked shape, preventing shape mismatches.
    """
    def __init__(
        self,
        projection_name: str = None,
        sampler_name: str = None,
        pipeline_cfg: PipelineConfig = None,
    ):
        """
        :param projector_cfg: Configuration for the projector (optional).
        :param pipeline_cfg: PipelineConfig (optional).
        :param sampler_cfg: SamplerConfig (optional).
        """
        # Default configurations
        self.pipeline_cfg = pipeline_cfg or PipelineConfig(resize_factor=1.0)

        # Create sampler, projector, resizer
        if projection_name is None:
            raise ValueError(
                "A 'projection_name' must be specified when creating a ProjectionPipeline instance. "
                f"These are the available options: {ProjectionRegistry.list_projections()}."
            )
        self.projection_name = projection_name
        self.sampler_name = sampler_name
        self.sampler = SamplerRegistry.get_sampler(sampler_name) if self.sampler_name else None
        self.projector = ProjectionRegistry.get_projection(projection_name, return_processor=True)
        self.resizer = self.pipeline_cfg.resizer_cfg.create_resizer()

        # Parallel setting
        self.n_jobs = self.pipeline_cfg.n_jobs

        # For un-stacking after backward:
        self._original_data = None   # the PipelineData if used
        self._keys_order = None      # list of data keys from stack_all
        self._stacked_shape = None   # shape (H, W, total_channels) from forward pass
    
    @classmethod
    def list_samplers(cls):
        return SamplerRegistry.list_samplers()
    
    @classmethod
    def list_projections(cls):
        return ProjectionRegistry.list_projections()

    def __repr__(self):
        projection_config = self.projector.config.config_object.config.model_dump()
        sampler_config = self.sampler.params

        # Formatting the projection and sampler configurations
        projection_config_str = "\n".join(f"{key}: {value}" for key, value in projection_config.items())
        sampler_config_str = (
            "\n".join(f"{key}: {value}" for key, value in sampler_config.items()) if sampler_config else "    No parameters"
        )

        return f"""
{self.projection_name.capitalize()} Projection - Configurations:
{projection_config_str}

{self.sampler_name} Sampler - Configurations:
{sampler_config_str}

Note: You can pass any updates to these configurations via kwargs.
"""
    
    def update(self, **kwargs):
        self.projector.config.update(**kwargs)
        self.sampler.update(**kwargs)
        self.pipeline_cfg.update(**kwargs)
        
        # Resizer
        self.resizer = self.pipeline_cfg.resizer_cfg.create_resizer()

        # Parallel setting
        self.n_jobs = self.pipeline_cfg.n_jobs
        pass

    def _resize_image(self, img, upsample=True):
        """Resize the input image using the ImageResizer."""
        return self.resizer.resize_image(img, upsample)
    
    def _prepare_data(self, data):
        """
        If data is PipelineData, call data.stack_all() => single (H, W, C), plus keys_order.
        Store references so we can unstack automatically after backward.
        """

        if isinstance(data, PipelineData):
            stacked, keys_order = data.stack_all()
            self._original_data = data
            self._keys_order = keys_order
            return stacked, keys_order
        elif isinstance(data, np.ndarray):
            self._original_data = None
            self._keys_order = None
            return data, None
        else:
            raise TypeError("Data must be either PipelineData or np.ndarray.")

    # === Forward Projection ===
    def project_with_sampler(self, data, **kwargs):
        """
        Forward projection on a single stacked array for all tangent points.
        Returns { "stacked": { "point_1": arr, ... } }
        """
        if not self.sampler:
            raise ValueError("Sampler is not set.")
        
        tangent_points = self.sampler.get_tangent_points()
        prepared_data, _ = self._prepare_data(data)
        # Store the shape so backward can override
        if isinstance(prepared_data, np.ndarray):
            self._stacked_shape = prepared_data.shape
        else:
            # Should not happen, but just in case
            self._stacked_shape = None

        projections = {"stacked": {}}
        for idx, (lat_deg, lon_deg) in enumerate(tangent_points, start=1):
            lat = deg_to_rad(lat_deg)
            lon = deg_to_rad(lon_deg)
            logger.debug(f"Forward projecting for point {idx}, lat={lat_deg}, lon={lon_deg}.")
            self.projector.config.update(phi1_deg=rad_to_deg(lat), lam0_deg=rad_to_deg(lon))
            # shadow_angle = kwargs.get('shadow_angle', 0)
            out_img = self.projector.forward(prepared_data)
            projections["stacked"][f"point_{idx}"] = out_img
            if self._original_data:
                projections[f"point_{idx}"] = self._original_data.unstack_new_instance(out_img, self._keys_order).as_dict()
            
        return projections

    def single_projection(self, data, **kwargs):
        """
        Single forward projection of a stacked array.
        """
        
        prepared_data, _ = self._prepare_data(data)
        if isinstance(prepared_data, np.ndarray):
            self._stacked_shape = prepared_data.shape
        out_img = self.projector.forward(prepared_data)
        if self._original_data:
            unstacked = self._original_data.unstack_new_instance(out_img, self._keys_order)
            output = {'stacked': out_img}
            output.update(unstacked.as_dict())
            return output
        else:
            return out_img   
        
    # === Backward Projection ===  
    def backward_with_sampler(self, rect_data, img_shape=None, **kwargs):
        """
        Handles backward projection and blends multiple equirectangular images into one
        using feathered blending to reduce visible edges.
        """
        if not self.sampler:
            raise ValueError("Sampler is not set.")

        # Override img_shape if forward shape is available
        if self._stacked_shape is not None:
            if img_shape != self._stacked_shape:
                logger.warning(
                    f"Overriding user-supplied img_shape={img_shape} with stacked_shape={self._stacked_shape} "
                    "to ensure consistent channel dimensions."
                )
            img_shape = self._stacked_shape

        tangent_points = self.sampler.get_tangent_points()
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)

        stacked_dict = rect_data.get("stacked")
        if stacked_dict is None:
            raise ValueError("rect_data must have a 'stacked' key with tangent-point images.")

        if self._stacked_shape is not None:
            self.projector.config.update(
                lon_points=img_shape[1],
                lat_points=img_shape[0]
            )

        tasks = []
        for idx, (lat_deg, lon_deg) in enumerate(tangent_points, start=1):
            rect_img = stacked_dict.get(f"point_{idx}")
            if rect_img is None:
                raise ValueError(f"Missing 'point_{idx}' in rect_data['stacked'].")

            if rect_img.shape[-1] != img_shape[-1]:
                raise ValueError(
                    f"rect_img has {rect_img.shape[-1]} channels, but final shape indicates {img_shape[-1]} channels.\n"
                    "Make sure the shapes match."
                )
            tasks.append((idx, lat_deg, lon_deg, rect_img))

        def _backward_task(idx, lat_deg, lon_deg, rect_img):
            logger.debug(f"[Parallel] Backward projecting point_{idx}, lat={lat_deg}, lon={lon_deg}...")

            self.projector.config.update(
                phi1_deg=lat_deg,
                lam0_deg=lon_deg,
            )

            equirect_img, mask = self.projector.backward(rect_img, return_mask=True)
            return idx, equirect_img, mask

        logger.info(f"Starting backward with n_jobs={self.n_jobs} on {len(tasks)} tasks.")
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_backward_task)(*task) for task in tasks
        )
        logger.info("All backward tasks completed.")

        # Blending logic with feathering
        for (idx, eq_img, mask) in results:
            # Calculate valid mask (feathering at the edges)
            valid_mask = np.max(eq_img > 0, axis=-1).astype(np.float32)

            # Feather the mask based on distance from the edges
            from scipy.ndimage import distance_transform_edt
            distance = distance_transform_edt(valid_mask)
            feathered_mask = distance / distance.max()  # Normalize to [0, 1]

            # Accumulate contributions with feathered weights
            combined += eq_img * feathered_mask[..., None]
            weight_map += feathered_mask

        # Normalize the combined image
        valid_weights = weight_map > 0
        combined[valid_weights] /= weight_map[valid_weights, None]

        # Ensure zero weights are explicitly set to zero
        combined[~valid_weights] = 0

        # Unstack if original data was used
        if self._original_data is not None and self._keys_order is not None:
            new_data = self._original_data.unstack_new_instance(combined, self._keys_order)
            output = {"stacked": combined}
            output.update(new_data.as_dict())
            return output
        else:
            return {"stacked": combined}




    def _backward_with_sampler(self, rect_data, img_shape=None, **kwargs):
        """
        If _stacked_shape is set from forward pass, override user-supplied `img_shape`
        to avoid shape mismatch. Then do the multi-channel backward pass, unstack if needed.

        :param rect_data: { "stacked": { "point_1": arr, ... } }
        :param img_shape: Potentially (H, W, 3) from user, but if we stacked 7 channels,
                        we override with (H, W, 7).
        :return: 
        If PipelineData was used, returns unstacked dict of { "rgb": arr, "depth": arr, ... }
        If user input was a raw array, returns { "stacked": combined }
        """
        if not self.sampler:
            raise ValueError("Sampler is not set.")
        
        # If we have a stacked_shape from forward, override
        if self._stacked_shape is not None:
            if img_shape != self._stacked_shape:
                logger.warning(
                    f"Overriding user-supplied img_shape={img_shape} with stacked_shape={self._stacked_shape} "
                    "to ensure consistent channel dimensions."
                )
            img_shape = self._stacked_shape

        tangent_points = self.sampler.get_tangent_points()
        combined = np.zeros(img_shape, dtype=np.float32)
        #weight_map = np.zeros(img_shape[:2], dtype=np.float32)
        weight_map = np.zeros(img_shape, dtype=np.float32)

        stacked_dict = rect_data.get("stacked")
        if stacked_dict is None:
            raise ValueError("rect_data must have a 'stacked' key with tangent-point images.")

        if self._stacked_shape is not None:
            self.projector.config.update(
                lon_points=img_shape[1],
                lat_points=img_shape[0]
            )

        tasks = []
        for idx, (lat_deg, lon_deg) in enumerate(tangent_points, start=1):
            rect_img = stacked_dict.get(f"point_{idx}")
            if rect_img is None:
                raise ValueError(f"Missing 'point_{idx}' in rect_data['stacked'].")

            if rect_img.shape[-1] != img_shape[-1]:
                raise ValueError(
                    f"rect_img has {rect_img.shape[-1]} channels, but final shape indicates {img_shape[-1]} channels.\n"
                    "Make sure the shapes match."
                )
            tasks.append((idx, lat_deg, lon_deg, rect_img))

        def _backward_task(idx, lat_deg, lon_deg, rect_img):
            logger.debug(f"[Parallel] Backward projecting point_{idx}, lat={lat_deg}, lon={lon_deg}...")

            self.projector.config.update(
                phi1_deg=lat_deg,
                lam0_deg=lon_deg,
            )

            equirect_img, mask = self.projector.backward(rect_img, return_mask=True)
            return idx, equirect_img, mask

        logger.info(f"Starting backward with n_jobs={self.n_jobs} on {len(tasks)} tasks.")
        results = Parallel( n_jobs=self.n_jobs)(
            delayed(_backward_task)(*task) for task in tasks
        )
        logger.info("All backward tasks completed.")

        # Merge
        import matplotlib.pyplot as plt
        for (idx, eq_img, mask) in results:
            
            new_data = self._original_data.unstack_new_instance(eq_img, self._keys_order).as_dict()
            plt.imshow(new_data['rgb'].astype(np.uint8))
            plt.show()
            #_mask = np.max((eq_img * mask[:, :, None] > 0), axis=-1) * 1.

            _mask = (eq_img  > 0) #* 1.

            combined[_mask] += eq_img[_mask] #* _mask #_mask[..., None]
            
            weight_map[_mask] += 1
        # Normalize combined image by valid weights
        valid_weights = weight_map > 0
        combined[valid_weights] /= weight_map[valid_weights]#, None]

        # Fill regions with zero weight to avoid NaNs
        # combined[~valid_weights] = 0

        # If we had PipelineData, unstack
        if self._original_data is not None and self._keys_order is not None:
            new_data = self._original_data.unstack_new_instance(combined, self._keys_order)
            output = {"stacked": combined}
            output.update(new_data.as_dict())
            return output
        else:
            return {"stacked": combined}

    def single_backward(self, rect_data, img_shape=None, **kwargs):
        """
        If we have self._stacked_shape, override user-supplied shape for channel consistency.
        If pipeline data was used, unstack automatically.
        """

        if self._stacked_shape is not None and img_shape != self._stacked_shape:
            logger.warning(
                f"Overriding user-supplied img_shape={img_shape} with stacked_shape={self._stacked_shape} "
                "for single_backward."
            )
            img_shape = self._stacked_shape

        if isinstance(rect_data, np.ndarray):
            out_img, _ = self.projector.backward(rect_data, return_mask=True)
            if self._original_data is not None and self._keys_order is not None:
                new_data = self._original_data.unstack_new_instance(out_img, self._keys_order)
                return new_data.as_dict()
            else:
                return out_img
        else:
            # Must have "stacked" key
            stacked_arr = rect_data.get("stacked")
            if stacked_arr is None:
                raise ValueError("Expecting key 'stacked' in rect_data for single_backward.")

            if stacked_arr.shape[-1] != img_shape[-1]:
                raise ValueError(
                    f"Stacked array has {stacked_arr.shape[-1]} channels, but final shape indicates {img_shape[-1]}.\n"
                    "Shapes must match."
                )
            out_img, _ = self.projector.backward(stacked_arr, return_mask=True)
            if self._original_data is not None and self._keys_order is not None:
                new_data = self._original_data.unstack_new_instance(out_img, self._keys_order)
                return new_data.as_dict()
            else:
                return out_img

    def project(self, data, **kwargs):
        self.update(**kwargs)
        if self.sampler:
            return self.project_with_sampler(data, **kwargs)
        else:
            return self.single_projection(data, **kwargs)
            
    def backward(self, data, img_shape=None, **kwargs):
        self.update(**kwargs)
        if self.sampler:
            return self.backward_with_sampler(data, img_shape, **kwargs)
        else:
            return self.single_backward(data, img_shape, **kwargs)
    