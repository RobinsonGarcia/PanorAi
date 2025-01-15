import math
import numpy as np
from scipy import ndimage
import cv2
import logging
import os
import sys

from .utils.remapper import RemapConfig  # <-- import the RemapConfig (and Remapper if needed)
from .utils.unsharp import UnsharpMaskConfig


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('DEBUG', 'False').lower() in ('true', '1') else logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logger.level)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]


def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0

def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi


class ProjectionStrategy:
    """Abstract base class for projection strategies."""
    def forward(self, img: np.ndarray, lat_center: float, lon_center: float, fov: tuple) -> np.ndarray:
        raise NotImplementedError("Forward projection method must be implemented.")

    def backward(self, rect_img: np.ndarray, img_shape: tuple, lat_center: float, lon_center: float, fov: tuple) -> np.ndarray:
        raise NotImplementedError("Backward projection method must be implemented.")


class GnomonicProjector(ProjectionStrategy):
    """Gnomonic projection implementation."""

    def __init__(
        self, 
        dims: tuple[int, int], 
        shadow_angle_deg: float, 
        unsharp_fn=None,
        remap_config=None
    ):
        """
        Initialize the GnomonicProjector.

        :param dims: Dimensions of the output image (H, W).
        :param shadow_angle_deg: Shadow angle in degrees.
        :param order: The order of spline interpolation. Default is 3.
        :param prefilter: Whether to apply a prefilter (ndimage). Default is True.
        :param mode: Points outside boundaries handling mode (ndimage). Default is "nearest".
        :param unsharp_fn: A function that performs unsharp masking. Default is None.
        :param remap_config: A RemapConfig object that configures 'cv2' or 'ndimage' remapping.
        """
        self.dims = dims
        self.shadow_angle_deg = shadow_angle_deg
        self.unsharp_fn = unsharp_fn if unsharp_fn is not None else (lambda x: x)

        # Create or store a Remapper based on configuration
        if remap_config is None:
            # Default to cv2 approach with cubic interpolation
            remap_config = RemapConfig(
                method="cv2",
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_WRAP
            )
        self.remapper = remap_config.create_remapper()

        logger.info("Initialized GnomonicProjector with parameters:")
        logger.info(f"  dims: {self.dims}")
        logger.info(f"  shadow_angle_deg: {self.shadow_angle_deg}")
        logger.info(f"  unsharp_fn: {self.unsharp_fn}")
        logger.info(f"  remap_config: {remap_config}")


    def point_forward(self, x: np.ndarray, y: np.ndarray, phi1: float, lamb0: float) -> tuple[np.ndarray, np.ndarray]:
        """Convert x, y coordinates to phi, lamb for forward projection."""
        logger.debug("Running point_forward projection...")
        rho = np.sqrt(x**2 + y**2)
        c = np.arctan2(rho, 1.0)
        sinc = np.sin(c)
        cosc = np.cos(c)

        phi = np.arcsin(cosc * np.sin(phi1) + (y * sinc * np.cos(phi1) / (rho + 1e-9)))
        lamb = lamb0 + np.arctan2(x * sinc, (rho * np.cos(phi1) * cosc - y * sinc * np.sin(phi1) + 1e-9))

        # Clip / wrap results
        phi = np.clip(phi, -np.pi/2, np.pi/2)
        lamb = (lamb + np.pi) % (2 * np.pi) - np.pi
        return phi, lamb


    def point_backward(self, phi: np.ndarray, lamb: np.ndarray, phi1: float, lamb0: float, fov: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Convert phi, lamb coordinates to x, y for backward projection."""
        logger.debug("Running point_backward projection...")
        fov_h, fov_w = fov
        cosc = np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(phi) * np.cos(lamb - lamb0)

        # Avoid division by zero
        cosc = np.maximum(cosc, 1e-10)

        x = (np.cos(phi) * np.sin(lamb - lamb0)) / (cosc * fov_w)
        y = (np.cos(phi1) * np.sin(phi) - np.sin(phi1) * np.cos(phi) * np.cos(lamb - lamb0)) / (cosc * fov_h)

        return x, y


    def forward(self, img: np.ndarray, lat_center: float, lon_center: float, fov: tuple[float, float]) -> np.ndarray:
        """Forward projection from equirectangular to rectilinear."""
        logger.info("Starting forward projection...")
        logger.debug(f"  Image shape: {img.shape}")
        logger.debug(f"  Lat center: {lat_center}, Lon center: {lon_center}, FOV: {fov}")

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        H, W = self.dims
        fov_h, fov_w = fov

        # Generate rectilinear grid
        x, y = np.meshgrid(
            np.linspace(-1, 1, W)*fov_w,
            np.linspace(-1, 1 , H)*fov_h
        )

        # Convert to spherical coordinates
        phi, lamb = self.point_forward(x, y, lat_center, lon_center)
        out_range_mask = (rad_to_deg(phi) >= (90 - self.shadow_angle_deg)) & (rad_to_deg(phi) <= 90) #=====> Understand why lat is in range -90 to 90


        # Scale to original image's coordinate system
        HH, WW, _ = img.shape
        # Vertical coordinate: map phi in range [-pi/2, pi/2] to [0, HH-1], also factoring shadow_angle_deg
        logger.debug(f"  Phi max: {phi.max()}, Phi min: {phi.min()}")
        phi = (phi / (np.pi/2) + 1) * 0.5 * (HH - 1) * (180 / (180 - self.shadow_angle_deg))
        logger.debug(f"  v max: {phi.max()}, v min: {phi.min()}")
        # Horizontal coordinate: lamb in range [-pi, pi] to [0, WW-1]
        lamb = (lamb / np.pi + 1) * 0.5 * (WW - 1) 

        # Use Remapper to do the actual pixel mapping
        rect_img = self.remapper.remap_image(img, phi, lamb)
        rect_img[out_range_mask] = 0.0


        # Apply optional unsharp mask
        return self.unsharp_fn(rect_img)


    def backward(self, rect_img: np.ndarray, img_shape: tuple, lat_center: float, lon_center: float, fov: tuple[float, float], return_mask: bool =False) -> np.ndarray:
        """Backward projection from rectilinear to equirectangular."""
        logger.info("Starting backward projection...")
        logger.debug(f"  Rectilinear image shape: {rect_img.shape}")
        logger.debug(f"  Target image shape: {img_shape}")
        logger.debug(f"  Lat center: {lat_center}, Lon center: {lon_center}, FOV: {fov}")
        H, W, _ = img_shape

        # Generate the equirectangular grid
        u, v = np.meshgrid(
            np.linspace(-1, 1, W),
            np.linspace(-1, (90 - self.shadow_angle_deg) / 90, H)
        )
        phi = v * (np.pi / 2)
        lamb = u * np.pi

        # Convert to rectilinear coordinates
        cosc = np.sin(lat_center) * np.sin(phi) + np.cos(lat_center) * np.cos(phi) * np.cos(lamb - lon_center)

        # Apply a mask: valid only where cosc >= 0
        valid_mask = cosc >= 0
        x, y = self.point_backward(phi, lamb, lat_center, lon_center, fov)

        # Scale to image coordinates
        x = (x + 1) * 0.5 * (rect_img.shape[1] - 1)
        y = (y + 1) * 0.5 * (rect_img.shape[0] - 1)
        
        # Create a mask for x,y that lie inside the image boundaries
        in_range_mask = (
            (x >= 0) & (x < rect_img.shape[1]) &
            (y >= 0) & (y < rect_img.shape[0])
        )

        # Combine the geometry valid_mask with the in_range_mask
        final_mask = valid_mask & in_range_mask

        # Remap from rect_img back to equirectangular
        back_img = self.remapper.remap_image(rect_img, y, x)

        # Apply valid_mask
        for c in range(back_img.shape[2]):
            channel = back_img[..., c]
            channel[~final_mask] = 0
            back_img[..., c] = channel

        if return_mask:
            return back_img, final_mask
        return back_img

 


logger = logging.getLogger(__name__)

# Suppose GnomonicProjector is already imported somewhere above
# from gnomonic_projector import GnomonicProjector

# Available projector classes
PROJECTOR_CLASSES = {
    "GnomonicProjector": GnomonicProjector,
    # Add other projector classes here if needed
}

class ProjectorConfig:
    """Configuration for the projector."""

    def __init__(
            self, 
            dims, 
            shadow_angle_deg, 
            projector_cls="GnomonicProjector", 
            unsharp=False, 
            unsharp_cfg=None,
            remap_cfg=None,
            **projector_kwargs
    ):
        """
        Initialize projector configuration.

        :param dims: Tuple (height, width) for projection dimensions.
        :param shadow_angle_deg: Shadow angle for the projector.
        :param projector_cls: Projector class to be instantiated (can be a string or class).
        :param unsharp: Boolean that controls whether to apply unsharp masking.
        :param unsharp_cfg: An UnsharpMaskConfig instance (optional). If provided, overrides the default unsharp config.
        :param remap_config: A RemapConfig instance (optional). If provided, projector uses its remapper. 
        :param projector_kwargs: Additional keyword arguments for the projector.
        """
        self.dims = dims
        self.shadow_angle_deg = shadow_angle_deg
        self.unsharp = unsharp

        # Set up unsharp configuration
        if self.unsharp and unsharp_cfg:
            self.unsharp_cfg = unsharp_cfg or UnsharpMaskConfig(sigma=2.0, kernel_size=7, strength=3.5)
            masker = self.unsharp_cfg.create_masker()
            self.unsharp_fn = masker.apply_unsharp_mask
        else:
            self.unsharp_fn = lambda x: x
            self.unsharp_cfg = None

        # Set up remap configuration
        # Default to a CV2-based remapping with cubic interpolation and BORDER_WRAP
        self.remap_config = remap_cfg or RemapConfig(method="cv2")


        # Validate or set projector class
        if isinstance(projector_cls, str):
            if projector_cls not in PROJECTOR_CLASSES:
                raise ValueError(f"Unknown projector class name: {projector_cls}")
            self.projector_cls = PROJECTOR_CLASSES[projector_cls]
        else:
            self.projector_cls = projector_cls
        
        self.projector_kwargs = projector_kwargs

        logger.info("Initialized ProjectorConfig with parameters:")
        logger.info(f"  dims: {self.dims}")
        logger.info(f"  shadow_angle_deg: {self.shadow_angle_deg}")
        logger.info(f"  projector_cls: {self.projector_cls}")
        logger.info(f"  unsharp: {self.unsharp}")
        logger.info(f"  unsharp_cfg: {self.unsharp_cfg if unsharp_cfg else None}")
        logger.info(f"  remap_config: {self.remap_config}")

    def create_projector(self):
        """
        Instantiate the projector with the configuration.
        Attach the unsharp masking function so the projector can call it if needed,
        along with a remap_config that the projector can use to initialize a Remapper.
        """
        projector_instance = self.projector_cls(
            dims=self.dims,
            shadow_angle_deg=self.shadow_angle_deg,
            unsharp_fn=self.unsharp_fn, 
            remap_config=self.remap_config,  # <-- pass the remap config here
            **self.projector_kwargs
        )
        return projector_instance