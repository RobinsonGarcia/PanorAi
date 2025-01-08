# remapper.py

import numpy as np
import cv2
from scipy import ndimage
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [stream_handler]

class Remapper:
    """
    Handles remapping using either scipy.ndimage or OpenCV.
    """

    def __init__(
        self,
        method="ndimage",
        order=3,
        prefilter=True,
        mode="nearest",
        interpolation=cv2.INTER_CUBIC,
        border_mode=cv2.BORDER_WRAP
    ):
        """
        :param method: The remapping method ('ndimage' or 'cv2'). Default is 'ndimage'.
        :param order: Spline interpolation order (for ndimage). Default is 3.
        :param prefilter: Prefilter option (for ndimage). Default is True.
        :param mode: Out-of-bounds mode (for ndimage). Default is 'nearest'.
        :param interpolation: Interpolation (for cv2). Default is cv2.INTER_CUBIC.
        :param border_mode: Border handling (for cv2). Default is cv2.BORDER_WRAP.
        """
        self.method = method
        self.order = order
        self.prefilter = prefilter
        self.mode = mode
        self.interpolation = interpolation
        self.border_mode = border_mode

        logger.info(f"Initialized Remapper with method={method}, order={order}, prefilter={prefilter}, "
                    f"mode={mode}, interpolation={interpolation}, border_mode={border_mode}")

    def remap_image(self, img, phi, lamb):
        """
        Remap the input image using either ndimage.map_coordinates or cv2.remap.

        :param img: Input image as a NumPy array (H, W, C).
        :param phi: Float array, same shape as output, specifying the "row" coordinates.
        :param lamb: Float array, same shape as output, specifying the "col" coordinates.
        :return: Remapped image as a NumPy array (same shape as phi,lamb + channels).
        """
        logger.debug(f"Starting remap with method={self.method}.")
        logger.debug(f"Image shape: {img.shape}, phi shape: {phi.shape}, lamb shape: {lamb.shape}")

        if self.method == "ndimage":
            # For an image with C channels
            if img.ndim == 2:
                # Grayscale single-channel
                remapped = ndimage.map_coordinates(
                    img,
                    [phi, lamb],
                    order=self.order,
                    prefilter=self.prefilter,
                    mode=self.mode
                )
            else:
                C = img.shape[2]
                remapped = np.stack([
                    ndimage.map_coordinates(
                        img[..., i],
                        [phi, lamb],
                        order=self.order,
                        prefilter=self.prefilter,
                        mode=self.mode
                    ) for i in range(C)
                ], axis=-1)

            logger.info("Remapping completed using ndimage.map_coordinates.")
            return remapped

        elif self.method == "cv2":
            # Convert to float32 for OpenCV
            map_x = lamb.astype(np.float32)
            map_y = phi.astype(np.float32)
            remapped = cv2.remap(
                img,
                map_x,
                map_y,
                interpolation=self.interpolation,
                borderMode=self.border_mode
            )
            logger.info("Remapping completed using cv2.remap.")
            return remapped

        else:
            raise ValueError(f"Unknown remapping method: {self.method}")


class RemapConfig:
    """
    Configuration for the Remapper.
    """

    def __init__(
        self,
        method="ndimage",
        order=3,
        prefilter=True,
        mode="nearest",
        interpolation=cv2.INTER_CUBIC,
        border_mode=cv2.BORDER_WRAP
    ):
        """
        :param method: 'ndimage' or 'cv2'.
        :param order: Spline interpolation order (ndimage).
        :param prefilter: Prefilter option (ndimage).
        :param mode: Out-of-bounds mode (ndimage).
        :param interpolation: Interpolation mode (cv2).
        :param border_mode: Border handling mode (cv2).
        """
        self.method = method
        self.order = order
        self.prefilter = prefilter
        self.mode = mode
        self.interpolation = interpolation
        self.border_mode = border_mode

    def __repr__(self):
        return (f"RemapConfig(method='{self.method}', order={self.order}, "
                f"prefilter={self.prefilter}, mode='{self.mode}', "
                f"interpolation={self.interpolation}, border_mode={self.border_mode})")

    def create_remapper(self):
        """
        Instantiate a Remapper with the stored config.
        """
        return Remapper(
            method=self.method,
            order=self.order,
            prefilter=self.prefilter,
            mode=self.mode,
            interpolation=self.interpolation,
            border_mode=self.border_mode
        )