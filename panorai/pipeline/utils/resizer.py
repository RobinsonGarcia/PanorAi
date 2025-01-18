from skimage.transform import resize
import logging
import sys
import cv2
import numpy as np
from typing import Union

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [stream_handler]


class ImageResizer:
    """Handles image resizing with explicit configuration."""

    def __init__(
        self,
        resize_factor: float = 1.0,
        method: str = "skimage",
        mode: str = "reflect",
        anti_aliasing: bool = True,
        interpolation: int = cv2.INTER_LINEAR
    ) -> None:
        """
        Initialize the ImageResizer with explicit attributes.

        Args:
            resize_factor (float): Factor by which to resize the image. >1 for upsampling, <1 for downsampling.
            method (str): Resizing method ('skimage' or 'cv2'). Default is 'skimage'.
            mode (str): Mode parameter for skimage resize. Default is "reflect".
            anti_aliasing (bool): Whether to apply anti-aliasing (only for skimage).
            interpolation (int): Interpolation method for cv2.resize. Default is cv2.INTER_LINEAR.
        """
        self.resize_factor = resize_factor
        self.method = method
        self.mode = mode
        self.anti_aliasing = anti_aliasing
        self.interpolation = interpolation

        logger.info(f"Initialized ImageResizer with resize_factor={resize_factor}, method={method}, "
                    f"mode={mode}, anti_aliasing={anti_aliasing}, interpolation={interpolation}")

    def resize_image(self, img: np.ndarray, upsample: bool = True) -> np.ndarray:
        """
        Resize the input image based on the stored configuration.

        Args:
            img (np.ndarray): Input image.
            upsample (bool): Whether we are upsampling or downsampling (affects effective resize_factor).

        Returns:
            np.ndarray: Resized image.
        """
        resize_factor = self.resize_factor
        if not upsample:
            resize_factor = 1 / resize_factor

        if resize_factor != 1.0:
            new_shape = (
                int(img.shape[0] * resize_factor),
                int(img.shape[1] * resize_factor),
            )
            logger.info(f"Resizing image with factor={resize_factor}.")
            logger.debug(f"Original shape: {img.shape}, New shape: {new_shape}.")

            if self.method == "skimage":
                if len(img.shape) == 3:  # e.g. RGB image
                    resized_img = resize(
                        img, (*new_shape, img.shape[2]),
                        mode=self.mode,
                        anti_aliasing=self.anti_aliasing
                    )
                else:  # Grayscale or single channel
                    resized_img = resize(
                        img, new_shape,
                        mode=self.mode,
                        anti_aliasing=self.anti_aliasing
                    )
                logger.info("Image resizing completed using skimage.")
                return resized_img
            elif self.method == "cv2":
                resized_img = cv2.resize(
                    img,
                    (new_shape[1], new_shape[0]),  # cv2 expects (width, height)
                    interpolation=self.interpolation
                )
                logger.info("Image resizing completed using cv2.")
                return resized_img
            else:
                raise ValueError(f"Unknown resizing method: {self.method}")

        logger.debug("No resizing applied; resize_factor is 1.0.")
        return img


class ResizerConfig:
    """Configuration for the resizer."""

    def __init__(
        self,
        resizer_cls: type = ImageResizer,
        resize_factor: float = 1.0,
        method: str = "skimage",
        mode: str = "reflect",
        anti_aliasing: bool = True,
        interpolation: int = cv2.INTER_LINEAR
    ) -> None:
        """
        Initialize resizer configuration.

        Args:
            resizer_cls (type): The ImageResizer class or a subclass to instantiate.
            resize_factor (float): Factor by which to resize the image.
            method (str): Resizing method ('skimage' or 'cv2').
            mode (str): Mode parameter for skimage resize.
            anti_aliasing (bool): Whether to apply anti-aliasing (only for skimage).
            interpolation (int): Interpolation method for cv2.resize.
        """
        self.resize_factor = resize_factor
        self.method = method
        self.mode = mode
        self.anti_aliasing = anti_aliasing
        self.interpolation = interpolation
        self.resizer_cls = resizer_cls

    def __repr__(self) -> str:
        return (f"ResizerConfig(resize_factor={self.resize_factor}, method='{self.method}', "
                f"mode='{self.mode}', anti_aliasing={self.anti_aliasing}, interpolation={self.interpolation})")

    def create_resizer(self) -> ImageResizer:
        """
        Create a new ImageResizer instance based on the current configuration.

        Returns:
            ImageResizer: An instance of ImageResizer (or subclass) with the specified configuration.
        """
        return self.resizer_cls(
            resize_factor=self.resize_factor,
            method=self.method,
            mode=self.mode,
            anti_aliasing=self.anti_aliasing,
            interpolation=self.interpolation
        )