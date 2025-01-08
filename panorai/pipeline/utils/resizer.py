from skimage.transform import resize
import logging
import sys
import cv2
# Logging setup for Jupyter notebooks
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [stream_handler]

class ImageResizer:
    """Handles image resizing with explicit configuration."""

    def __init__(self, resize_factor=1.0, method="skimage", mode="reflect", anti_aliasing=True, interpolation=cv2.INTER_LINEAR):
        """
        Initialize the ImageResizer with explicit attributes.

        :param resize_factor: Factor by which to resize the image. >1 for upsampling, <1 for downsampling.
        :param method: Resizing method ('skimage' or 'cv2'). Default is 'skimage'.
        :param mode: The mode parameter for the skimage resize function. Default is "reflect".
        :param anti_aliasing: Whether to apply anti-aliasing during resizing (only for skimage). Default is True.
        :param interpolation: Interpolation method for cv2.resize. Default is cv2.INTER_LINEAR.
        """
        self.resize_factor = resize_factor
        self.method = method
        self.mode = mode
        self.anti_aliasing = anti_aliasing
        self.interpolation = interpolation

        logger.info(f"Initialized ImageResizer with resize_factor={resize_factor}, method={method}, "
                    f"mode={mode}, anti_aliasing={anti_aliasing}, interpolation={interpolation}")

    def resize_image(self, img, upsample=True):
        """
        Resize the input image based on the configuration.

        :param img: Input image as a NumPy array.
        :param upsample: Whether to apply upsampling or downsampling.
        :return: Resized image.
        """
        resize_factor = self.resize_factor
        if not upsample:
            resize_factor = 1 / resize_factor

        if resize_factor != 1.0:
            new_shape = (
                int(img.shape[0] * resize_factor),
                int(img.shape[1] * resize_factor),
            )
            logger.info(f"Resizing image with resize_factor={resize_factor}.")
            logger.debug(f"Original shape: {img.shape}, New shape: {new_shape}.")

            if self.method == "skimage":
                if len(img.shape) == 3:  # RGB image
                    resized_img = resize(
                        img, (*new_shape, img.shape[2]),
                        mode=self.mode,
                        anti_aliasing=self.anti_aliasing
                    )
                else:  # Grayscale image
                    resized_img = resize(
                        img, new_shape,
                        mode=self.mode,
                        anti_aliasing=self.anti_aliasing
                    )
                logger.info("Image resizing completed using skimage.")
            elif self.method == "cv2":
                resized_img = cv2.resize(
                    img, (new_shape[1], new_shape[0]),  # cv2 expects (width, height)
                    interpolation=self.interpolation
                )
                logger.info("Image resizing completed using cv2.")
            else:
                raise ValueError(f"Unknown resizing method: {self.method}")

            return resized_img

        logger.debug("No resizing applied; resize_factor is 1.0.")
        return img
    
class ResizerConfig:
    """Configuration for the resizer."""

    def __init__(self, resizer_cls=ImageResizer, resize_factor=1.0, method="skimage", mode="reflect", anti_aliasing=True, interpolation=cv2.INTER_LINEAR):
        """
        Initialize resizer configuration.

        :param resize_factor: Factor by which to resize the image. >1 for upsampling, <1 for downsampling.
        :param method: Resizing method ('skimage' or 'cv2'). Default is 'skimage'.
        :param mode: The mode parameter for the skimage resize function. Default is "reflect".
        :param anti_aliasing: Whether to apply anti-aliasing during resizing (only for skimage). Default is True.
        :param interpolation: Interpolation method for cv2.resize. Default is cv2.INTER_LINEAR.
        """
        self.resize_factor = resize_factor
        self.method = method
        self.mode = mode
        self.anti_aliasing = anti_aliasing
        self.interpolation = interpolation
        self.resizer_cls = resizer_cls

    def __repr__(self):
        return (f"ResizerConfig(resize_factor={self.resize_factor}, method='{self.method}', "
                f"mode='{self.mode}', anti_aliasing={self.anti_aliasing}, interpolation={self.interpolation})")

    def create_resizer(self):
        return self.resizer_cls(
            resize_factor=self.resize_factor,
            method=self.method,
            mode=self.mode,
            anti_aliasing=self.anti_aliasing,
            interpolation=self.interpolation
        )
