import logging
import sys

# Logging setup for Jupyter notebooks (or standard Python scripts)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [stream_handler]

class UnsharpMasker:
    """
    Applies an unsharp mask operation to sharpen an image using Gaussian blur subtraction.
    """

    def __init__(self, sigma=1.0, kernel_size=7, strength=1.5):
        """
        Initialize the UnsharpMasker with explicit attributes.

        :param sigma: Standard deviation for Gaussian blur.
        :param kernel_size: Kernel size (must be an odd number) for Gaussian blur.
        :param strength: Strength of the sharpening. Higher values produce a stronger effect.
        """
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.strength = strength

        logger.info(f"Initialized UnsharpMasker with sigma={sigma}, kernel_size={kernel_size}, strength={strength}")

    def apply_unsharp_mask(self, image):
        """
        Apply the unsharp mask to the input image.

        :param image: Input image as a NumPy array.
        :return: Sharpened image.
        """
        import cv2
        logger.debug("Starting unsharp masking process.")
        logger.debug(f"Applying GaussianBlur with kernel_size={self.kernel_size}, sigma={self.sigma}")

        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)

        logger.debug(f"Combining original image with blurred image for sharpening with strength={self.strength}.")
        # unsharp_mask = original_image * (1 + strength) + blurred_image * (-strength)
        sharpened = cv2.addWeighted(image, 1.0 + self.strength, blurred, -self.strength, 0)

        logger.info("Unsharp mask applied successfully.")
        return sharpened


class UnsharpMaskConfig:
    """
    Configuration for the UnsharpMasker.
    """

    def __init__(self, masker_cls=UnsharpMasker, sigma=1.0, kernel_size=7, strength=1.5):
        """
        Initialize unsharp mask configuration.

        :param sigma: Standard deviation for Gaussian blur.
        :param kernel_size: Kernel size (must be an odd number) for Gaussian blur.
        :param strength: Strength of the sharpening. Higher values produce a stronger effect.
        :param masker_cls: The class to use for creating the unsharp masker.
        """
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.strength = strength
        self.masker_cls = masker_cls

    def __repr__(self):
        return (f"UnsharpMaskConfig(sigma={self.sigma}, kernel_size={self.kernel_size}, "
                f"strength={self.strength})")

    def create_masker(self):
        """
        Create and return an instance of the unsharp masker class with the current config.
        """
        return self.masker_cls(
            sigma=self.sigma,
            kernel_size=self.kernel_size,
            strength=self.strength
        )