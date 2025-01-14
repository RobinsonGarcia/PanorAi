import cv2
import numpy as np
import logging


class PreprocessEquirectangularImage:
    # Set up the logger for the class
    logger = logging.getLogger("EquirectangularImage")
    logger.setLevel(logging.DEBUG)

    @classmethod
    def extend_height(cls, image, shadow_angle):
        """
        Extends the height of an equirectangular image based on the given additional FOV.

        Parameters:
            image (np.ndarray): Input equirectangular image.
            shadow_angle (float): Additional field of view in degrees to extend vertically.

        Returns:
            extended_image (np.ndarray): Image with extended bottom region.
            h_prime (int): Number of rows added to the original image.
        """
        cls.logger.info("Starting height extension with shadow_angle=%.2f", shadow_angle)

        if not isinstance(image, np.ndarray):
            cls.logger.error("Image is not a valid numpy array.")
            raise TypeError("Image must be a numpy array.")
        if shadow_angle <= 0:
            cls.logger.info("No extension needed as shadow_angle=0 or less.")
            return image  # No extension needed

        fov_original = 180.0

        if len(image.shape) == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape
        h_prime = int((shadow_angle / fov_original) * height)
        cls.logger.debug("Original height: %d, Additional height: %d", height, h_prime)

        black_extension = np.zeros((h_prime, width, channels), dtype=image.dtype)
        extended_image = np.vstack((image, black_extension))

        cls.logger.info("Height extension complete. New height: %d", extended_image.shape[0])
        return extended_image

    @classmethod
    def undo_extend_height(cls, extended_image, shadow_angle):
        """
        Removes the extra bottom rows that were added by 'extend_height', 
        computed from the same shadow_angle that was used to extend the image.

        Parameters:
            extended_image (np.ndarray): The extended equirectangular image.
            shadow_angle (float): Additional field of view in degrees that was used to extend.

        Returns:
            np.ndarray: The image with the extended part removed.
        """
        cls.logger.info("Attempting to remove extension based on shadow_angle=%.2f", shadow_angle)

        if not isinstance(extended_image, np.ndarray):
            cls.logger.error("Image is not a valid numpy array.")
            raise TypeError("extended_image must be a numpy array.")

        shadow_angle = np.abs(shadow_angle)

        # We need to reverse the logic used in extend_height.
        # In extend_height, new_height = original_height + h_prime,
        # where h_prime = int((shadow_angle / 180.0) * original_height).
        # Thus extended_height = original_height + h_prime
        # => original_height ≈ extended_height / (1 + shadow_angle/180.0).
        
        fov_original = 180.0
        ext_height, ext_width, ext_channels = extended_image.shape
        
        # Estimate the original height (may be off-by-one due to integer rounding)
        estimated_original_height = int(
            round(ext_height / (1.0 + shadow_angle / fov_original))
        )
        
        # h_prime is then the difference
        h_prime_est = ext_height - estimated_original_height

        cls.logger.debug(
            "Extended image height: %d, Estimated original height: %d, "
            "Estimated h_prime: %d", ext_height, estimated_original_height, h_prime_est
        )

        if h_prime_est <= 0:
            cls.logger.warning(
                "Computed extension (%d) is <= 0. Possibly shadow_angle was never used to extend. "
                "Returning the input image.", h_prime_est
            )
            return extended_image

        # Slice the image to remove the bottom h_prime_est rows
        restored_image = extended_image[:estimated_original_height, :, :]

        cls.logger.info(
            "Extended rows removed. New height: %d", restored_image.shape[0]
        )
        return restored_image

    @classmethod
    def rotate(cls, image, delta_lat, delta_lon):
        """
        Rotates an equirectangular image based on latitude (delta_lat) and longitude (delta_lon) shifts.

        Parameters:
            image (np.ndarray): Input equirectangular image.
            delta_lat (float): Latitude rotation in degrees.
            delta_lon (float): Longitude rotation in degrees.

        Returns:
            np.ndarray: Rotated equirectangular image.
        """
        cls.logger.info("Starting rotation with delta_lat=%.2f, delta_lon=%.2f", delta_lat, delta_lon)
     
        if len(image.shape) == 2:
            H, W = image.shape
            C = 1
        else:
            H, W, C = image.shape
        cls.logger.debug("Image dimensions: Height=%d, Width=%d, Channels=%d", H, W, C)

        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        xv, yv = np.meshgrid(x, y)

        lon = (xv / (W - 1)) * 360.0 - 180.0
        lat = 90.0 - (yv / (H - 1)) * 180.0

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x_sphere = np.cos(lat_rad) * np.cos(lon_rad)
        y_sphere = np.cos(lat_rad) * np.sin(lon_rad)
        z_sphere = np.sin(lat_rad)

        delta_lat_rad = np.radians(delta_lat)
        delta_lon_rad = np.radians(delta_lon)

        # Rotate around the X-axis (latitude shift)
        x_rot = x_sphere
        y_rot = y_sphere * np.cos(delta_lat_rad) - z_sphere * np.sin(delta_lat_rad)
        z_rot = y_sphere * np.sin(delta_lat_rad) + z_sphere * np.cos(delta_lat_rad)

        # Rotate around the Z-axis (longitude shift)
        x_final = x_rot * np.cos(delta_lon_rad) - y_rot * np.sin(delta_lon_rad)
        y_final = x_rot * np.sin(delta_lon_rad) + y_rot * np.cos(delta_lon_rad)
        z_final = z_rot

        lon_final = np.arctan2(y_final, x_final)
        lat_final = np.arcsin(z_final)

        lon_final_deg = np.degrees(lon_final)
        lat_final_deg = np.degrees(lat_final)

        x_rot_map = ((lon_final_deg + 180.0) / 360.0) * (W - 1)
        y_rot_map = ((90.0 - lat_final_deg) / 180.0) * (H - 1)

        map_x = x_rot_map.astype(np.float32)
        map_y = y_rot_map.astype(np.float32)

        rotated_image = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )

        cls.logger.info("Rotation complete.")
        return rotated_image

    @classmethod
    def preprocess(cls, image, **kwargs):
        """
        Preprocess an equirectangular image by optionally extending its height and then rotating it.

        Parameters:
            image (np.ndarray): Input equirectangular image.
            **kwargs: Parameters for preprocessing:
                - shadow_angle (float): Additional field of view in degrees to extend. Default is 0.
                - delta_lat (float): Latitude rotation in degrees. Default is 0.
                - delta_lon (float): Longitude rotation in degrees. Default is 0.

        Returns:
            processed_image (np.ndarray): Preprocessed (extended + rotated) image.
            h_prime (int): Number of rows added during extension.
        """
        shadow_angle = kwargs.get("shadow_angle", 0)
        delta_lat = kwargs.get("delta_lat", 0)
        delta_lon = kwargs.get("delta_lon", 0)

        cls.logger.info(
            "Starting preprocessing with parameters: shadow_angle=%.2f, delta_lat=%.2f, delta_lon=%.2f",
            shadow_angle, delta_lat, delta_lon
        )

        # Step 1: Extend the image height
        processed_image = cls.extend_height(image, shadow_angle) if shadow_angle >= 0 else cls.undo_extend_height(image, shadow_angle)
        
        # Step 2: Rotate the image
        processed_image = cls.rotate(processed_image, delta_lat, delta_lon)
        if len(processed_image.shape)==2:
            processed_image = processed_image[:, :, None]

        cls.logger.info("Preprocessing complete.")
        return processed_image

    @classmethod
    def save_image(cls, image, file_path):
        """
        Saves the current image to the specified file path.

        Parameters:
            image (np.ndarray): Image to save.
            file_path (str): Path to which the image should be saved.
        """
        if not isinstance(image, np.ndarray):
            cls.logger.error("Image is not a valid numpy array.")
            raise TypeError("Image must be a numpy array.")
        cv2.imwrite(file_path, image)
        cls.logger.info("Image saved to %s", file_path)
