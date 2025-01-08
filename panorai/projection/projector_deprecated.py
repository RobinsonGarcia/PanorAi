import math
import numpy as np
from scipy import ndimage
import cv2
from unsharp import UnsharpMaskConfig

def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi

import numpy as np
from scipy import ndimage


class ProjectionStrategy:
    """Abstract base class for projection strategies."""

    def forward(self, img: np.ndarray, lat_center: float, lon_center: float, fov: tuple) -> np.ndarray:
        raise NotImplementedError("Forward projection method must be implemented.")

    def backward(self, rect_img: np.ndarray, img_shape: tuple, lat_center: float, lon_center: float, fov: tuple) -> np.ndarray:
        raise NotImplementedError("Backward projection method must be implemented.")

import logging
import os
import sys
import numpy as np
from scipy import ndimage

# Configure logging for Jupyter notebooks
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('DEBUG', 'False').lower() in ('true', '1') else logging.INFO)

# StreamHandler for Jupyter notebook output
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logger.level)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]  # Replace other handlers with the notebook handler


class GnomonicProjector(ProjectionStrategy):
    """Gnomonic projection implementation."""

    def __init__(self, dims: tuple[int, int], shadow_angle_deg: float, order: int = 3, 
                 prefilter: bool = True, mode: str = "nearest", unsharp_fn=None):
        """
        Initialize the GnomonicProjector.

        Args:
            dims (tuple[int, int]): Dimensions of the output image (H, W).
            shadow_angle_deg (float): Shadow angle in degrees.
            order (int, optional): The order of the spline interpolation. Default is 3.
            prefilter (bool, optional): Whether to apply a prefilter before interpolation. Default is True.
            mode (str, optional): Points outside boundaries handling mode. Default is "nearest".
            unsharp (bool, optional): Whether to unsharp rectilinear projections. Default is True.
        """
        self.dims = dims
        self.shadow_angle_deg = shadow_angle_deg
        self.order = order
        self.prefilter = prefilter
        self.mode = mode
        self.unsharp_fn = unsharp_fn  # Store the unsharp masking function
        

        # Log initialization parameters
        logger.info("Initialized GnomonicProjector with parameters:")
        logger.info(f"  dims: {self.dims}")
        logger.info(f"  shadow_angle_deg: {self.shadow_angle_deg}")
        logger.info(f"  order: {self.order}")
        logger.info(f"  prefilter: {self.prefilter}")
        logger.info(f"  mode: {self.mode}")
        logger.info(f"  unsharp_fn: {self.unsharp_fn}")
        

    def point_forward(self, x: np.ndarray, y: np.ndarray, phi1: float, lamb0: float) -> tuple[np.ndarray, np.ndarray]:
        """Convert x, y coordinates to phi, lamb for forward projection."""
        logger.debug("Running point_forward projection...")
        rho = np.sqrt(x**2 + y**2)
        c = np.arctan2(rho, 1)
        sinc = np.sin(c)
        cosc = np.cos(c)

        phi = np.arcsin(cosc * np.sin(phi1) + (y * sinc * np.cos(phi1) / rho))
        lamb = lamb0 + np.arctan2(x * sinc, rho * np.cos(phi1) * cosc - y * sinc * np.sin(phi1))

        # Handle out-of-range values
        phi = np.clip(phi, -np.pi / 2, np.pi / 2)
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
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        H, W = self.dims
        fov_h, fov_w = fov

        # Generate the rectilinear grid
        x, y = np.meshgrid(
            np.linspace(-1, 1, W) * fov_w,
            np.linspace(-1, 1, H) * fov_h
        )

        # Convert to spherical coordinates
        phi, lamb = self.point_forward(x, y, lat_center, lon_center)

        # Scale to image coordinates
        HH, WW, C = img.shape
        phi = (phi / (np.pi / 2) + 1) * 0.5 * (HH - 1) * (180 / (180 - self.shadow_angle_deg))
        lamb = (lamb / np.pi + 1) * 0.5 * (WW - 1)

        # Interpolate pixel values
        """
        rect_img = np.stack([
            ndimage.map_coordinates(
                img[:, :, i],
                [phi, lamb],
                order=self.order,
                prefilter=self.prefilter,
                mode=self.mode
            ) for i in range(C)
        ], axis=-1)
        """

        map_x = lamb.astype(np.float32)
        map_y = phi.astype(np.float32)

        rect_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


        return self.unsharp_fn(rect_img)

    def backward(self, rect_img: np.ndarray, img_shape: tuple, lat_center: float, lon_center: float, fov: tuple[float, float]) -> np.ndarray:
        """Backward projection from rectilinear to equirectangular."""
        logger.info("Starting backward projection...")
        logger.debug(f"  Rectilinear image shape: {rect_img.shape}")
        logger.debug(f"  Target image shape: {img_shape}")
        logger.debug(f"  Lat center: {lat_center}, Lon center: {lon_center}, FOV: {fov}")
        H, W, _ = img_shape
        fov_h, fov_w = fov

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

        # Interpolate pixel values, applying the valid mask
        equirect_img = np.stack([
            ndimage.map_coordinates(
                rect_img[:, :, i],
                [y.ravel(), x.ravel()],
                order=self.order,
                prefilter=self.prefilter
            ).reshape(H, W) * valid_mask for i in range(rect_img.shape[2])
        ], axis=-1)

        return equirect_img



# Available projector and sampler classes
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
            order=3, 
            prefilter=True, 
            mode="nearest", 
            unsharp=False, 
            unsharp_cfg=None, 
            **projector_kwargs):
        """
        Initialize projector configuration.

        :param dims: Tuple (height, width) for projection dimensions.
        :param shadow_angle_deg: Shadow angle for the projector.
        :param projector_cls: Projector class to be instantiated (can be a string or class).
        :param order: The order of spline interpolation for the projector. Default is 3.
        :param prefilter: Whether to apply a prefilter before interpolation. Default is True.
        :param mode: Points outside boundaries handling mode. Default is "nearest".
        :param unsharp: Boolean that controls whether to apply unsharp masking.
        :param unsharp_cfg: An UnsharpMaskConfig instance (optional). If provided, overrides the default unsharp config.
        :param projector_kwargs: Additional keyword arguments for the projector.
        """
        self.dims = dims
        self.shadow_angle_deg = shadow_angle_deg
        self.order = order
        self.prefilter = prefilter
        self.mode = mode
        self.unsharp = unsharp

        # Set up unsharp configuration
        if self.unsharp or unsharp_cfg:
            self.unsharp_cfg = unsharp_cfg or UnsharpMaskConfig(sigma=2.0, kernel_size=7, strength=3.5)
            masker = self.unsharp_cfg.create_masker()
            self.unsharp_fn = masker.apply_unsharp_mask
        else:
            self.unsharp_fn = lambda x:x

        if isinstance(projector_cls, str):
            if projector_cls not in PROJECTOR_CLASSES:
                raise ValueError(f"Unknown projector class name: {projector_cls}")
            self.projector_cls = PROJECTOR_CLASSES[projector_cls]
        else:
            self.projector_cls = projector_cls
        
        self.projector_kwargs = projector_kwargs

    def create_projector(self):
        """
        Instantiate the projector with the configuration.
        Attach the unsharp masking function so the projector can call it if needed.
        """
        projector_instance = self.projector_cls(
            dims=self.dims,
            shadow_angle_deg=self.shadow_angle_deg,
            order=self.order,
            prefilter=self.prefilter,
            mode=self.mode,
            unsharp_fn=self.unsharp_fn,  # <-- pass the unsharp function here
            **self.projector_kwargs
        )
        return projector_instance














class __GnomonicProjector(ProjectionStrategy):
    """Gnomonic projection implementation."""

    def __init__(self, dims: tuple[int, int], shadow_angle_deg: float, order: int = 3, prefilter: bool = True, mode: str = "nearest"):
        """
        Initialize the GnomonicProjector.

        Args:
            dims (tuple[int, int]): Dimensions of the output image (H, W).
            shadow_angle_deg (float): Shadow angle in degrees.
            order (int, optional): The order of the spline interpolation. Default is 3.
            prefilter (bool, optional): Whether to apply a prefilter before interpolation. Default is True.
            mode (str, optional): Points outside boundaries handling mode. Default is "nearest".
        """
        self.dims = dims
        self.shadow_angle_deg = shadow_angle_deg
        self.order = order
        self.prefilter = prefilter
        self.mode = mode

    def point_forward(self, x: np.ndarray, y: np.ndarray, phi1: float, lamb0: float) -> tuple[np.ndarray, np.ndarray]:
        """Convert x, y coordinates to phi, lamb for forward projection."""
        rho = np.sqrt(x**2 + y**2)
        c = np.arctan2(rho, 1)
        sinc = np.sin(c)
        cosc = np.cos(c)

        phi = np.arcsin(cosc * np.sin(phi1) + (y * sinc * np.cos(phi1) / rho))
        lamb = lamb0 + np.arctan2(x * sinc, rho * np.cos(phi1) * cosc - y * sinc * np.sin(phi1))

        # Handle out-of-range values
        phi = np.clip(phi, -np.pi / 2, np.pi / 2)
        lamb = (lamb + np.pi) % (2 * np.pi) - np.pi
        return phi, lamb

    def point_backward(self, phi: np.ndarray, lamb: np.ndarray, phi1: float, lamb0: float, fov: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Convert phi, lamb coordinates to x, y for backward projection."""
        fov_h, fov_w = fov
        cosc = np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(phi) * np.cos(lamb - lamb0)

        # Avoid division by zero
        cosc = np.maximum(cosc, 1e-10)

        x = (np.cos(phi) * np.sin(lamb - lamb0)) / (cosc * fov_w)
        y = (np.cos(phi1) * np.sin(phi) - np.sin(phi1) * np.cos(phi) * np.cos(lamb - lamb0)) / (cosc * fov_h)

        return x, y

    def forward(self, img: np.ndarray, lat_center: float, lon_center: float, fov: tuple[float, float]) -> np.ndarray:
        """Forward projection from equirectangular to rectilinear."""
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        H, W = self.dims
        fov_h, fov_w = fov

        # Generate the rectilinear grid
        x, y = np.meshgrid(
            np.linspace(-1, 1, W) * fov_w,
            np.linspace(-1, 1, H) * fov_h
        )

        # Convert to spherical coordinates
        phi, lamb = self.point_forward(x, y, lat_center, lon_center)

        # Scale to image coordinates
        HH, WW, C = img.shape
        phi = (phi / (np.pi / 2) + 1) * 0.5 * (HH - 1) * (180 / (180 - self.shadow_angle_deg))
        lamb = (lamb / np.pi + 1) * 0.5 * (WW - 1)

        # Interpolate pixel values
        rect_img = np.stack([
            ndimage.map_coordinates(
                img[:, :, i],
                [phi, lamb],
                order=self.order,
                prefilter=self.prefilter,
                mode=self.mode
            ) for i in range(C)
        ], axis=-1)

        return rect_img

    def backward(self, rect_img: np.ndarray, img_shape: tuple, lat_center: float, lon_center: float, fov: tuple[float, float]) -> np.ndarray:
        """Backward projection from rectilinear to equirectangular."""
        H, W, _ = img_shape
        fov_h, fov_w = fov

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

        # Interpolate pixel values, applying the valid mask
        equirect_img = np.stack([
            ndimage.map_coordinates(
                rect_img[:, :, i],
                [y.ravel(), x.ravel()],
                order=self.order,
                prefilter=self.prefilter
            ).reshape(H, W) * valid_mask for i in range(rect_img.shape[2])
        ], axis=-1)

        return equirect_img
    
class _GnomonicProjector(ProjectionStrategy):
    """Gnomonic projection implementation."""

    def __init__(self, dims: tuple[int, int], shadow_angle_deg: float):
        self.dims = dims
        self.shadow_angle_deg = shadow_angle_deg
        

    def point_forward(self, x: np.ndarray, y: np.ndarray, phi1: float, lamb0: float) -> tuple[np.ndarray, np.ndarray]:
        """Convert x, y coordinates to phi, lamb for forward projection."""
        rho = np.sqrt(x**2 + y**2)
        c = np.arctan2(rho, 1)
        sinc = np.sin(c)
        cosc = np.cos(c)

        phi = np.arcsin(cosc * np.sin(phi1) + (y * sinc * np.cos(phi1) / rho))
        lamb = lamb0 + np.arctan2(x * sinc, rho * np.cos(phi1) * cosc - y * sinc * np.sin(phi1))

        # Handle out-of-range values
        phi = np.clip(phi, -np.pi / 2, np.pi / 2)
        lamb = (lamb + np.pi) % (2 * np.pi) - np.pi
        return phi, lamb

    def point_backward(self, phi: np.ndarray, lamb: np.ndarray, phi1: float, lamb0: float, fov: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Convert phi, lamb coordinates to x, y for backward projection."""
        fov_h, fov_w = fov
        cosc = np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(phi) * np.cos(lamb - lamb0)

        # Avoid division by zero
        cosc = np.maximum(cosc, 1e-10)

        x = (np.cos(phi) * np.sin(lamb - lamb0)) / (cosc * fov_w)
        y = (np.cos(phi1) * np.sin(phi) - np.sin(phi1) * np.cos(phi) * np.cos(lamb - lamb0)) / (cosc * fov_h)

        return x, y

    def forward(self, img: np.ndarray, lat_center: float, lon_center: float, fov: tuple[float, float]) -> np.ndarray:
        """Forward projection from equirectangular to rectilinear."""
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        H, W = self.dims
        fov_h, fov_w = fov

        # Generate the rectilinear grid
        x, y = np.meshgrid(
            np.linspace(-1, 1, W) * fov_w,
            np.linspace(-1, 1, H) * fov_h
        )

        # Convert to spherical coordinates
        phi, lamb = self.point_forward(x, y, lat_center, lon_center)

        # Scale to image coordinates
        HH, WW, C = img.shape
        phi = (phi / (np.pi / 2) + 1) * 0.5 * (HH - 1) * (180 / (180 - self.shadow_angle_deg))
        lamb = (lamb / np.pi + 1) * 0.5 * (WW - 1)

        # Interpolate pixel values
        rect_img = np.stack([
            ndimage.map_coordinates(img[:, :, i], [phi, lamb], order=3, prefilter=True, mode="mirror") for i in range(C)
        ], axis=-1)

        return rect_img
    def backward(self, rect_img: np.ndarray, img_shape: tuple, lat_center: float, lon_center: float, fov: tuple[float, float]) -> np.ndarray:
        """Backward projection from rectilinear to equirectangular."""
        H, W, _ = img_shape
        fov_h, fov_w = fov

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

        # Interpolate pixel values, applying the valid mask
        equirect_img = np.stack([
            ndimage.map_coordinates(rect_img[:, :, i], [y.ravel(), x.ravel()], order=0, prefilter=True)
            .reshape(H, W) * valid_mask for i in range(rect_img.shape[2])
        ], axis=-1)

        return equirect_img
   
class good_old_GnomonicProjector:
    def __init__(self,dims,shadow_angle_deg):
        self.f_projection=None
        self.b_projection=None
        self.dims=dims
        self.scanner_shadow_angle=shadow_angle_deg
        pass
    
    def point_forward(self,x,y,phi1,lamb0,fov):
        rho=np.sqrt(x**2+y**2)
        c=np.arctan2(rho,1)
        sinc=np.sin(c)
        cosc=np.cos(c)

        phi=np.arcsin(cosc*np.sin(phi1)+(y*sinc*np.cos(phi1)/rho))
        lamb=lamb0+np.arctan2(x*sinc,rho*np.cos(phi1)*cosc-y*np.sin(phi1)*sinc)
        
        phi=np.where(phi<-np.pi/2,np.pi/2-phi,phi)
        lamb=np.where(lamb<-np.pi,2*np.pi+lamb,lamb)

        phi=np.where(phi>np.pi/2,-np.pi/2+phi,phi)
        lamb=np.where(lamb>np.pi,-2*np.pi+lamb,lamb)
        
        return phi,lamb
    

    def forward(self,img,phi1,lamb0,fov=(1,1)):
        
        if len(img.shape)==2:
            img = np.stack([img,img,img],axis=-1)
        
        fov_h, fov_w = fov
        
        H,W=self.dims
        
        x , y = np.meshgrid(np.linspace( -1, 1, W) * fov_w ,\
                            np.linspace( -1, 1 , H) * fov_h)
        
        phi, lamb = self.point_forward(x ,y ,phi1 , lamb0, fov)
        
        mask = ( phi > np.pi/3)&( phi < np.pi/2)
        
        phi = phi/(np.pi/2)
                  
        lamb =  lamb/np.pi

        HH,WW,C=img.shape

        phi=( 0.5* ( phi + 1) ) * ( HH - 1) * ( ( 180 / ( 180 - self.scanner_shadow_angle) ) )
        #phi = phi * (HH -1)
        
        
        lamb=( 0.5 * ( lamb + 1 ) ) * ( WW - 1 )

        o_img = [ ndimage.map_coordinates( img[:,:,i], np.stack([phi,lamb]), order=0, prefilter=True, mode="nearest") for i in range(C)]
        
        o_img = np.stack( o_img, axis=-1) # grid-wrap
        
        self.f_projection=o_img
        self.phi1=phi1
        self.lamb0=lamb0
        self.fov=fov
        return o_img
    
    def point_backward(self,phi,lamb,phi1,lamb0,fov):
        fov_h,fov_w = fov
        cosc=np.sin(phi1)*np.sin(phi)+np.cos(phi1)*np.cos(phi)*np.cos(lamb-lamb0)

        K=1/cosc
        x=K*np.cos(phi)*np.sin(lamb-lamb0)/fov_w
        y=K*(np.cos(phi1)*np.sin(phi)-np.sin(phi1)*np.cos(phi)*np.cos(lamb-lamb0))/fov_h
        
        x=0.5*(x+1)
        y=0.5*(y+1)

        HH, WW = self.dims
        
        x=x*(WW-1)
        y=y*(HH-1)
        self.cosc=cosc
        return x,y
    
    def backward(self,face,img_shape,phi1=None,lamb0=None,fov=None):

        #if len(img.shape)==2:
        #    img = np.stack([img,img,img],axis=-1)


        if len(face.shape)==2:
            face = np.stack([face,face,face],axis=-1)
            
        H,W,_= img_shape
  

        u , v=  np.meshgrid(
                    np.linspace(-1 , 1 , W), \
                    np.linspace( -1 ,\
                                ( (90 - self.scanner_shadow_angle) / 90) , \
                                H))

        #u , v=np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))

        phi = v * (np.pi/2)
        lamb = u * np.pi
        
       
        x,y=self.point_backward(phi,lamb,phi1,lamb0,fov)

        coords = np.stack([x,y])


        oo=[ndimage.map_coordinates(face.T[i,:,:], coords,order=0, prefilter=True)*(self.cosc>=0) for i in range(3)]
        
        oo=np.stack(oo,axis=-1)
        self.b_projection=oo
        return oo