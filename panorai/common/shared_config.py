import cv2
from ..projection_deprecated.projector import ProjectorConfig
from ..projection_deprecated.projector import UnsharpMaskConfig
from ..projection_deprecated.projector import RemapConfig
from ..pipeline import ResizerConfig, ProjectionPipeline
from ..sampler.sampler import SamplerConfig
from ..pipeline import PipelineConfig



class PipelineFullConfig:
    """Unified configuration for the ProjectionPipeline and its sub-components."""

    def __init__(
        self,
        dims=(1024, 1024),
        shadow_angle_deg=30,
        projector_cls="GnomonicProjector",
        unsharp=False,
        unsharp_sigma=1.0,
        unsharp_kernel_size=7,
        unsharp_strength=1.5,
        remap_method="ndimage",
        remap_order=3,
        remap_prefilter=True,
        remap_mode="nearest",
        remap_interpolation=cv2.INTER_CUBIC,
        remap_border_mode=cv2.BORDER_WRAP,
        sampler_cls="CubeSampler",
        sampler_kwargs=None,
        resize_factor=1.0,
        n_jobs=1,
    ):
        """
        Initialize a unified configuration for the ProjectionPipeline.

        :param dims: Tuple (height, width) for projection dimensions.
        :param shadow_angle_deg: Shadow angle for the projector.
        :param projector_cls: Projector class name or class.
        :param unsharp: Whether to apply unsharp masking.
        :param unsharp_sigma: Sigma for unsharp masking Gaussian blur.
        :param unsharp_kernel_size: Kernel size for unsharp masking (odd number).
        :param unsharp_strength: Strength of unsharp masking.
        :param remap_method: Method for remapping ('ndimage' or 'cv2').
        :param remap_order: Order of interpolation for remapping (ndimage).
        :param remap_prefilter: Prefilter option for remapping (ndimage).
        :param remap_mode: Out-of-bounds mode for remapping (ndimage).
        :param remap_interpolation: Interpolation method for remapping (cv2).
        :param remap_border_mode: Border handling mode for remapping (cv2).
        :param sampler_cls: Sampler class name or class.
        :param sampler_kwargs: Additional arguments for the sampler.
        :param resize_factor: Resize factor for the pipeline resizer.
        :param n_jobs: Number of parallel calls to proejctor.backward.
        """
        # Initialize projector configuration parameters
        self.dims = dims
        self.shadow_angle_deg = shadow_angle_deg
        self.projector_cls = projector_cls
        self.unsharp = unsharp
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_kernel_size = unsharp_kernel_size
        self.unsharp_strength = unsharp_strength

        # Initialize remap configuration parameters
        self.remap_method = remap_method
        self.remap_order = remap_order
        self.remap_prefilter = remap_prefilter
        self.remap_mode = remap_mode
        self.remap_interpolation = remap_interpolation
        self.remap_border_mode = remap_border_mode

        # Initialize sampler configuration
        self.sampler_cls = sampler_cls
        self.sampler_kwargs = sampler_kwargs or {}

        # Initialize pipeline-level configuration
        self.resize_factor = resize_factor
        self.n_jobs = n_jobs

    def create_pipeline(self):
        """
        Instantiate the ProjectionPipeline using the stored configuration.

        :return: A configured ProjectionPipeline instance.
        """
        # Create projector configuration
        projector_cfg = ProjectorConfig(
            dims=self.dims,
            shadow_angle_deg=self.shadow_angle_deg,
            projector_cls=self.projector_cls,
            unsharp=self.unsharp,
            unsharp_cfg=UnsharpMaskConfig(
                sigma=self.unsharp_sigma,
                kernel_size=self.unsharp_kernel_size,
                strength=self.unsharp_strength,
            ),
            remap_cfg=RemapConfig(
                method=self.remap_method,
                order=self.remap_order,
                prefilter=self.remap_prefilter,
                mode=self.remap_mode,
                interpolation=self.remap_interpolation,
                border_mode=self.remap_border_mode,
            ),
        )

        # Create sampler configuration
        sampler_cfg = SamplerConfig(
            sampler_cls=self.sampler_cls,
            **self.sampler_kwargs,
        )

        # Create pipeline-level configuration
        pipeline_cfg = PipelineConfig(
            resizer_cfg=ResizerConfig(resize_factor=self.resize_factor),
            n_jobs=self.n_jobs
        )

        # Return a fully configured pipeline
        return ProjectionPipeline(
            projector_cfg=projector_cfg,
            pipeline_cfg=pipeline_cfg,
            sampler_cfg=sampler_cfg,
        )

    def __repr__(self):
        """
        String representation of the full configuration.
        """
        return (
            f"PipelineFullConfig(\n"
            f"  dims={self.dims},\n"
            f"  shadow_angle_deg={self.shadow_angle_deg},\n"
            f"  projector_cls={self.projector_cls},\n"
            f"  unsharp={self.unsharp},\n"
            f"  unsharp_sigma={self.unsharp_sigma},\n"
            f"  unsharp_kernel_size={self.unsharp_kernel_size},\n"
            f"  unsharp_strength={self.unsharp_strength},\n"
            f"  remap_method={self.remap_method},\n"
            f"  remap_order={self.remap_order},\n"
            f"  remap_prefilter={self.remap_prefilter},\n"
            f"  remap_mode={self.remap_mode},\n"
            f"  remap_interpolation={self.remap_interpolation},\n"
            f"  remap_border_mode={self.remap_border_mode},\n"
            f"  sampler_cls={self.sampler_cls},\n"
            f"  sampler_kwargs={self.sampler_kwargs},\n"
            f"  resize_factor={self.resize_factor}\n"
            f")"
        )