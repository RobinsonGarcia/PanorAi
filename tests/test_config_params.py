import pytest
import numpy as np
import cv2
import sqlite3
import os
import json
import matplotlib.pyplot as plt
import logging

from skimage.metrics import mean_squared_error
from panorai import PipelineFullConfig, PipelineData

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for verbose output
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
ch.setFormatter(formatter)
logger.handlers = [ch]  # Replace existing handlers

@pytest.fixture(scope="module")
def sample_data():
    """
    Loads real data from an NPZ file and returns a PipelineData instance.
    Adjust paths as needed.
    """
    data_path = '../images/sample.npz'
    logger.info(f"Loading data from {data_path}...")

    arr = np.load(data_path)
    logger.debug(f"Data keys: {list(arr.keys())}")

    rgb = arr['rgb']  # shape: (H, W, 3)
    logger.info(f"RGB shape={rgb.shape}, dtype={rgb.dtype}")
    depth = np.sqrt(np.sum(arr['z']**2, axis=-1))  # shape: (H, W)
    logger.info(f"Depth shape={depth.shape}, dtype={depth.dtype}")
    xyz = arr['z']    # shape: (H, W, 3)
    logger.info(f"XYZ shape={xyz.shape}, dtype={xyz.dtype}")

    # Wrap in a PipelineData object
    data = PipelineData.from_dict({
        "rgb": rgb,
        "depth": depth,
        "xyz_depth": xyz
    })
    logger.info("PipelineData created successfully.")
    return data

@pytest.fixture(scope="session", autouse=True)
def db_and_charts(request):
    """
    Session-level fixture that:
      1) Creates (or opens) a SQLite DB for logging results
      2) Yields so tests can run
      3) After tests finish, queries the DB and produces a chart of the best configurations.
    """
    db_path = "test_results.db"
    logger.info(f"Opening SQLite DB at {db_path}.")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    logger.debug("Creating table test_results (if not exists).")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_json TEXT,
            mse REAL
        )
        """
    )
    conn.commit()

    request.config.db_conn = conn

    yield  # ---- tests run here ----

    logger.info("Tests completed. Generating summary chart of best results.")
    conn.commit()

    # Retrieve results
    rows = cursor.execute("SELECT config_json, mse FROM test_results").fetchall()
    if not rows:
        logger.warning("No test results found in the database.")
        conn.close()
        return

    # Convert rows -> list of (config_dict, mse)
    results = []
    for config_json, mse in rows:
        config_dict = json.loads(config_json)
        results.append((config_dict, mse))

    # Sort by MSE ascending
    results.sort(key=lambda r: r[1])
    logger.info(f"Total test results: {len(results)}. Best MSE so far: {results[0][1]}")

    top_n = 10
    top_results = results[:top_n]

    plt.figure(figsize=(10, 6))
    x_vals = range(len(top_results))
    mse_vals = [r[1] for r in top_results]
    labels = [
        f"{r[0].get('dims')},\n"
        f"unsharp={r[0].get('unsharp')},\n"
        f"{r[0].get('remap_method')}"
        for r in top_results
    ]

    plt.bar(x_vals, mse_vals, color="skyblue")
    plt.xticks(x_vals, labels, rotation=30, ha="right")
    plt.ylabel("MSE (lower is better)")
    plt.title(f"Top {top_n} Pipeline Configs by MSE")
    plt.tight_layout()
    chart_path = "best_configs_chart.png"
    plt.savefig(chart_path)
    logger.info(f"Saved chart of best configs to {chart_path}")

    conn.close()
    logger.info("Closed SQLite DB connection.")

dims_list = [
    (64, 128),
    (128, 256),
]

shadow_angles = [
    0,
    30,
]

unsharp_flags = [
    False,
    True,
]

unsharp_sigmas = [
    0.5,
    2.0,
]

unsharp_kernels = [
    3,
    7,
]

unsharp_strengths = [
    1.0,
    2.5,
]

remap_methods = [
    "ndimage",
    "cv2",
]

remap_orders = [
    1,
    3,
]

remap_prefilters = [
    True,
    False,
]

remap_modes = [
    "nearest",
    "reflect",
]

remap_interpolations = [
    cv2.INTER_LINEAR,
    cv2.INTER_CUBIC,
]

remap_border_modes = [
    cv2.BORDER_WRAP,
    cv2.BORDER_CONSTANT,
]

sampler_clses = [
    "CubeSampler",
    #"FibonacciSampler",
    # "IcosahedronSampler",
]

resize_factors = [
    1.0,
    0.5,
]

sampler_kwargs_options = [
    {},
    {"n_points": 5},
    {"n_points": 20},
]

@pytest.mark.parametrize("dims", dims_list)
@pytest.mark.parametrize("shadow_angle_deg", shadow_angles)
@pytest.mark.parametrize("unsharp", unsharp_flags)
@pytest.mark.parametrize("unsharp_sigma", unsharp_sigmas)
@pytest.mark.parametrize("unsharp_kernel_size", unsharp_kernels)
@pytest.mark.parametrize("unsharp_strength", unsharp_strengths)
@pytest.mark.parametrize("remap_method", remap_methods)
@pytest.mark.parametrize("sampler_cls", sampler_clses)
@pytest.mark.parametrize("resize_factor", resize_factors)
def test_pipeline_config(
    dims,
    shadow_angle_deg,
    unsharp,
    unsharp_sigma,
    unsharp_kernel_size,
    unsharp_strength,
    remap_method,
    sampler_cls,
    resize_factor,
    sample_data,
    request
):
    """
    Main test that parametrize covers.
    Each combo of parameters is run as a separate test.
    We handle the sub-parameters for ndimage vs cv2 below.
    """
    logger.info("==== NEW TEST PARAM COMBINATION ====")
    logger.info(f"dims={dims}, shadow_angle={shadow_angle_deg}, unsharp={unsharp}, sigma={unsharp_sigma}, "
                f"kernel={unsharp_kernel_size}, strength={unsharp_strength}, remap={remap_method}, "
                f"sampler={sampler_cls}, resize={resize_factor}")

    if sampler_cls == "CubeSampler":
        sampler_kwargs_list = [{}]
    else:
        sampler_kwargs_list = sampler_kwargs_options

    for sampler_kwargs in sampler_kwargs_list:
        logger.debug(f"Sampler kwargs: {sampler_kwargs}")
        if remap_method == "ndimage":
            for order in remap_orders:
                for prefilter in remap_prefilters:
                    for mode in remap_modes:
                        logger.debug(f"Constructing config: order={order}, prefilter={prefilter}, mode={mode}")
                        config = build_full_config(
                            dims=dims,
                            shadow_angle_deg=shadow_angle_deg,
                            unsharp=unsharp,
                            unsharp_sigma=unsharp_sigma,
                            unsharp_kernel_size=unsharp_kernel_size,
                            unsharp_strength=unsharp_strength,
                            remap_method=remap_method,
                            remap_order=order,
                            remap_prefilter=prefilter,
                            remap_mode=mode,
                            remap_interpolation=None,
                            remap_border_mode=None,
                            sampler_cls=sampler_cls,
                            sampler_kwargs=sampler_kwargs,
                            resize_factor=resize_factor,
                            n_jobs=3,
                        )
                        run_forward_backward_and_check(config, sample_data, request)
        else:  # remap_method == 'cv2'
            for interpolation in remap_interpolations:
                for border_mode in remap_border_modes:
                    logger.debug(f"Constructing config: interpolation={interpolation}, border_mode={border_mode}")
                    config = build_full_config(
                        dims=dims,
                        shadow_angle_deg=shadow_angle_deg,
                        unsharp=unsharp,
                        unsharp_sigma=unsharp_sigma,
                        unsharp_kernel_size=unsharp_kernel_size,
                        unsharp_strength=unsharp_strength,
                        remap_method=remap_method,
                        remap_order=None,
                        remap_prefilter=None,
                        remap_mode=None,
                        remap_interpolation=interpolation,
                        remap_border_mode=border_mode,
                        sampler_cls=sampler_cls,
                        sampler_kwargs=sampler_kwargs,
                        resize_factor=resize_factor,
                        n_jobs=3,
                    )
                    run_forward_backward_and_check(config, sample_data, request)

def build_full_config(
    dims,
    shadow_angle_deg,
    unsharp,
    unsharp_sigma,
    unsharp_kernel_size,
    unsharp_strength,
    remap_method,
    remap_order,
    remap_prefilter,
    remap_mode,
    remap_interpolation,
    remap_border_mode,
    sampler_cls,
    sampler_kwargs,
    resize_factor,
    n_jobs
):
    if remap_order is None:
        remap_order = 3
    if n_jobs is None:
        n_jobs = 1
    if remap_prefilter is None:
        remap_prefilter = True
    if remap_mode is None:
        remap_mode = "nearest"
    if remap_interpolation is None:
        remap_interpolation = cv2.INTER_CUBIC
    if remap_border_mode is None:
        remap_border_mode = cv2.BORDER_WRAP

    logger.debug("Building PipelineFullConfig object.")
    config = PipelineFullConfig(
        dims=dims,
        shadow_angle_deg=shadow_angle_deg,
        projector_cls="GnomonicProjector",
        unsharp=unsharp,
        unsharp_sigma=unsharp_sigma,
        unsharp_kernel_size=unsharp_kernel_size,
        unsharp_strength=unsharp_strength,
        remap_method=remap_method,
        remap_order=remap_order,
        remap_prefilter=remap_prefilter,
        remap_mode=remap_mode,
        remap_interpolation=remap_interpolation,
        remap_border_mode=remap_border_mode,
        sampler_cls=sampler_cls,
        sampler_kwargs=sampler_kwargs,
        resize_factor=resize_factor,
        n_jobs=n_jobs
    )
    return config

def run_forward_backward_and_check(config, pipeline_data, request):
    """
    1. Create pipeline from config.
    2. Run forward projection with sampler.
    3. Run backward projection.
    4. Compute MSE
    5. Insert results into SQLite DB
    6. Basic assertions on MSE
    """
    logger.info("Creating pipeline from config.")
    pipeline = config.create_pipeline()

    logger.info("Starting forward projection with sampler.")
    rect_data = pipeline.project_with_sampler(pipeline_data, fov=(1, 1))
    logger.info("Forward projection completed.")

    logger.info("Starting backward projection.")
    rgb_shape = pipeline_data.as_dict()["rgb"].shape
    reconstructed_dict = pipeline.backward_with_sampler(
        rect_data, img_shape=rgb_shape, fov=(1, 1)
    )
    logger.info("Backward projection completed.")

    original_rgb = pipeline_data.as_dict()["rgb"]
    reconstructed_rgb = reconstructed_dict["rgb"]
    reconstructed_rgb = np.clip(reconstructed_rgb, 0, 1)

    mse_val = mean_squared_error(original_rgb, reconstructed_rgb)
    logger.info(f"Computed MSE: {mse_val:.6f}")

    # Insert into DB
    config_json = config_to_json(config)
    db_conn = request.config.db_conn
    cursor = db_conn.cursor()
    logger.debug("Inserting test result into DB.")
    cursor.execute(
        "INSERT INTO test_results (config_json, mse) VALUES (?, ?)",
        (config_json, mse_val),
    )
    db_conn.commit()

    # Basic checks
    if np.isnan(mse_val):
        logger.error("MSE is NaN!")
    assert not np.isnan(mse_val), f"MSE is NaN with config={config}"
    if mse_val >= 1.0:
        logger.warning(f"MSE={mse_val} is unexpectedly large for config={config}")
    assert mse_val < 1.0, f"Unexpectedly large MSE={mse_val} with config={config}"

def config_to_json(config):
    d = {
        "dims": config.dims,
        "shadow_angle_deg": config.shadow_angle_deg,
        "projector_cls": config.projector_cls,
        "unsharp": config.unsharp,
        "unsharp_sigma": config.unsharp_sigma,
        "unsharp_kernel_size": config.unsharp_kernel_size,
        "unsharp_strength": config.unsharp_strength,
        "remap_method": config.remap_method,
        "remap_order": config.remap_order,
        "remap_prefilter": config.remap_prefilter,
        "remap_mode": config.remap_mode,
        "remap_interpolation": config.remap_interpolation,
        "remap_border_mode": config.remap_border_mode,
        "sampler_cls": config.sampler_cls,
        "sampler_kwargs": config.sampler_kwargs,
        "resize_factor": config.resize_factor,
    }
    return json.dumps(d)