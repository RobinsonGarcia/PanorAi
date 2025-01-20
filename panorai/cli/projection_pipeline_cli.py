import argparse
import os
import json
import numpy as np
import logging
from datetime import datetime
import cv2
from panorai.pipeline.pipeline import ProjectionPipeline
from panorai.pipeline.pipeline_data import PipelineData
from panorai.submodules.projections import ProjectionRegistry
from panorai.sampler.registry import SamplerRegistry

def setup_logging(verbose):
    logging_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging_level,
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Projection Pipeline CLI",
    )

    # Utility options
    parser.add_argument("--list-projections", action="store_true", help="List all available projections.")
    parser.add_argument("--list-samplers", action="store_true", help="List all available samplers.")
    parser.add_argument("--list-files", action="store_true", help="List all files inside the provided NPZ input.")
    parser.add_argument("--show-pipeline", action="store_true", help="Show details of the instantiated pipeline object.")

    # Input parameters
    parser.add_argument("--input", type=str, help="Path to the input file or directory (required).")
    parser.add_argument("--array_files", type=str, nargs="*", help="Keys for data in the .npz file (e.g., rgb, depth).")

    # Projection parameters
    parser.add_argument("--projection_name", type=str, help="Name of the projection to use (required).")
    parser.add_argument("--sampler_name", type=str, default=None, help="Name of the sampler to use (optional).")
    parser.add_argument("--operation", choices=["project", "backward"], help="Operation to perform.")
    parser.add_argument("--kwargs", nargs="*", default=[], help="Additional arguments for the operation in key=value format.")

    # Output options
    parser.add_argument("--output_dir", type=str, default=".cache", help="Base directory to save output files.")
    parser.add_argument("--save_npz", action="store_true", help="Save results as a single .npz file.")
    parser.add_argument("--save_png", action="store_true", help="Save PNG images for each result.")
    parser.add_argument("--cmap", type=str, default="jet", help="Colormap name for single-channel arrays.")

    # Logging and verbosity
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    return parser.parse_args()

def list_projections():
    logging.info("Available Projections:")
    for projection in ProjectionRegistry.list_projections():
        logging.info(f" - {projection}")

def list_samplers():
    logging.info("Available Samplers:")
    for sampler in SamplerRegistry.list_samplers():
        logging.info(f" - {sampler}")

def list_npz_files(input_path):
    with np.load(input_path) as data:
        logging.info("Files in NPZ:")
        for key in data.keys():
            logging.info(f" - {key}")

def create_unique_output_dir(base_dir, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = f"{args.projection_name}_{args.operation or 'both'}"
    unique_dir = os.path.join(base_dir, f"run_{timestamp}_{params}")
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

def save_metadata(output_dir, args):
    metadata = {
        "input": args.input,
        "projection_name": args.projection_name,
        "sampler_name": args.sampler_name,
        "operation": args.operation,
        "kwargs": args.kwargs,
        "output_dir": output_dir,
        "command": " ".join(os.sys.argv)
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Saved metadata to {metadata_path}.")

def normalize_array(array: np.ndarray) -> np.ndarray:
    """
    Normalize a float array to the range [0, 255] for display.
    Returns a uint8 array.
    """
    if array.size == 0:
        return array.astype(np.uint8)
    min_val, max_val = np.min(array), np.max(array)
    if min_val == max_val:
        return np.zeros_like(array, dtype=np.uint8)
    norm = (array - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)

def apply_colormap(array_2d: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    """
    Apply a colormap to a single-channel (H, W) array.
    Returns a 3-channel BGR image for visualization.
    """
    norm_img = normalize_array(array_2d)

    cmap_dict = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
        # add more if needed
    }
    cv2_cmap = cmap_dict.get(cmap_name.lower(), cv2.COLORMAP_JET)

    color_img = cv2.applyColorMap(norm_img, cv2_cmap)
    return color_img

def compose_3channel(array_3d: np.ndarray) -> np.ndarray:
    """
    For a 3-channel numeric array (e.g., shape (H, W, 3)), normalize each channel independently
    and compose a single 8-bit 3-channel image.

    Returns a (H, W, 3)-shaped uint8 array suitable for cv2.imwrite().
    """
    # Assume array_3d.shape == (H, W, 3).
    out = np.zeros_like(array_3d, dtype=np.uint8)
    for c in range(3):
        out[..., c] = normalize_array(array_3d[..., c])
    return out

def _flatten_result_for_npz(result_dict, prefix=""):
    """
    Recursively traverse the 'result' dictionary and collect:
    { "<prefix>.<original_key>": (np_array, original_key) }
    """
    flat_data = {}
    for key, val in result_dict.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(val, dict):
            nested = _flatten_result_for_npz(val, prefix=new_key)
            flat_data.update(nested)
            continue

        if hasattr(val, "__dict__") and all(isinstance(v, np.ndarray) for v in val.__dict__.values()):
            for subk, arr in val.__dict__.items():
                final_key = f"{new_key}.{subk}"
                flat_data[final_key] = (arr, subk)
            continue

        if isinstance(val, np.ndarray):
            flat_data[new_key] = (val, key)
        else:
            logging.debug(f"Ignoring key '{new_key}' - not dict or ndarray.")
    return flat_data

def save_output(
    result: dict,
    output_dir: str,
    save_npz: bool,
    operation: str = None,
    save_png: bool = False,
    cmap: str = "jet"
):
    """
    Saves the entire `result` dictionary into a single compressed NPZ (output.npz) if requested,
    and optionally exports PNG images using the input's data format to decide how to save.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Flatten the results
    flat_dict = _flatten_result_for_npz(result)

    # 1) Save as NPZ if requested
    if save_npz:
        npz_path = os.path.join(output_dir, "output.npz")
        arrays_for_npz = {k: v[0] for k, v in flat_dict.items()}  # just the arrays
        np.savez_compressed(npz_path, **arrays_for_npz)
        logging.info(f"Saved all arrays to {npz_path}.")

    # 2) Optionally save PNG images
    if save_png:
        for full_key, (array, original_name) in flat_dict.items():
            safe_key = full_key.replace(".", "_")

            # Case 1: It's labeled "rgb" => treat as color image
            if original_name.lower() == "rgb":
                # Must be 3-channel, or we skip
                if array.ndim == 3 and array.shape[2] == 3:
                    #
                    # -- FIX: Convert from float64 (or any non-uint8) to uint8 if needed --
                    #
                    if array.dtype != np.uint8:
                        # Normalize to 0..255 and cast
                        array = normalize_array(array)  # returns uint8 in [0..255]
                    
                    # Now we can safely convert color without errors
                    bgr_img = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                    
                    png_path = os.path.join(output_dir, f"{safe_key}.png")
                    cv2.imwrite(png_path, bgr_img)
                    logging.info(f"Saved RGB array '{original_name}' to {png_path} (converted to BGR).")
                else:
                    logging.warning(
                        f"'{original_name}' was labeled RGB but has shape {array.shape}. Skipping."
                    )
                continue

            # Otherwise, treat as numeric data (depth/xyz/etc.) ...
            
            if array.ndim == 2:
                # Single-channel => apply colormap
                color_img = apply_colormap(array, cmap_name=cmap)
                png_path = os.path.join(output_dir, f"{safe_key}_colormap.png")
                cv2.imwrite(png_path, color_img)
                logging.info(f"Saved numeric array '{original_name}' with colormap to {png_path}.")

            elif array.ndim == 3 and array.shape[2] == 1:
                # Single-channel but (H, W, 1)
                color_img = apply_colormap(array[..., 0], cmap_name=cmap)
                png_path = os.path.join(output_dir, f"{safe_key}_colormap.png")
                cv2.imwrite(png_path, color_img)
                logging.info(f"Saved numeric array '{original_name}' (1-channel) with colormap to {png_path}.")

            elif array.ndim == 3 and array.shape[2] == 3:
                # 3-channel numeric array => compose into single image
                composed = compose_3channel(array)
                png_path = os.path.join(output_dir, f"{safe_key}.png")
                cv2.imwrite(png_path, composed)
                logging.info(
                    f"Saved numeric 3-channel array '{original_name}' as single image {png_path}."
                )
            else:
                # Not easily visualized
                logging.debug(
                    f"Skipping PNG for '{full_key}' (original='{original_name}'), shape {array.shape} "
                    " - not 2D or 3D with 1/3 channels."
                )
    else:
        logging.debug("PNG saving skipped (save_png=False).")

def load_input(input_path, array_files, preprocess_params):
    if input_path.endswith(".npz"):
        with np.load(input_path) as data:
            available_keys = list(data.keys())
            if not array_files:
                logging.info("No --array_files specified. Using all available files in the NPZ:")
                for key in available_keys:
                    logging.info(f" - {key}")
                array_files = available_keys
            else:
                missing_keys = [key for key in array_files if key not in available_keys]
                if missing_keys:
                    logging.error("The following keys are not available in the NPZ file:")
                    for key in missing_keys:
                        logging.error(f" - {key}")
                    logging.error("Available keys are:")
                    for key in available_keys:
                        logging.error(f" - {key}")
                    exit(1)
            pipeline_data = PipelineData.from_dict({key: data[key] for key in array_files})
    else:
        from skimage.io import imread
        pipeline_data = PipelineData(rgb=imread(input_path))

    shadow_angle = preprocess_params.get("shadow_angle", 0)
    delta_lat = preprocess_params.get("delta_lat", 0)
    delta_lon = preprocess_params.get("delta_lon", 0)

    if shadow_angle or delta_lat or delta_lon:
        pipeline_data.preprocess(shadow_angle=shadow_angle, delta_lat=delta_lat, delta_lon=delta_lon)

    return pipeline_data

def parse_kwargs(kwargs_list):
    kwargs = {}
    for item in kwargs_list:
        key, value = item.split("=", 1)
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            pass
        kwargs[key] = value
    return kwargs

def main():
    args = parse_args()
    setup_logging(args.verbose)

    if args.list_projections:
        list_projections()
        return

    if args.list_samplers:
        list_samplers()
        return

    if args.list_files:
        if not args.input or not args.input.endswith(".npz"):
            logging.error("Please provide a valid NPZ input to list files.")
            return
        list_npz_files(args.input)
        return

    if args.show_pipeline:
        if not args.projection_name:
            logging.error("Missing required argument: --projection_name. Use --help for more information.")
            return
        pipeline = ProjectionPipeline(projection_name=args.projection_name, sampler_name=args.sampler_name)
        logging.info("Instantiated Pipeline Object:")
        logging.info(repr(pipeline))
        return

    if not args.input or not args.projection_name:
        logging.error("Missing required arguments. Use --help for more information.")
        return

    # Parse extra kwargs
    kwargs = parse_kwargs(args.kwargs)
    preprocess_params = {
        "shadow_angle": kwargs.pop("shadow_angle", 0),
        "delta_lat": kwargs.pop("delta_lat", 0),
        "delta_lon": kwargs.pop("delta_lon", 0),
    }

    # Prepare output directory and metadata
    output_dir = create_unique_output_dir(args.output_dir, args)
    save_metadata(output_dir, args)

    # Load input data
    input_data = load_input(args.input, args.array_files, preprocess_params)

    # Create pipeline
    pipeline = ProjectionPipeline(projection_name=args.projection_name, sampler_name=args.sampler_name)

    # Main operations
    if args.operation == "project":
        result = pipeline.project(data=input_data, **kwargs)
        save_output(
            result=result,
            output_dir=output_dir,
            save_npz=args.save_npz,
            operation="project",
            save_png=args.save_png,
            cmap=args.cmap
        )

    elif args.operation == "backward":
        result = pipeline.backward(data=input_data.as_dict(), **kwargs)
        save_output(
            result=result,
            output_dir=output_dir,
            save_npz=args.save_npz,
            operation="backward",
            save_png=args.save_png,
            cmap=args.cmap
        )

    else:
        # If no operation is specified, do both (project + backward)
        projection_result = pipeline.project(data=input_data, **kwargs)
        save_output(
            result=projection_result,
            output_dir=output_dir,
            save_npz=args.save_npz,
            operation="project",
            save_png=args.save_png,
            cmap=args.cmap
        )

        backward_result = pipeline.backward(data=projection_result, **kwargs)
        save_output(
            result=backward_result,
            output_dir=output_dir,
            save_npz=args.save_npz,
            operation="backward",
            save_png=args.save_png,
            cmap=args.cmap
        )

if __name__ == "__main__":
    main()