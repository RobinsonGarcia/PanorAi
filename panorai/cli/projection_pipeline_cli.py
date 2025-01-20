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
        epilog="""
Examples:
1. List available projections and samplers:
   $ panorai-cli --list-projections
   $ panorai-cli --list-samplers

2. Perform a forward projection:
   $ panorai-cli --input ../images/eq_sample.png --projection_name=gnomonic --operation project

3. Perform a forward projection with additional preprocessing:
   $ panorai-cli --input ../images/eq_sample.png --projection_name=gnomonic --operation project --kwargs shadow_angle=30

4. Perform backward projection:
   $ panorai-cli --input output.npz --projection_name=gnomonic --operation backward

5. Perform both projection and backward:
   $ panorai-cli --input ../images/eq_sample.png --projection_name=gnomonic

6. List all files in an NPZ input:
   $ panorai-cli --input sample.npz --list-files

7. Show pipeline details:
   $ panorai-cli --projection_name=gnomonic --show-pipeline
"""
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
    parser.add_argument("--output_dir", type=str, default=".cache", help="Base directory to save the output files.")
    parser.add_argument("--save_npz", action="store_true", help="Save results as a single .npz file.")

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

def normalize_array(array):
    """Normalize an array to the range [0, 255] for uint8 saving."""
    if np.min(array) == np.max(array):
        return np.zeros_like(array, dtype=np.uint8)
    normalized = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    return normalized.astype(np.uint8)

def save_output(result, output_dir, save_npz, operation=None):
    operation_suffix = f"_{operation}" if operation else ""

    for point, content in result.items():
        if point == "stacked":
            continue

        point_dir = os.path.join(output_dir, f"{point}{operation_suffix}")
        os.makedirs(point_dir, exist_ok=True)

        if isinstance(content, dict):
            for key, pipeline_data in content.items():
                if isinstance(pipeline_data, PipelineData):
                    for data_type, array in pipeline_data.__dict__.items():
                        if isinstance(array, np.ndarray):
                            image_path = os.path.join(point_dir, f"{key}_{data_type}.png")
                            if data_type == "rgb":
                                # Save RGB directly without conversion
                                cv2.imwrite(image_path, array)
                            else:
                                normalized_array = normalize_array(array)
                                cv2.imwrite(image_path, normalized_array)
                            logging.info(f"Saved {data_type} of {key} in {point} to {image_path}.")
        elif isinstance(content, np.ndarray):
            image_path = os.path.join(point_dir, f"{point}{operation_suffix}.png")
            if content.ndim == 2:  # Grayscale-like arrays
                normalized_array = normalize_array(content)
                cv2.imwrite(image_path, normalized_array)
            else:
                cv2.imwrite(image_path, content)
            logging.info(f"Saved array for {point} to {image_path}.")

    if save_npz:
        filename = f"output{operation_suffix}.npz"
        np.savez_compressed(os.path.join(output_dir, filename), **result)
        logging.info(f"Saved all results to {os.path.join(output_dir, filename)}.")

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

    kwargs = parse_kwargs(args.kwargs)
    preprocess_params = {
        "shadow_angle": kwargs.pop("shadow_angle", 0),
        "delta_lat": kwargs.pop("delta_lat", 0),
        "delta_lon": kwargs.pop("delta_lon", 0),
    }

    output_dir = create_unique_output_dir(args.output_dir, args)
    save_metadata(output_dir, args)

    input_data = load_input(args.input, args.array_files, preprocess_params)
    pipeline = ProjectionPipeline(projection_name=args.projection_name, sampler_name=args.sampler_name)

    if args.operation == "project":
        result = pipeline.project(data=input_data, **kwargs)
        save_output(result, output_dir, args.save_npz, operation="project")
    elif args.operation == "backward":
        result = pipeline.backward(data=input_data.as_dict(), **kwargs)
        save_output(result, output_dir, args.save_npz, operation="backward")
    else:  # Perform both project and backward if no operation specified
        projection_result = pipeline.project(data=input_data, **kwargs)
        save_output(projection_result, output_dir, args.save_npz, operation="project")

        backward_result = pipeline.backward(data=projection_result, **kwargs)
        save_output(backward_result, output_dir, args.save_npz, operation="backward")

if __name__ == "__main__":
    main()