import argparse
import os
import numpy as np
from panorai.pipeline.pipeline import ProjectionPipeline
from panorai.pipeline.pipeline_data import PipelineData
from panorai.submodules.projections import ProjectionRegistry
from panorai.sampler.registry import SamplerRegistry


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
"""
    )

    # Utility options
    parser.add_argument("--list-projections", action="store_true", help="List all available projections.")
    parser.add_argument("--list-samplers", action="store_true", help="List all available samplers.")
    parser.add_argument("--list-files", action="store_true", help="List all files inside the provided NPZ input.")

    # Input parameters
    parser.add_argument("--input", type=str, help="Path to the input file or directory.")
    parser.add_argument("--array_files", type=str, nargs="*", help="Keys for data in the .npz file (e.g., rgb, depth).")

    # Projection parameters
    parser.add_argument("--projection_name", type=str, help="Name of the projection to use.")
    parser.add_argument("--sampler_name", type=str, default=None, help="Name of the sampler to use (optional).")
    parser.add_argument("--operation", choices=["project", "backward"], help="Operation to perform.")
    parser.add_argument("--kwargs", nargs="*", default=[], help="Additional arguments for the operation in key=value format.")

    # Output options
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the output files.")
    parser.add_argument("--save_npz", action="store_true", help="Save results as a single .npz file.")

    return parser.parse_args()


def list_projections():
    print("Available Projections:")
    for projection in ProjectionRegistry.list_projections():
        print(f" - {projection}")


def list_samplers():
    print("Available Samplers:")
    for sampler in SamplerRegistry.list_samplers():
        print(f" - {sampler}")


def list_npz_files(input_path):
    with np.load(input_path) as data:
        print("Files in NPZ:")
        for key in data.keys():
            print(f" - {key}")


def save_output(result, output_dir, save_npz):
    os.makedirs(output_dir, exist_ok=True)
    if save_npz:
        np.savez_compressed(os.path.join(output_dir, "output.npz"), **result)
        print(f"Saved all results to {os.path.join(output_dir, 'output.npz')}")
    else:
        for point, v in result.items():
            if point != "stacked":
                if isinstance(v, np.ndarray):
                    output_path = os.path.join(output_dir, f"arr_{point}.png")
                    from skimage.io import imsave
                    imsave(output_path, v.astype(np.uint8))
                    print(f"Saved arr_{point} to {output_path}")
                    continue

                for key, value in v.items():
                    output_path = os.path.join(output_dir, f"{point}_{key}.png")
                    from skimage.io import imsave
                    imsave(output_path, value.astype(np.uint8))
                    print(f"Saved {key} to {output_path}")

def load_input(input_path, array_files, preprocess_params):
    """
    Load input data and optionally preprocess it.

    Args:
        input_path (str): Path to the input file or directory.
        array_files (list): Keys for data in the .npz file.
        preprocess_params (dict): Parameters for preprocessing (shadow_angle, delta_lat, delta_lon).

    Returns:
        PipelineData: The loaded and optionally preprocessed data.
    """
    if input_path.endswith(".npz"):
        with np.load(input_path) as data:
            available_keys = list(data.keys())
            if not array_files:
                # If no array_files are provided, use all available files
                print("No --array_files specified. Using all available files in the NPZ:")
                for key in available_keys:
                    print(f" - {key}")
                array_files = available_keys
            else:
                # Check if all specified keys exist in the NPZ file
                missing_keys = [key for key in array_files if key not in available_keys]
                if missing_keys:
                    print("Error: The following keys are not available in the NPZ file:")
                    for key in missing_keys:
                        print(f" - {key}")
                    print("Available keys are:")
                    for key in available_keys:
                        print(f" - {key}")
                    exit(1)  # Exit gracefully with an error message
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

    if args.list_projections:
        list_projections()
        return

    if args.list_samplers:
        list_samplers()
        return

    if args.list_files:
        if not args.input or not args.input.endswith(".npz"):
            print("Please provide a valid NPZ input to list files.")
            return
        list_npz_files(args.input)
        return

    if not args.input or not args.projection_name:
        print("Missing required arguments. Use --help for more information.")
        return

    kwargs = parse_kwargs(args.kwargs)
    preprocess_params = {
        "shadow_angle": kwargs.pop("shadow_angle", 0),
        "delta_lat": kwargs.pop("delta_lat", 0),
        "delta_lon": kwargs.pop("delta_lon", 0),
    }

    input_data = load_input(args.input, args.array_files, preprocess_params)
    pipeline = ProjectionPipeline(projection_name=args.projection_name, sampler_name=args.sampler_name)

    if args.operation == "project":
        result = pipeline.project(data=input_data, **kwargs)
        save_output(result, args.output_dir, args.save_npz)
    elif args.operation == "backward":
        result = pipeline.backward(data=input_data.as_dict(), **kwargs)
        save_output(result, args.output_dir, args.save_npz)
    else:  # Perform both project and backward if no operation specified
        projection_result = pipeline.project(data=input_data, **kwargs)
        backward_result = pipeline.backward(data=projection_result, **kwargs)
        save_output(backward_result, args.output_dir, args.save_npz)


if __name__ == "__main__":
    main()