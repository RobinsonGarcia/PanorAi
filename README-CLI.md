# Panorai CLI Guide

The `panorai` command-line interface (CLI) provides a convenient way to perform spherical image processing tasks, such as projection, sampling, and unsharp masking. This guide covers installation, usage, and examples of how to use the CLI effectively.

---

## Installation

To install the `panorai` package, run:

```bash
python setup.py install
```

or

```bash
pip install .
```

After installation, the `panorai` CLI tool will be available globally.

---

## Usage

The general syntax for the CLI is:

```bash
panorai --input <input_file_or_directory> --operation <operation> [options]
```

### Common Options

| Option               | Description                                                                                              |
|----------------------|----------------------------------------------------------------------------------------------------------|
| `--input`            | Path to the input file or directory. Supports `.npz` files or directories with projection outputs.        |
| `--operation`        | The operation to perform. Choices are `project` or `backward`.                                           |
| `--array_files`      | (Optional) Keys for arrays in the `.npz` file (e.g., `rgb`, `depth`). Required for `.npz` inputs.         |
| `--projection_name`  | Name of the projection to use (e.g., `gnomonic`).                                                        |
| `--sampler_name`     | (Optional) Name of the sampler to use (e.g., `CubeSampler`, `FibonacciSampler`).                         |
| `--output_dir`       | Directory to save the output files. Default is `./output`.                                               |
| `--save_npz`         | Save the output as a single `.npz` file.                                                                 |
| `--kwargs`           | Additional parameters for projection/sampling, in the format `key=value`.                                |
| `--list-projections` | List all available projections.                                                                          |
| `--list-samplers`    | List all available samplers.                                                                             |

---

## Examples

### 1. **List Available Projections and Samplers**

List all registered projections:

```bash
panorai --list-projections
```

List all registered samplers:

```bash
panorai --list-samplers
```

### 2. **Forward Projection**

Perform a forward projection using the `gnomonic` projection:

```bash
panorai --input sample.npz --array_files rgb depth --projection_name gnomonic --operation project --output_dir ./output
```

- **Input:** `sample.npz` contains `rgb` and `depth` arrays.
- **Output:** Projected images are saved in the `./output` directory as `.png` files.

### 3. **Save Results as a Single `.npz` File**

Save the projection results into a single `.npz` file:

```bash
panorai --input sample.npz --array_files rgb depth --projection_name gnomonic --operation project --save_npz
```

This creates `output.npz` in the default output directory.

### 4. **Backward Projection**

Reconstruct the equirectangular image from projections stored in a directory:

```bash
panorai --input ./output/ --projection_name gnomonic --operation backward --output_dir ./reconstructed
```

This saves the reconstructed image in the `./reconstructed` directory.

---

## Advanced Options

You can pass additional parameters to the projection or sampling process using the `--kwargs` option. For example:

```bash
panorai --input sample.npz --array_files rgb depth --projection_name gnomonic --operation project --kwargs fov_deg=40 subdivisions=2
```

In this case:
- `fov_deg=40` sets the field of view to 40 degrees.
- `subdivisions=2` specifies 2 subdivisions for the sampler.

---

## Output Formats

1. **Default Output**: Individual `.png` files for each projection or reconstruction result.
2. **Single `.npz` File**: Use the `--save_npz` option to save all results into a single `.npz` file.

---

## Help

To view the full list of options, run:

```bash
panorai --help
```

---

Enjoy using `panorai` for your panoramic image processing tasks!