# Projection Pipeline README

## Overview

The `ProjectionPipeline` is a modular framework for managing sampling, projection, and blending strategies. It is designed to process and combine rectilinear projections into a seamless equirectangular image, leveraging configurable components for flexibility and performance.

This README explains the workings of the `backward_with_sampler` method, which handles the core blending logic for combining multiple backprojected images into a single equirectangular image. It also details the solution implemented to resolve issues with visible edges in overlapping regions.

---

## Problem Statement

When projecting rectilinear images onto an equirectangular space, overlapping regions often lead to visible seams or marks at the edges of the projections. This issue is particularly noticeable when:
1. Overlapping areas are not blended smoothly.
2. Weight maps reinforce abrupt transitions instead of smoothing them.
3. Edge regions dominate the blending process.

---

## Solution

The updated blending logic in the `backward_with_sampler` method employs **feathering** to ensure smooth transitions between overlapping regions. The primary steps and techniques used are detailed below.

---

## How the Blending Works

### 1. **Dynamic Mask Creation**
Each backprojected image (`eq_img`) is analyzed to create a dynamic mask. Pixels with non-zero values in the image are considered valid. This mask isolates the regions of interest in each projection.

```python
valid_mask = np.max(eq_img > 0, axis=-1).astype(np.float32)
```

---

### 2. **Edge Feathering**
To ensure smooth blending near the edges, a **distance transform** is applied to the mask. This computes the distance of each pixel from the nearest zero pixel, creating a gradient that emphasizes the center of the projection.

```python
from scipy.ndimage import distance_transform_edt
distance = distance_transform_edt(valid_mask)
feathered_mask = distance / distance.max()  # Normalize to [0, 1]
```

This feathered mask ensures that contributions near the edges fade smoothly, reducing the visibility of transitions between projections.

---

### 3. **Weighted Accumulation**
Each projection's contribution to the final image is scaled by the feathered mask. This approach gives more weight to the central regions of each projection, which are less prone to distortions or inaccuracies.

```python
combined += eq_img * feathered_mask[..., None]
weight_map += feathered_mask
```

---

### 4. **Normalization**
After accumulating all projections, the `combined` image is normalized using the `weight_map`. This step ensures that the final pixel values are scaled appropriately based on the total contribution from overlapping projections.

```python
valid_weights = weight_map > 0
combined[valid_weights] /= weight_map[valid_weights, None]
combined[~valid_weights] = 0
```

---

### 5. **Output Handling**
The method ensures that the resulting equirectangular image can be unstacked into its original components if the input was in a stacked format. This allows for flexibility in downstream tasks.

```python
if self._original_data is not None and self._keys_order is not None:
    new_data = self._original_data.unstack_new_instance(combined, self._keys_order)
    output = {"stacked": combined}
    output.update(new_data.as_dict())
    return output
else:
    return {"stacked": combined}
```

---

## Advantages of the Solution

1. **Seamless Blending:** Feathering reduces the visibility of seams by creating smooth transitions between overlapping regions.
2. **Edge-Aware Weighting:** Central regions of projections are prioritized, minimizing the impact of distorted edge areas.
3. **Dynamic Masking:** Each projection's mask is computed dynamically, ensuring robustness against irregular input data.
4. **Normalization:** Ensures pixel values are consistent and free of artifacts or NaNs.

---

## Expected Results

- **No Visible Edges:** Overlapping regions blend smoothly without harsh transitions or visible seams.
- **Improved Quality:** Central regions dominate the blending process, producing a cleaner and more accurate equirectangular image.
- **Efficiency:** The method leverages parallel processing for fast backprojection and blending.

---

## How to Use

1. Initialize the `ProjectionPipeline` with the desired configuration:
   ```python
   pipeline = ProjectionPipeline(projection_name="my_projection", sampler_name="my_sampler")
   ```

2. Provide rectilinear projections in the required format and call the `backward` method:
   ```python
   result = pipeline.backward(rect_data=my_rect_data, img_shape=(1024, 2048, 3))
   ```

3. Access the combined equirectangular image from the output:
   ```python
   final_image = result["stacked"]
   ```

---

## Future Improvements

1. **Customizable Feathering:** Allow users to specify the feathering parameters, such as the distance transform weight.
2. **Multi-Pass Blending:** Introduce iterative blending to refine transitions further in complex datasets.
3. **Edge-Aware Smoothing:** Incorporate advanced techniques like bilateral filtering for better handling of high-contrast edges.

---

## Contact

For questions or suggestions, please reach out to the maintainers. Contributions are welcome to further enhance this framework.

**Happy Coding!**