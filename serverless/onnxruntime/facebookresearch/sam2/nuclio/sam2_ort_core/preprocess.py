"""Image preprocessing for the SAM2 ONNX Runtime encoder.

Reproduces the PyTorch ``SAM2Transforms`` pipeline using PIL + NumPy only:
RGB conversion, bilinear resize to ``input_size``, ``/255``, ImageNet
normalization, and HWC -> NCHW with ``float32`` dtype.
"""

from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import Float32, jaxtyped
from PIL import Image

# ImageNet normalization constants (matches SAM2 SAM2Transforms).
IMAGENET_MEAN: Float32[np.ndarray, "3"] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: Float32[np.ndarray, "3"] = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@jaxtyped(typechecker=beartype)
def preprocess_image(image: Image.Image, input_size: int = 1024) -> Float32[np.ndarray, "1 3 h w"]:
    """Preprocess a PIL image into the encoder input tensor.

    Steps:
        1. Convert to RGB.
        2. Bilinear resize to ``(input_size, input_size)``.
        3. Scale to ``[0, 1]`` (divide by 255).
        4. ImageNet normalization ``(x - mean) / std``.
        5. HWC -> NCHW, ``float32``.

    Returns:
        Array of shape ``[1, 3, input_size, input_size]`` with dtype ``float32``.
    """
    rgb = image.convert("RGB")
    resized = rgb.resize((input_size, input_size), resample=Image.Resampling.BILINEAR)

    arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC, [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD  # broadcast over channel axis
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = arr[np.newaxis, ...]  # add batch dim -> NCHW

    return np.ascontiguousarray(arr, dtype=np.float32)
