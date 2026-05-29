"""Shared pytest fixtures for sam2_ort_core tests."""

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rgb_image_256() -> Image.Image:
    """Synthetic 256x256 RGB image."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def rgb_image_640x360() -> Image.Image:
    """Synthetic non-square 640x360 (WxH) RGB image."""
    rng = np.random.default_rng(1)
    # PIL size is (width, height); numpy array is (height, width, channels)
    arr = rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")
