"""手順12: image preprocessing tests."""

import numpy as np
import pytest
from PIL import Image

from sam2_ort_core.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess_image


def test_preprocess_shape_256(rgb_image_256: Image.Image) -> None:
    out = preprocess_image(rgb_image_256)
    assert out.shape == (1, 3, 1024, 1024)
    assert out.dtype == np.float32


def test_preprocess_shape_640x360(rgb_image_640x360: Image.Image) -> None:
    """非正方形画像も 1024x1024 にリサイズされる."""
    out = preprocess_image(rgb_image_640x360)
    assert out.shape == (1, 3, 1024, 1024)
    assert out.dtype == np.float32


def test_preprocess_custom_input_size(rgb_image_256: Image.Image) -> None:
    out = preprocess_image(rgb_image_256, input_size=512)
    assert out.shape == (1, 3, 512, 512)


def test_preprocess_normalization_white() -> None:
    """全白画像 (255) は (1.0 - mean) / std になる."""
    white = Image.fromarray(np.full((1024, 1024, 3), 255, dtype=np.uint8), mode="RGB")
    out = preprocess_image(white)
    expected = (1.0 - IMAGENET_MEAN) / IMAGENET_STD  # shape (3,)
    for c in range(3):
        np.testing.assert_allclose(out[0, c], expected[c], rtol=1e-5, atol=1e-5)


def test_preprocess_normalization_black() -> None:
    """全黒画像 (0) は (0.0 - mean) / std になる."""
    black = Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8), mode="RGB")
    out = preprocess_image(black)
    expected = (0.0 - IMAGENET_MEAN) / IMAGENET_STD
    for c in range(3):
        np.testing.assert_allclose(out[0, c], expected[c], rtol=1e-5, atol=1e-5)


def test_preprocess_converts_non_rgb() -> None:
    """L (grayscale) や RGBA も RGB 化されて 3 channel になる."""
    gray = Image.fromarray(np.full((100, 100), 128, dtype=np.uint8), mode="L")
    out = preprocess_image(gray)
    assert out.shape == (1, 3, 1024, 1024)


def test_preprocess_channel_order_rgb() -> None:
    """RGB の channel 順が保持される (R != G != B のテスト画像で確認)."""
    arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
    arr[..., 0] = 255  # R full
    arr[..., 1] = 0  # G zero
    arr[..., 2] = 0  # B zero
    img = Image.fromarray(arr, mode="RGB")
    out = preprocess_image(img)
    r_expected = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    g_expected = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    assert out[0, 0].mean() == pytest.approx(r_expected, abs=1e-3)
    assert out[0, 1].mean() == pytest.approx(g_expected, abs=1e-3)
