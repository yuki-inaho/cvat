"""手順15: beartype / jaxtyping boundary tests.

public API に @beartype / @jaxtyped を付与し、型・shape 違反が早期に
例外として出ることを確認する。
"""

from __future__ import annotations

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import Float32, jaxtyped
from PIL import Image

from sam2_ort_core.preprocess import preprocess_image
from sam2_ort_core.serialize import serialize_embedding


def test_preprocess_rejects_ndarray_input() -> None:
    """preprocess は PIL.Image を要求する。4ch ndarray を渡すと境界 check が拒否する.

    @jaxtyped(typechecker=beartype) は jaxtyping.TypeCheckError (TypeError 派生) を
    送出するため TypeError で受ける。
    """
    arr_4ch = np.zeros((64, 64, 4), dtype=np.float32)  # channel数4の配列
    with pytest.raises(TypeError):
        preprocess_image(arr_4ch)  # type: ignore[arg-type]


def test_preprocess_rejects_wrong_input_size_type() -> None:
    """input_size に非 int を渡すと境界 check が拒否する."""
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    with pytest.raises(TypeError):
        preprocess_image(img, input_size="1024")  # type: ignore[arg-type]


def test_serialize_rejects_wrong_type() -> None:
    """serialize_embedding は Sam2ImageEmbedding を要求する."""
    with pytest.raises(BeartypeCallHintParamViolation):
        serialize_embedding({"not": "an embedding"})  # type: ignore[arg-type]


def test_jaxtyping_rejects_wrong_channel_count() -> None:
    """jaxtyping shape annotation が 3ch 以外の入力を境界で拒否する.

    境界に jaxtyping を集約していることの確認用に、shape 注釈付き関数へ
    channel 数 4 の tensor を渡して TypeError 系の例外が出ることを示す。
    """

    @jaxtyped(typechecker=__import__("beartype").beartype)
    def _accepts_3ch(x: Float32[np.ndarray, "1 3 h w"]) -> int:
        return int(x.shape[1])

    wrong = np.zeros((1, 4, 1024, 1024), dtype=np.float32)  # channel数4
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - jaxtyping raises beartype violation
        _accepts_3ch(wrong)
    assert "3" in str(excinfo.value) or "shape" in str(excinfo.value).lower()


def test_jaxtyping_rejects_wrong_dtype() -> None:
    """Float32 注釈が float64 入力を拒否する."""

    @jaxtyped(typechecker=__import__("beartype").beartype)
    def _accepts_f32(x: Float32[np.ndarray, "1 3 h w"]) -> int:
        return int(x.shape[1])

    wrong_dtype = np.zeros((1, 3, 1024, 1024), dtype=np.float64)
    with pytest.raises(Exception):  # noqa: B017, PT011
        _accepts_f32(wrong_dtype)
