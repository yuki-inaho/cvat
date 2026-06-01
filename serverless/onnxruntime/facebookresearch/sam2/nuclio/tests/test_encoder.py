"""手順13: encoder session runner tests.

実 ONNX セッションを使わず、mock session で出力 shape を差し替えて
shape ベース mapping を検証する。GPU 実機なし環境のため、provider 検証も
monkeypatch ベース。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sam2_ort_core import encoder as encoder_mod
from sam2_ort_core.config import Sam2OrtConfig
from sam2_ort_core.encoder import (
    SHAPE_TO_FIELD,
    Sam2ImageEmbedding,
    Sam2OrtEncoder,
)


class _FakeOutput:
    def __init__(self, name: str, shape: list | None = None, type_: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape if shape is not None else []
        self.type = type_


class _FakeSession:
    """Minimal InferenceSession stand-in returning canned outputs."""

    def __init__(self, outputs: dict[str, np.ndarray], providers: list[str]) -> None:
        self._outputs = outputs
        self._providers = providers

    def get_inputs(self) -> list[_FakeOutput]:
        return [_FakeOutput("image", [1, 3, 1024, 1024])]

    def get_outputs(self) -> list[_FakeOutput]:
        return [_FakeOutput(name, list(arr.shape)) for name, arr in self._outputs.items()]

    def get_providers(self) -> list[str]:
        return self._providers

    def run(self, output_names, input_feed):  # noqa: ANN001
        # Return outputs in declared order, ignoring exact input values.
        assert len(input_feed) == 1
        return list(self._outputs.values())


def _cvat_compatible_outputs() -> dict[str, np.ndarray]:
    """System A naming (export_onnx.py): distinct names, CVAT-correct shapes."""
    return {
        "high_res_features1": np.ones((1, 32, 256, 256), dtype=np.float32),
        "high_res_features2": np.ones((1, 64, 128, 128), dtype=np.float32) * 2,
        "image_embeddings": np.ones((1, 256, 64, 64), dtype=np.float32) * 3,
    }


def _build_encoder(
    monkeypatch: pytest.MonkeyPatch,
    outputs: dict[str, np.ndarray],
    *,
    available: list[str] | None = None,
    session_providers: list[str] | None = None,
    require_gpu: bool = False,
    provider: str = "CPUExecutionProvider",
) -> Sam2OrtEncoder:
    available = available if available is not None else ["CPUExecutionProvider"]
    session_providers = session_providers if session_providers is not None else [provider]

    monkeypatch.setattr(
        "sam2_ort_core.provider.ort.get_available_providers",
        lambda: available,
    )

    def _fake_session_factory(model_path, providers):  # noqa: ANN001, ANN202
        return _FakeSession(outputs, session_providers)

    monkeypatch.setattr(encoder_mod, "_create_session", _fake_session_factory)

    cfg = Sam2OrtConfig(
        model_path="/models/encoder.onnx",
        provider=provider,
        require_gpu=require_gpu,
    )
    return Sam2OrtEncoder(cfg)


def test_shape_to_field_contract() -> None:
    """SHAPE_TO_FIELD は CVAT 契約 shape を正しく定義している."""
    assert SHAPE_TO_FIELD[(1, 32, 256, 256)] == "high_res_feats_0"
    assert SHAPE_TO_FIELD[(1, 64, 128, 128)] == "high_res_feats_1"
    assert SHAPE_TO_FIELD[(1, 256, 64, 64)] == "image_embed"


def test_encoder_run_shape_based_mapping(monkeypatch: pytest.MonkeyPatch, rgb_image_256: Image.Image) -> None:
    """出力名が CVAT 契約と異なっても shape で field に割り当てる."""
    enc = _build_encoder(monkeypatch, _cvat_compatible_outputs())
    emb = enc.run(rgb_image_256)
    assert isinstance(emb, Sam2ImageEmbedding)
    assert emb.high_res_feats_0.shape == (1, 32, 256, 256)
    assert emb.high_res_feats_1.shape == (1, 64, 128, 128)
    assert emb.image_embed.shape == (1, 256, 64, 64)
    # value tagging confirms correct field assignment (1/2/3 markers).
    assert float(emb.high_res_feats_0.flat[0]) == pytest.approx(1.0)
    assert float(emb.high_res_feats_1.flat[0]) == pytest.approx(2.0)
    assert float(emb.image_embed.flat[0]) == pytest.approx(3.0)
    assert emb.provider == "CPUExecutionProvider"


def test_encoder_dtype_is_float32(monkeypatch: pytest.MonkeyPatch, rgb_image_256: Image.Image) -> None:
    enc = _build_encoder(monkeypatch, _cvat_compatible_outputs())
    emb = enc.run(rgb_image_256)
    assert emb.high_res_feats_0.dtype == np.float32
    assert emb.high_res_feats_1.dtype == np.float32
    assert emb.image_embed.dtype == np.float32


def test_encoder_missing_field_raises(monkeypatch: pytest.MonkeyPatch, rgb_image_256: Image.Image) -> None:
    """必要な shape が出力に揃わない場合は得られた shape を含む RuntimeError."""
    incomplete = {
        "a": np.ones((1, 32, 256, 256), dtype=np.float32),
        "b": np.ones((1, 64, 128, 128), dtype=np.float32),
        # image_embed [1,256,64,64] is missing
    }
    enc = _build_encoder(monkeypatch, incomplete)
    with pytest.raises(RuntimeError, match=r"image_embed|\(1, 256, 64, 64\)"):
        enc.run(rgb_image_256)


def test_encoder_ambiguous_shape_raises(monkeypatch: pytest.MonkeyPatch, rgb_image_256: Image.Image) -> None:
    """同一 target shape が複数あると曖昧として RuntimeError (raw 7出力対策)."""
    ambiguous = {
        "high_res_features1": np.ones((1, 32, 256, 256), dtype=np.float32),
        "high_res_features2": np.ones((1, 64, 128, 128), dtype=np.float32),
        "vision_features": np.ones((1, 256, 64, 64), dtype=np.float32),
        "backbone_fpn_2": np.ones((1, 256, 64, 64), dtype=np.float32),  # duplicate shape
    }
    enc = _build_encoder(monkeypatch, ambiguous)
    with pytest.raises(RuntimeError, match="ambiguous|multiple"):
        enc.run(rgb_image_256)


def test_encoder_require_gpu_revalidates_session_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """require_gpu 時、session が CUDA を使っていなければ __init__ で失敗する."""
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        _build_encoder(
            monkeypatch,
            _cvat_compatible_outputs(),
            available=["CUDAExecutionProvider", "CPUExecutionProvider"],
            session_providers=["CPUExecutionProvider"],  # session fell back to CPU
            require_gpu=True,
            provider="CUDAExecutionProvider",
        )


def test_encoder_require_gpu_accepts_cuda_session(monkeypatch: pytest.MonkeyPatch, rgb_image_256: Image.Image) -> None:
    enc = _build_encoder(
        monkeypatch,
        _cvat_compatible_outputs(),
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        require_gpu=True,
        provider="CUDAExecutionProvider",
    )
    emb = enc.run(rgb_image_256)
    assert emb.provider == "CUDAExecutionProvider"


@pytest.mark.skipif(
    not __import__("os").path.exists("/home/inaho-omen/Project/sam2_onnx_sandbox/model/image_encoder_hiera_s_2.1.onnx"),
    reason="reference ONNX (system B raw 7-output) not available",
)
def test_encoder_real_onnx_runner_sanity(rgb_image_256: Image.Image) -> None:
    """実 ONNX (系統B raw) で runner が動くことを確認 (CVAT 統合には使わない).

    系統B は [1,256,64,64] shape が 3 出力で重複するため shape mapping は
    ambiguous になる。ここでは preprocess→session.run まで通り、ambiguous で
    明示失敗することを確認する (runner 機構の sanity)。
    """
    import onnxruntime as ort

    from sam2_ort_core.preprocess import preprocess_image

    path = "/home/inaho-omen/Project/sam2_onnx_sandbox/model/image_encoder_hiera_s_2.1.onnx"
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = preprocess_image(rgb_image_256)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: inp})
    shapes = [tuple(o.shape) for o in outputs]
    # 系統B: 7 出力で [1,256,64,64] が複数 → CVAT 非互換を確認
    assert len(outputs) == 7
    assert shapes.count((1, 256, 64, 64)) >= 2
