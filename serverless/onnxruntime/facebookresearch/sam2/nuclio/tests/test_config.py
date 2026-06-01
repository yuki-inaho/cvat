"""手順10: Sam2OrtConfig tests."""

import pytest

from sam2_ort_core.config import Sam2OrtConfig


def test_config_requires_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """SAM2_ORT_MODEL_PATH 未設定なら RuntimeError (暗黙 default 禁止)."""
    monkeypatch.delenv("SAM2_ORT_MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="SAM2_ORT_MODEL_PATH"):
        Sam2OrtConfig.from_env()


def test_config_from_env_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """全 env 設定時に値が正しく読み込まれる."""
    monkeypatch.setenv("SAM2_ORT_MODEL_PATH", "/models/encoder.onnx")
    monkeypatch.setenv("SAM2_ORT_PROVIDER", "CUDAExecutionProvider")
    monkeypatch.setenv("SAM2_ORT_REQUIRE_GPU", "true")
    cfg = Sam2OrtConfig.from_env()
    assert cfg.model_path == "/models/encoder.onnx"
    assert cfg.provider == "CUDAExecutionProvider"
    assert cfg.require_gpu is True
    assert cfg.input_size == 1024


def test_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """provider と require_gpu の default 値."""
    monkeypatch.setenv("SAM2_ORT_MODEL_PATH", "/models/encoder.onnx")
    monkeypatch.delenv("SAM2_ORT_PROVIDER", raising=False)
    monkeypatch.delenv("SAM2_ORT_REQUIRE_GPU", raising=False)
    cfg = Sam2OrtConfig.from_env()
    assert cfg.provider == "CUDAExecutionProvider"
    assert cfg.require_gpu is True


def test_config_rejects_implicit_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU 要求 (require_gpu=true) のまま provider に CPU を指定すると矛盾として失敗する."""
    monkeypatch.setenv("SAM2_ORT_MODEL_PATH", "/models/encoder.onnx")
    monkeypatch.setenv("SAM2_ORT_PROVIDER", "CPUExecutionProvider")
    monkeypatch.setenv("SAM2_ORT_REQUIRE_GPU", "true")
    with pytest.raises(RuntimeError, match="implicit CPU fallback|require_gpu"):
        Sam2OrtConfig.from_env()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("True", True), ("1", True), ("false", False), ("0", False), ("no", False)],
)
def test_config_require_gpu_parsing(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    """require_gpu の文字列 → bool パース."""
    monkeypatch.setenv("SAM2_ORT_MODEL_PATH", "/models/encoder.onnx")
    monkeypatch.setenv("SAM2_ORT_REQUIRE_GPU", value)
    # CPU provider + require_gpu=true は矛盾するので provider は default(CUDA) のまま
    monkeypatch.delenv("SAM2_ORT_PROVIDER", raising=False)
    cfg = Sam2OrtConfig.from_env()
    assert cfg.require_gpu is expected
