"""手順11: provider detection / fail-fast tests.

GPU 実機がない環境のため、`ort.get_available_providers` を monkeypatch して
CUDA 有無を制御する。CPU 成功を GPU 成功と偽らないことを担保する。
"""

import pytest

from sam2_ort_core import provider as provider_mod
from sam2_ort_core.config import Sam2OrtConfig


def _make_config(*, require_gpu: bool, provider: str = "CUDAExecutionProvider") -> Sam2OrtConfig:
    return Sam2OrtConfig(
        model_path="/models/encoder.onnx",
        provider=provider,
        require_gpu=require_gpu,
    )


def test_provider_requires_cuda_execution_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU mode で CUDA provider が無ければ RuntimeError (CPU fallback 禁止)."""
    monkeypatch.setattr(
        provider_mod.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    cfg = _make_config(require_gpu=True)
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider not available"):
        provider_mod.validate_provider(cfg)


def test_provider_available_returns_resolved_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA がある場合は requested provider を先頭にした provider list を返す."""
    monkeypatch.setattr(
        provider_mod.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    cfg = _make_config(require_gpu=True)
    resolved = provider_mod.validate_provider(cfg)
    assert resolved[0] == "CUDAExecutionProvider"
    assert "CUDAExecutionProvider" in resolved


def test_provider_cpu_mode_allows_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """require_gpu=false なら CPU-only provider でも通る (明示的 CPU mode)."""
    monkeypatch.setattr(
        provider_mod.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    cfg = _make_config(require_gpu=False, provider="CPUExecutionProvider")
    resolved = provider_mod.validate_provider(cfg)
    assert resolved[0] == "CPUExecutionProvider"


def test_provider_requested_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """要求した provider 自体が利用不可なら RuntimeError."""
    monkeypatch.setattr(
        provider_mod.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    cfg = _make_config(require_gpu=False, provider="TensorrtExecutionProvider")
    with pytest.raises(RuntimeError, match="not available"):
        provider_mod.validate_provider(cfg)
