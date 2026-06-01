"""手順18: encoder init-time provider self-test.

`Sam2OrtEncoder.__init__` を通した end-to-end の self-test を検証する:

- test mode (require_gpu=false): CPU-only provider でも init 成功し、
  §4.3 の IO self-test ログ (Input/Output の name+shape) が出る。
- production mode (require_gpu=true): CUDA が available に無ければ init で失敗。

provider 単体の `validate_provider` (test_provider.py) や encoder の shape
mapping (test_encoder.py) とは別観点 = 「起動時 self-test の成功/失敗」に絞る。
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from sam2_ort_core import encoder as encoder_mod
from sam2_ort_core.config import Sam2OrtConfig


class _FakeIO:
    def __init__(self, name: str, shape: list, type_: str) -> None:
        self.name = name
        self.shape = shape
        self.type = type_


class _FakeSession:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_inputs(self) -> list[_FakeIO]:
        return [_FakeIO("image", [1, 3, 1024, 1024], "tensor(float)")]

    def get_outputs(self) -> list[_FakeIO]:
        return [
            _FakeIO("high_res_features1", [1, 32, 256, 256], "tensor(float)"),
            _FakeIO("high_res_features2", [1, 64, 128, 128], "tensor(float)"),
            _FakeIO("image_embeddings", [1, 256, 64, 64], "tensor(float)"),
        ]

    def get_providers(self) -> list[str]:
        return self._providers

    def run(self, output_names, input_feed):  # noqa: ANN001
        return [
            np.ones((1, 32, 256, 256), dtype=np.float32),
            np.ones((1, 64, 128, 128), dtype=np.float32),
            np.ones((1, 256, 64, 64), dtype=np.float32),
        ]


def _patch(monkeypatch: pytest.MonkeyPatch, *, available: list[str], session_providers: list[str]) -> None:
    monkeypatch.setattr("sam2_ort_core.provider.ort.get_available_providers", lambda: available)
    monkeypatch.setattr(
        encoder_mod,
        "_create_session",
        lambda model_path, providers: _FakeSession(session_providers),
    )


def test_self_test_succeeds_in_test_mode_and_logs_io(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """test mode: CPU-only でも init 成功し、Input/Output の self-test ログが出る."""
    _patch(monkeypatch, available=["CPUExecutionProvider"], session_providers=["CPUExecutionProvider"])
    cfg = Sam2OrtConfig(model_path="/models/encoder.onnx", provider="CPUExecutionProvider", require_gpu=False)

    with caplog.at_level(logging.INFO, logger="sam2-ort"):
        enc = encoder_mod.Sam2OrtEncoder(cfg)

    assert enc.provider == "CPUExecutionProvider"
    text = caplog.text
    assert "[SAM2-ORT] Input: image" in text
    assert "[SAM2-ORT] Output[0]:" in text
    assert "[SAM2-ORT] Output[2]:" in text
    assert "[1, 256, 64, 64]" in text


def test_self_test_fails_in_production_mode_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """production mode: CUDA が available に無ければ init で fail-fast (CPU に落ちない)."""
    _patch(monkeypatch, available=["CPUExecutionProvider"], session_providers=["CPUExecutionProvider"])
    cfg = Sam2OrtConfig(model_path="/models/encoder.onnx", provider="CUDAExecutionProvider", require_gpu=True)

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider not available"):
        encoder_mod.Sam2OrtEncoder(cfg)


def test_self_test_succeeds_in_production_mode_with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """production mode: CUDA が available かつ session も CUDA を使えば init 成功."""
    _patch(
        monkeypatch,
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    cfg = Sam2OrtConfig(model_path="/models/encoder.onnx", provider="CUDAExecutionProvider", require_gpu=True)

    enc = encoder_mod.Sam2OrtEncoder(cfg)
    assert enc.provider == "CUDAExecutionProvider"
