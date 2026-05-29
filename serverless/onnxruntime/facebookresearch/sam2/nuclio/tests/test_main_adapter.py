"""手順16: Nuclio main.py adapter tests.

handler は event parse + encoder 呼び出し + serialize + response のみを行う
薄い adapter であることを検証する。実 ONNX / 実 GPU は不要で、encoder は
monkeypatch で差し替える。

main.py は nuclio ディレクトリ直下にあり (handler: main:handler)、pytest の
testpaths=["tests"] からは直接 import できないため、親ディレクトリを sys.path
へ追加してから import する。
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# main.py は nuclio dir 直下 (tests/ の親)。Nuclio runtime では PYTHONPATH=
# /opt/nuclio/sam2 で解決されるが、test では親 dir を sys.path に通す。
_NUCLIO_DIR = Path(__file__).resolve().parent.parent
if str(_NUCLIO_DIR) not in sys.path:
    sys.path.insert(0, str(_NUCLIO_DIR))

import main  # noqa: E402

from sam2_ort_core.encoder import Sam2ImageEmbedding  # noqa: E402


class _FakeLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.errors: list[str] = []

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)


class _FakeResponse:
    def __init__(self, *, body, headers, content_type, status_code) -> None:
        self.body = body
        self.headers = headers
        self.content_type = content_type
        self.status_code = status_code


class _FakeUserData:
    pass


class _FakeContext:
    """Minimal stand-in for the Nuclio context object."""

    def __init__(self) -> None:
        self.logger = _FakeLogger()
        self.user_data = _FakeUserData()

    @staticmethod
    def Response(*, body, headers, content_type, status_code) -> _FakeResponse:  # noqa: N802
        return _FakeResponse(body=body, headers=headers, content_type=content_type, status_code=status_code)


class _FakeEvent:
    def __init__(self, body) -> None:
        self.body = body


class _FakeEncoder:
    """Returns a fixed, CVAT-shaped embedding without any ONNX session."""

    provider = "CUDAExecutionProvider"

    def run(self, image: Image.Image) -> Sam2ImageEmbedding:
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        return Sam2ImageEmbedding(
            high_res_feats_0=np.ones((1, 32, 256, 256), dtype=np.float32),
            high_res_feats_1=np.ones((1, 64, 128, 128), dtype=np.float32),
            image_embed=np.ones((1, 256, 64, 64), dtype=np.float32),
            provider="CUDAExecutionProvider",
        )


def _png_base64() -> str:
    image = Image.new("RGB", (64, 64), "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_handler_success_returns_three_fields() -> None:
    """有効な base64 PNG → status 200 + 3 field の JSON dict."""
    ctx = _FakeContext()
    ctx.user_data.encoder = _FakeEncoder()
    event = _FakeEvent({"image": _png_base64()})

    resp = main.handler(ctx, event)

    assert resp.status_code == 200
    assert resp.content_type == "application/json"
    payload = json.loads(resp.body)
    assert set(payload.keys()) == {"high_res_feats_0", "high_res_feats_1", "image_embed"}
    # base64 decodes to the expected float32 byte lengths.
    assert len(base64.b64decode(payload["high_res_feats_0"])) == 1 * 32 * 256 * 256 * 4
    assert len(base64.b64decode(payload["high_res_feats_1"])) == 1 * 64 * 128 * 128 * 4
    assert len(base64.b64decode(payload["image_embed"])) == 1 * 256 * 64 * 64 * 4


def test_handler_invalid_body_returns_500() -> None:
    """body に image key が無い → status 500 + error JSON。"""
    ctx = _FakeContext()
    ctx.user_data.encoder = _FakeEncoder()
    event = _FakeEvent({"not_image": "x"})

    resp = main.handler(ctx, event)

    assert resp.status_code == 500
    assert json.loads(resp.body) == {"error": "Internal server error"}
    assert ctx.logger.errors  # error was logged


def test_handler_invalid_base64_returns_500() -> None:
    """image が PNG として開けない不正データ → status 500。"""
    ctx = _FakeContext()
    ctx.user_data.encoder = _FakeEncoder()
    event = _FakeEvent({"image": base64.b64encode(b"not a png").decode()})

    resp = main.handler(ctx, event)

    assert resp.status_code == 500
    assert json.loads(resp.body) == {"error": "Internal server error"}


def test_init_context_builds_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """init_context は Sam2OrtConfig.from_env と Sam2OrtEncoder を使い、
    encoder を user_data に格納する (provider self-test の発火点)。"""
    built: dict[str, object] = {}

    sentinel_config = object()
    monkeypatch.setattr(main.Sam2OrtConfig, "from_env", classmethod(lambda cls: sentinel_config))

    def _fake_encoder_ctor(config):  # noqa: ANN001, ANN202
        built["config"] = config
        return _FakeEncoder()

    monkeypatch.setattr(main, "Sam2OrtEncoder", _fake_encoder_ctor)

    ctx = _FakeContext()
    main.init_context(ctx)

    assert built["config"] is sentinel_config
    assert isinstance(ctx.user_data.encoder, _FakeEncoder)
    # provider is surfaced in the init log line.
    assert any("CUDAExecutionProvider" in line for line in ctx.logger.infos)


def test_handler_holds_no_inference_logic() -> None:
    """adapter は推論 logic を持たない: encoder.run の戻り値をそのまま serialize
    するだけで、handler 自身は前処理/ONNX 実行を行わない (encoder 未設定なら 500)。"""
    ctx = _FakeContext()  # user_data.encoder を意図的に設定しない
    event = _FakeEvent({"image": _png_base64()})

    resp = main.handler(ctx, event)

    # encoder が無ければ AttributeError → 500 (handler 内に推論経路を持たない証拠)
    assert resp.status_code == 500
