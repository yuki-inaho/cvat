"""手順14: base64 JSON serialization round-trip tests."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from sam2_ort_core.encoder import Sam2ImageEmbedding
from sam2_ort_core.serialize import deserialize_embedding, serialize_embedding


def _make_embedding(provider: str = "CUDAExecutionProvider") -> Sam2ImageEmbedding:
    rng = np.random.default_rng(42)
    return Sam2ImageEmbedding(
        high_res_feats_0=rng.standard_normal((1, 32, 256, 256)).astype(np.float32),
        high_res_feats_1=rng.standard_normal((1, 64, 128, 128)).astype(np.float32),
        image_embed=rng.standard_normal((1, 256, 64, 64)).astype(np.float32),
        provider=provider,
    )


def test_serialize_returns_expected_fields() -> None:
    emb = _make_embedding()
    payload = serialize_embedding(emb)
    assert set(payload.keys()) == {"high_res_feats_0", "high_res_feats_1", "image_embed"}
    for value in payload.values():
        assert isinstance(value, str)
        base64.b64decode(value)  # decodes without error


def test_serialize_roundtrip() -> None:
    emb = _make_embedding()
    payload = serialize_embedding(emb)
    restored = deserialize_embedding(payload, provider=emb.provider)
    assert np.array_equal(restored.high_res_feats_0, emb.high_res_feats_0)
    assert np.array_equal(restored.high_res_feats_1, emb.high_res_feats_1)
    assert np.array_equal(restored.image_embed, emb.image_embed)
    assert restored.provider == emb.provider
    assert restored.high_res_feats_0.dtype == np.float32
    assert restored.high_res_feats_0.shape == (1, 32, 256, 256)
    assert restored.high_res_feats_1.shape == (1, 64, 128, 128)
    assert restored.image_embed.shape == (1, 256, 64, 64)


def test_serialize_matches_reference_encoding() -> None:
    """既存 main.py 互換: np.ascontiguousarray -> tobytes -> b64encode."""
    emb = _make_embedding()
    payload = serialize_embedding(emb)
    expected = base64.b64encode(np.ascontiguousarray(emb.image_embed).tobytes()).decode()
    assert payload["image_embed"] == expected


def test_deserialize_rejects_size_mismatch() -> None:
    """field 名と byte 長が整合しない場合は明示失敗する."""
    emb = _make_embedding()
    payload = serialize_embedding(emb)
    # corrupt: image_embed bytes truncated
    raw = base64.b64decode(payload["image_embed"])
    payload["image_embed"] = base64.b64encode(raw[:-4]).decode()
    with pytest.raises((ValueError, RuntimeError)):
        deserialize_embedding(payload, provider=emb.provider)
