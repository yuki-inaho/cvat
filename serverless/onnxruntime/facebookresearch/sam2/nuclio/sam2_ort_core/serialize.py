"""base64 JSON serialization for SAM2 embeddings (existing CVAT UI compatible).

Matches the reference PyTorch ``main.py`` encoding:
``base64.b64encode(np.ascontiguousarray(arr).tobytes())``.
"""

from __future__ import annotations

import base64

import numpy as np
from beartype import beartype

from sam2_ort_core.encoder import Sam2ImageEmbedding

# CVAT response field -> expected (shape, dtype). Shapes are the fixed contract
# from the frontend decoder (inference.worker.ts).
_FIELD_SPEC: dict[str, tuple[tuple[int, ...], type[np.floating]]] = {
    "high_res_feats_0": ((1, 32, 256, 256), np.float32),
    "high_res_feats_1": ((1, 64, 128, 128), np.float32),
    "image_embed": ((1, 256, 64, 64), np.float32),
}


@beartype
def serialize_embedding(embedding: Sam2ImageEmbedding) -> dict[str, str]:
    """Serialize an embedding into the CVAT base64 JSON response body.

    Returns a dict with the three CVAT response fields (provider metadata is
    intentionally not part of the wire contract).
    """
    return {
        "high_res_feats_0": _encode(embedding.high_res_feats_0),
        "high_res_feats_1": _encode(embedding.high_res_feats_1),
        "image_embed": _encode(embedding.image_embed),
    }


@beartype
def deserialize_embedding(data: dict[str, str], provider: str) -> Sam2ImageEmbedding:
    """Reconstruct an embedding from a serialized CVAT response body.

    Raises:
        ValueError: if a field is missing or its byte length is inconsistent
            with the expected shape/dtype.
    """
    fields: dict[str, np.ndarray] = {}
    for field, (shape, dtype) in _FIELD_SPEC.items():
        if field not in data:
            raise ValueError(f"Missing field {field!r} in serialized payload.")
        raw = base64.b64decode(data[field])
        arr = np.frombuffer(raw, dtype=dtype)
        expected_count = int(np.prod(shape))
        if arr.size != expected_count:
            raise ValueError(
                f"Field {field!r}: decoded {arr.size} elements but expected {expected_count} for shape {shape}."
            )
        fields[field] = np.ascontiguousarray(arr.reshape(shape))

    return Sam2ImageEmbedding(
        high_res_feats_0=fields["high_res_feats_0"],
        high_res_feats_1=fields["high_res_feats_1"],
        image_embed=fields["image_embed"],
        provider=provider,
    )


def _encode(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode()
