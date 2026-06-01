"""SAM2 ONNX Runtime encoder runner (Nuclio-independent core logic).

Maps ONNX encoder outputs to CVAT response fields **by shape** so that
differing ONNX output names (e.g. ``export_onnx.py`` emits
``image_embeddings`` / ``high_res_features1`` / ``high_res_features2``) are
absorbed without hard-coded name assumptions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from beartype import beartype
from PIL import Image

from sam2_ort_core.config import Sam2OrtConfig
from sam2_ort_core.preprocess import preprocess_image
from sam2_ort_core.provider import validate_provider

logger = logging.getLogger("sam2-ort")

_CUDA_PROVIDER = "CUDAExecutionProvider"

# CVAT response field assignment keyed by output shape (§12.4).
SHAPE_TO_FIELD: dict[tuple[int, ...], str] = {
    (1, 32, 256, 256): "high_res_feats_0",
    (1, 64, 128, 128): "high_res_feats_1",
    (1, 256, 64, 64): "image_embed",
}


@dataclass(frozen=True)
class Sam2ImageEmbedding:
    """Encoder outputs mapped to CVAT response fields plus provider metadata."""

    high_res_feats_0: np.ndarray  # [1, 32, 256, 256] float32
    high_res_feats_1: np.ndarray  # [1, 64, 128, 128] float32
    image_embed: np.ndarray  # [1, 256, 64, 64] float32
    provider: str


def _create_session(model_path: str, providers: list[str]) -> ort.InferenceSession:
    """Create an ONNX Runtime InferenceSession.

    Isolated as a module-level seam so tests can substitute a fake session
    without a real GPU or ONNX model file.
    """
    return ort.InferenceSession(model_path, providers=providers)


class Sam2OrtEncoder:
    """Runs the SAM2 encoder ONNX model and returns a :class:`Sam2ImageEmbedding`."""

    def __init__(self, config: Sam2OrtConfig) -> None:
        self._config = config
        resolved_providers = validate_provider(config)
        self._session = _create_session(config.model_path, resolved_providers)
        self._input_name = self._session.get_inputs()[0].name
        self._active_providers = list(self._session.get_providers())

        # Provider self-test log (§4.3): emit provider, model path, and the
        # model's declared input/output names + shapes at init time so that
        # `just sam2-ort-logs` can confirm CUDAExecutionProvider and IO contract.
        logger.info("[SAM2-ORT] Model: %s", config.model_path)
        for inp in self._session.get_inputs():
            logger.info("[SAM2-ORT] Input: %s %s %s", inp.name, inp.shape, inp.type)
        for idx, out in enumerate(self._session.get_outputs()):
            logger.info("[SAM2-ORT] Output[%d]: %s %s %s", idx, out.name, out.shape, out.type)
        logger.info("[SAM2-ORT] Session providers: %s", self._active_providers)

        # Re-validate: do not let a session silently fall back to CPU when GPU
        # is required (CUDA may be "available" yet unused by the session).
        if config.require_gpu and _CUDA_PROVIDER not in self._active_providers:
            raise RuntimeError(
                "GPU mode required but the InferenceSession is using "
                f"{self._active_providers}. CUDAExecutionProvider is not active. "
                "Refusing to run on CPU."
            )

        self._provider = self._active_providers[0] if self._active_providers else config.provider

    @property
    def provider(self) -> str:
        """The active execution provider reported by the InferenceSession."""
        return self._provider

    @beartype
    def run(self, image: Image.Image) -> Sam2ImageEmbedding:
        """Preprocess, run inference, and map outputs to CVAT fields by shape."""
        input_tensor = preprocess_image(image, input_size=self._config.input_size)
        outputs = self._session.run(None, {self._input_name: input_tensor})

        mapped = self._map_outputs_by_shape(outputs)
        return Sam2ImageEmbedding(
            high_res_feats_0=mapped["high_res_feats_0"],
            high_res_feats_1=mapped["high_res_feats_1"],
            image_embed=mapped["image_embed"],
            provider=self._provider,
        )

    @staticmethod
    def _map_outputs_by_shape(outputs: Sequence[object]) -> dict[str, np.ndarray]:
        # ``InferenceSession.run`` returns ``Sequence[ndarray | SparseTensor |
        # list | dict]``; the SAM2 encoder only emits dense arrays, and every
        # element is coerced via ``np.asarray`` below, so accept ``object``.
        observed_shapes = [tuple(int(d) for d in np.asarray(o).shape) for o in outputs]
        mapped: dict[str, np.ndarray] = {}

        for arr, shape in zip(outputs, observed_shapes):
            field = SHAPE_TO_FIELD.get(shape)
            if field is None:
                continue
            if field in mapped:
                raise RuntimeError(
                    f"Ambiguous encoder outputs: multiple tensors match shape "
                    f"{shape} -> {field!r}. Observed shapes: {observed_shapes}. "
                    "This ONNX is likely a raw (no_mem_embed-unadded) encoder that "
                    "is NOT CVAT-compatible; use the export_onnx.py 3-output encoder."
                )
            mapped[field] = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))

        missing = [field for field in SHAPE_TO_FIELD.values() if field not in mapped]
        if missing:
            raise RuntimeError(
                f"Encoder outputs missing required fields {missing}. "
                f"Observed shapes: {observed_shapes}. "
                f"Expected shapes: {list(SHAPE_TO_FIELD)}."
            )
        return mapped
