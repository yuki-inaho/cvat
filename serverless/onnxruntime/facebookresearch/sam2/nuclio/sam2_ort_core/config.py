"""Sam2OrtConfig: explicit environment-driven configuration (no implicit fallback)."""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"


def _parse_bool(value: str, *, var_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise RuntimeError(
        f"{var_name} has an invalid boolean value {value!r}. Use one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}."
    )


@dataclass(frozen=True)
class Sam2OrtConfig:
    """Configuration for the SAM2 ONNX Runtime encoder.

    All values are resolved explicitly from environment variables via
    :meth:`from_env`. There is no implicit default model path and no silent
    CPU fallback when GPU is required.
    """

    model_path: str
    provider: str
    require_gpu: bool
    input_size: int = 1024

    @classmethod
    def from_env(cls) -> Sam2OrtConfig:
        """Build a config from environment variables.

        Raises:
            RuntimeError: if ``SAM2_ORT_MODEL_PATH`` is unset, if a boolean
                value cannot be parsed, or if GPU is required while the
                requested provider is the CPU provider (implicit fallback).
        """
        model_path = os.environ.get("SAM2_ORT_MODEL_PATH")
        if not model_path:
            raise RuntimeError(
                "SAM2_ORT_MODEL_PATH is not set. The model path must be provided "
                "explicitly; there is no implicit default."
            )

        provider = os.environ.get("SAM2_ORT_PROVIDER", _CUDA_PROVIDER)
        require_gpu = _parse_bool(
            os.environ.get("SAM2_ORT_REQUIRE_GPU", "true"),
            var_name="SAM2_ORT_REQUIRE_GPU",
        )

        if require_gpu and provider == _CPU_PROVIDER:
            raise RuntimeError(
                "SAM2_ORT_REQUIRE_GPU is true but SAM2_ORT_PROVIDER is "
                f"{_CPU_PROVIDER!r}. This is an implicit CPU fallback and is "
                "forbidden. Set SAM2_ORT_PROVIDER to a GPU provider "
                "(e.g. CUDAExecutionProvider) or disable SAM2_ORT_REQUIRE_GPU."
            )

        return cls(model_path=model_path, provider=provider, require_gpu=require_gpu)
