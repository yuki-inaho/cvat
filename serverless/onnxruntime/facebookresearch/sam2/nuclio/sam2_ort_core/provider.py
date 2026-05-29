"""Provider detection and fail-fast validation for SAM2 ONNX Runtime.

GPU 成功を CPU 成功で代替しない。GPU が要求された場合に
``CUDAExecutionProvider`` が利用不可なら例外で停止する。
"""

from __future__ import annotations

import logging

import onnxruntime as ort

from sam2_ort_core.config import Sam2OrtConfig

logger = logging.getLogger("sam2-ort")

_CUDA_PROVIDER = "CUDAExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


def validate_provider(config: Sam2OrtConfig) -> list[str]:
    """Validate available execution providers against the config.

    Returns:
        Ordered provider list to pass to ``InferenceSession`` with the
        requested provider first (CPU appended as a final entry only when the
        requested provider already satisfies the policy, never as a silent
        GPU substitute).

    Raises:
        RuntimeError: if GPU is required but ``CUDAExecutionProvider`` is not
            available, or if the requested provider is not available at all.
    """
    available = list(ort.get_available_providers())
    logger.info("[SAM2-ORT] ONNX Runtime version: %s", ort.__version__)
    logger.info("[SAM2-ORT] Requested provider: %s", config.provider)
    logger.info("[SAM2-ORT] Available providers: %s", available)

    if config.require_gpu and _CUDA_PROVIDER not in available:
        raise RuntimeError(
            f"CUDAExecutionProvider not available. Available: {available}. "
            "GPU mode requires CUDA. Do not fall back to CPU."
        )

    if config.provider not in available:
        raise RuntimeError(f"Requested provider {config.provider!r} is not available. Available: {available}.")

    resolved = [config.provider]
    # CPU は最後の保険としてのみ付与する (GPU 要求時の暗黙代替ではない)。
    if config.provider != _CPU_PROVIDER and _CPU_PROVIDER in available:
        resolved.append(_CPU_PROVIDER)

    logger.info("[SAM2-ORT] Resolved provider order: %s", resolved)
    return resolved
