# Copyright (C) 2024-2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT
"""Nuclio handler (thin adapter) for the SAM2 ONNX Runtime GPU encoder.

The handler only parses the Nuclio event, delegates inference to
``Sam2OrtEncoder`` (core logic), serializes the result, and builds the
response. It holds no inference logic itself (DRY/SOLID: core/adapter split).
"""

import base64
import io
import json

from PIL import Image

from sam2_ort_core.config import Sam2OrtConfig
from sam2_ort_core.encoder import Sam2OrtEncoder
from sam2_ort_core.serialize import serialize_embedding


def init_context(context):
    context.logger.info("Init SAM2 ORT encoder... 0%")
    config = Sam2OrtConfig.from_env()
    # Provider self-test fires here: Sam2OrtEncoder.__init__ validates the
    # CUDAExecutionProvider (fail-fast in GPU mode) and logs provider/model IO.
    encoder = Sam2OrtEncoder(config)
    context.user_data.encoder = encoder
    context.logger.info(f"Init SAM2 ORT encoder...100% provider={encoder.provider}")


def handler(context, event):
    try:
        data = event.body
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = Image.open(buf).convert("RGB")
        embedding = context.user_data.encoder.run(image)
        return context.Response(
            body=json.dumps(serialize_embedding(embedding)),
            headers={},
            content_type="application/json",
            status_code=200,
        )
    except Exception as e:
        context.logger.error(f"SAM2 ORT error: {e}")
        return context.Response(
            body=json.dumps({"error": "Internal server error"}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
