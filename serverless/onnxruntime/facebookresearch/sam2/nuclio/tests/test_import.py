"""手順9: package import smoke test."""


def test_import_sam2_ort_core() -> None:
    import sam2_ort_core

    assert sam2_ort_core is not None
