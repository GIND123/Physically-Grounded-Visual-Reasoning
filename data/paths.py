"""Utility functions for resolving MVTec AD image paths."""

import os
from config.settings import MVTEC

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def get_image_paths(category: str, split: str, subset: str) -> list[str]:
    """Return sorted list of image paths for a given MVTec split/subset."""
    img_dir = os.path.join(MVTEC, category, split, subset)
    if not os.path.exists(img_dir):
        return []
    return sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
    )


def get_defect_types(category: str) -> list[str]:
    """Return list of defect type names for a category (from test directory)."""
    test_dir = os.path.join(MVTEC, category, "test")
    if not os.path.exists(test_dir):
        return []
    return sorted(
        d for d in os.listdir(test_dir)
        if d != "good" and os.path.isdir(os.path.join(test_dir, d))
    )
