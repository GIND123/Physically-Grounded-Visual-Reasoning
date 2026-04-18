"""
Simple (non-SAM) defect mask generation from hypothesis bounding boxes.

Used in Stage 2 synthesis where SAM is not loaded.
For SAM-based masks, see models/sam_mask.py.
"""

import math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def detect_object_mask_simple(image: Image.Image, threshold: int = 30) -> np.ndarray:
    """
    Simple object detection via background thresholding.

    Works well for MVTec AD where objects are typically on dark backgrounds.
    For texture categories (carpet, grid, leather, tile, wood) returns full image mask.

    Returns boolean ndarray of shape (H, W).
    """
    img_np = np.array(image.convert("RGB"))
    gray   = np.mean(img_np, axis=2)

    # Texture categories: mostly non-dark → return full mask
    dark_ratio = (gray < threshold).sum() / gray.size
    if dark_ratio < 0.15:
        return np.ones(gray.shape, dtype=bool)

    # Object-on-background: threshold to isolate object
    from scipy.ndimage import binary_fill_holes, binary_opening
    object_mask = gray > threshold
    object_mask = binary_fill_holes(object_mask)
    object_mask = binary_opening(object_mask, iterations=3)

    return object_mask


def generate_defect_mask(
    image: Image.Image,
    bbox_normalized: list[float],
    mask_shape: str = "ellipse",
    target_ratio_range: tuple[float, float] = (0.03, 0.15),
) -> tuple[Image.Image, float]:
    """
    Generate a precise defect mask from a hypothesis bounding box.

    Args:
        image:              PIL Image (RGB)
        bbox_normalized:    [x1, y1, x2, y2] in [0, 1] from hypothesis
        mask_shape:         "ellipse" | "rectangle" | "irregular"
        target_ratio_range: (min_ratio, max_ratio) of image pixels

    Returns:
        (defect_mask_image, actual_ratio)
        defect_mask_image: PIL Image in "L" mode (0/255)
    """
    w, h = image.size
    min_ratio, max_ratio = target_ratio_range

    obj_mask = detect_object_mask_simple(image)

    x1 = max(0, int(bbox_normalized[0] * w))
    y1 = max(0, int(bbox_normalized[1] * h))
    x2 = min(w, int(bbox_normalized[2] * w))
    y2 = min(h, int(bbox_normalized[3] * h))

    if x2 - x1 < 20:
        cx = (x1 + x2) // 2
        x1, x2 = max(0, cx - 20), min(w, cx + 20)
    if y2 - y1 < 20:
        cy = (y1 + y2) // 2
        y1, y2 = max(0, cy - 20), min(h, cy + 20)

    defect_mask = np.zeros((h, w), dtype=np.uint8)

    if mask_shape == "ellipse":
        cy_c = (y1 + y2) // 2
        cx_c = (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
        Y, X = np.ogrid[:h, :w]
        defect_mask[
            (X - cx_c) ** 2 / max(rx ** 2, 1) + (Y - cy_c) ** 2 / max(ry ** 2, 1) <= 1
        ] = 255

    elif mask_shape == "irregular":
        cy_c = (y1 + y2) // 2
        cx_c = (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
        Y, X    = np.ogrid[:h, :w]
        radii   = np.sqrt((X - cx_c) ** 2 + (Y - cy_c) ** 2)
        angles  = np.arctan2(Y - cy_c, X - cx_c)
        np.random.seed(hash(str(bbox_normalized)) % 2 ** 31)
        noise     = np.random.uniform(0.6, 1.4, size=angles.shape)
        threshold_r = np.sqrt((rx * np.cos(angles)) ** 2 + (ry * np.sin(angles)) ** 2)
        defect_mask[radii < threshold_r * noise] = 255

    else:  # rectangle
        defect_mask[y1:y2, x1:x2] = 255

    # Constrain to object
    defect_mask = np.minimum(defect_mask, (obj_mask * 255).astype(np.uint8))

    ratio = (defect_mask > 0).sum() / (h * w)

    # Shrink if too large
    if ratio > max_ratio:
        scale = math.sqrt(max_ratio / max(ratio, 1e-6)) * 0.85
        cx_c  = (x1 + x2) / 2
        cy_c  = (y1 + y2) / 2
        hw    = (x2 - x1) * scale / 2
        hh    = (y2 - y1) * scale / 2
        defect_mask = np.zeros((h, w), dtype=np.uint8)
        defect_mask[
            int(max(0, cy_c - hh)) : int(min(h, cy_c + hh)),
            int(max(0, cx_c - hw)) : int(min(w, cx_c + hw)),
        ] = 255
        defect_mask = np.minimum(defect_mask, (obj_mask * 255).astype(np.uint8))

    # Expand if too small / empty
    if (defect_mask > 0).sum() / (h * w) < 0.005:
        obj_ys, obj_xs = np.where(obj_mask)
        if len(obj_ys) > 0:
            cy_c   = int(np.median(obj_ys))
            cx_c   = int(np.median(obj_xs))
            radius = int(math.sqrt(min_ratio * h * w / math.pi))
            Y, X   = np.ogrid[:h, :w]
            defect_mask[(X - cx_c) ** 2 + (Y - cy_c) ** 2 <= radius ** 2] = 255

    # Smooth edges
    smooth = gaussian_filter(defect_mask.astype(float), sigma=2)
    final  = (smooth > 127).astype(np.uint8) * 255
    return Image.fromarray(final).convert("L"), float((final > 0).sum() / (h * w))
