"""
PreciseMaskGenerator — SAM-based defect mask generation.

Uses Segment Anything Model (SAM) to:
  1. Segment the main object from background
  2. Place a small, precise defect mask constrained within the object boundary
  3. Ensure mask covers 3–15% of image (not the entire object)
"""

import math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from config.settings import SAM_CKPT, DEVICE


def load_sam(checkpoint: str = None):
    """Load SAM ViT-H and return (predictor, auto_generator) tuple."""
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

    ckpt = checkpoint or SAM_CKPT
    sam  = sam_model_registry["vit_h"](checkpoint=ckpt).to(DEVICE)

    predictor = SamPredictor(sam)
    auto_gen  = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=1000,
    )
    return predictor, auto_gen


class PreciseMaskGenerator:
    """
    Generates physically grounded defect masks using SAM object segmentation.

    Example usage::

        predictor, auto_gen = load_sam()
        mask_gen = PreciseMaskGenerator(predictor, auto_gen)

        image = Image.open("normal.png").convert("RGB")
        obj_mask = mask_gen.get_object_mask(image)
        defect_mask, ratio = mask_gen.generate_defect_mask(
            image, bbox_normalized=[0.35, 0.40, 0.55, 0.60],
            object_mask=obj_mask, mask_shape="ellipse",
        )
    """

    def __init__(self, sam_predictor, sam_auto, target_defect_ratio=(0.03, 0.15)):
        self.predictor  = sam_predictor
        self.auto_gen   = sam_auto
        self.min_ratio, self.max_ratio = target_defect_ratio

    def get_object_mask(self, image: Image.Image) -> np.ndarray:
        """Segment the main object from background using SAM auto segmentation."""
        img_np = np.array(image.convert("RGB"))
        masks  = self.auto_gen.generate(img_np)

        if not masks:
            h, w  = img_np.shape[:2]
            result = np.zeros((h, w), dtype=bool)
            border = int(min(h, w) * 0.05)
            result[border : h - border, border : w - border] = True
            return result

        masks.sort(key=lambda x: x["area"], reverse=True)
        for m in masks:
            ratio = m["area"] / (img_np.shape[0] * img_np.shape[1])
            if 0.05 < ratio < 0.90:
                return m["segmentation"]

        return masks[0]["segmentation"]

    def generate_defect_mask(
        self,
        image: Image.Image,
        bbox_normalized: list[float],
        object_mask: np.ndarray = None,
        mask_shape: str = "ellipse",
    ) -> tuple[Image.Image, float]:
        """
        Generate a precise defect mask constrained within the object boundary.

        Args:
            image:            PIL Image (RGB)
            bbox_normalized:  [x1, y1, x2, y2] in [0, 1] from LLM hypothesis
            object_mask:      Pre-computed SAM boolean mask (optional)
            mask_shape:       "ellipse" | "rectangle" | "irregular"

        Returns:
            (defect_mask_image, actual_defect_ratio)
            defect_mask_image: PIL Image in "L" mode (0/255)
        """
        w, h = image.size

        if object_mask is None:
            object_mask = self.get_object_mask(image)

        x1 = max(0, int(bbox_normalized[0] * w))
        y1 = max(0, int(bbox_normalized[1] * h))
        x2 = min(w, int(bbox_normalized[2] * w))
        y2 = min(h, int(bbox_normalized[3] * h))

        # Ensure minimum 20px in each dimension
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
            ellipse = (X - cx_c) ** 2 / max(rx ** 2, 1) + (Y - cy_c) ** 2 / max(ry ** 2, 1) <= 1
            defect_mask[ellipse] = 255

        elif mask_shape == "irregular":
            cy_c = (y1 + y2) // 2
            cx_c = (x1 + x2) // 2
            ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
            Y, X    = np.ogrid[:h, :w]
            angles  = np.arctan2(Y - cy_c, X - cx_c)
            radii   = np.sqrt((X - cx_c) ** 2 + (Y - cy_c) ** 2)
            np.random.seed(42)
            noise     = np.random.uniform(0.7, 1.3, size=angles.shape)
            threshold = np.sqrt((rx * np.cos(angles)) ** 2 + (ry * np.sin(angles)) ** 2)
            defect_mask[radii < threshold * noise] = 255

        else:  # rectangle
            defect_mask[y1:y2, x1:x2] = 255

        # Constrain to object boundary
        obj_uint8 = (object_mask.astype(np.uint8)) * 255
        defect_mask = np.minimum(defect_mask, obj_uint8)

        ratio = (defect_mask > 0).sum() / (h * w)

        # Shrink if too large
        if ratio > self.max_ratio:
            scale = math.sqrt(self.max_ratio / max(ratio, 1e-6)) * 0.9
            cx_c   = (x1 + x2) / 2
            cy_c   = (y1 + y2) / 2
            hw     = (x2 - x1) * scale / 2
            hh     = (y2 - y1) * scale / 2
            defect_mask = np.zeros((h, w), dtype=np.uint8)
            defect_mask[
                max(0, int(cy_c - hh)) : min(h, int(cy_c + hh)),
                max(0, int(cx_c - hw)) : min(w, int(cx_c + hw)),
            ] = 255
            defect_mask = np.minimum(defect_mask, obj_uint8)

        # Expand if too small or empty
        if (defect_mask > 0).sum() / (h * w) < self.min_ratio:
            obj_ys, obj_xs = np.where(object_mask)
            if len(obj_ys) > 0:
                cy_c = int(np.median(obj_ys))
                cx_c = int(np.median(obj_xs))
                radius = int(math.sqrt(self.min_ratio * h * w / math.pi))
                Y, X = np.ogrid[:h, :w]
                circle = (X - cx_c) ** 2 + (Y - cy_c) ** 2 <= radius ** 2
                defect_mask[circle] = 255
                defect_mask = np.minimum(defect_mask, obj_uint8)

        # Smooth edges
        smooth   = gaussian_filter(defect_mask.astype(float), sigma=3)
        final    = (smooth > 127).astype(np.uint8) * 255
        actual_ratio = (final > 0).sum() / (h * w)

        return Image.fromarray(final).convert("L"), float(actual_ratio)
