"""Shared image utility functions."""

import io
import base64
from PIL import Image, ImageDraw, ImageFont


def image_to_base64(img_path: str, max_size: int = 512) -> str:
    """Encode an image file to base64 JPEG string (for GPT-4o vision API)."""
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def add_label(img: Image.Image, text: str) -> Image.Image:
    """Overlay a black banner with white label text at the top of an image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([(0, 0), (img.width, 18)], fill="black")
    draw.text((4, 1), text, fill="white", font=font)
    return img
