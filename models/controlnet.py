"""
ControlNet-Canny edge-conditioned inpainting.

Preserves object structure while LoRA generates realistic defects.
Uses pretrained thibaud/controlnet-sd21-canny-diffusers (no fine-tuning needed).
"""

import gc
import numpy as np
import cv2
import torch
from PIL import Image

from config.settings import (
    DEVICE, IMG_SIZE, BASE_MODEL_ID, CONTROLNET_MODEL, LORA_DIR,
)


def extract_canny_edges(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """Extract Canny edges from a PIL image. Returns RGB PIL Image."""
    img_np = np.array(image.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges  = cv2.Canny(gray, low, high)
    return Image.fromarray(edges).convert("RGB")


def generate_with_controlnet(
    category: str,
    normal_path: str,
    mask_path: str,
    defect_type: str,
    controlnet_model,
    severity: str = "moderate",
    seed: int = 42,
) -> tuple[Image.Image, Image.Image]:
    """
    Generate a defect image using LoRA + ControlNet-Canny inpainting.

    Args:
        category:          MVTec category name
        normal_path:       Path to the normal (defect-free) reference image
        mask_path:         Path to the binary defect mask
        defect_type:       Defect type name (e.g. "broken_large")
        controlnet_model:  Pre-loaded ControlNetModel instance
        severity:          "minor" | "moderate" | "severe"
        seed:              Random seed for reproducibility

    Returns:
        (generated_image, canny_edge_image) as PIL Images
    """
    from diffusers import (
        StableDiffusionControlNetInpaintPipeline,
        UniPCMultistepScheduler,
    )
    from peft import PeftModel

    cat_lora_dir = f"{LORA_DIR}/{category}"

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)

    if os.path.exists(f"{cat_lora_dir}/.done"):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, cat_lora_dir)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()

    normal = Image.open(normal_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    mask   = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    mask_np = (np.array(mask) > 127).astype(np.uint8) * 255
    mask    = Image.fromarray(mask_np)

    canny = extract_canny_edges(normal)

    prompt = (
        f"a {severity} {defect_type.replace('_', ' ')} defect on a "
        f"{category.replace('_', ' ')}, industrial inspection, macro photograph"
    )
    negative_prompt = (
        "cartoon, illustration, painting, low quality, blurry, "
        "watermark, text, different object, wrong shape"
    )

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.autocast("cuda"):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=normal,
            mask_image=mask,
            control_image=canny,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=50,
            height=IMG_SIZE, width=IMG_SIZE,
            generator=generator,
        ).images[0]

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return result, canny


import os  # needed by generate_with_controlnet


def load_controlnet():
    """Load the pretrained ControlNet-Canny model."""
    from diffusers import ControlNetModel
    return ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, torch_dtype=torch.float16,
    )
