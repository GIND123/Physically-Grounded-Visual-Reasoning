"""
Defect synthesis — LoRA + ControlNet generation driven by LLM hypotheses.

For each (category, defect_type):
  1. Load LoRA-adapted inpainting pipeline
  2. Read hypothesis JSON (from llm.py)
  3. Generate defect mask (from mask_gen.py)
  4. Run diffusion with hypothesis prompt + Canny conditioning
  5. Save (generated, mask, normal_reference) triplets
"""

import os
import gc
import random
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from config.settings import (
    DEVICE, IMG_SIZE, LORA_DIR, SYN_DIR, HYPO_DIR,
    BASE_MODEL_ID, CONTROLNET_MODEL, N_SYNTHETIC_PER_DEFECT,
)
from data.paths import get_image_paths
from synthesis.mask_gen import generate_defect_mask


def extract_canny(image: Image.Image, low: int = 80, high: int = 200) -> Image.Image:
    """Extract Canny edge map from a PIL image. Returns RGB PIL Image."""
    img_np = np.array(image.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges  = cv2.Canny(gray, low, high)
    return Image.fromarray(edges).convert("RGB")


def synthesize_defects_for_category(
    category: str,
    controlnet_model=None,
    n_per_defect: int = None,
) -> dict[str, int]:
    """
    Generate synthetic defect images for all defect types of a category.

    Args:
        category:         MVTec category name
        controlnet_model: Pre-loaded ControlNetModel (or None to skip ControlNet)
        n_per_defect:     Images to generate per defect type (default: N_SYNTHETIC_PER_DEFECT)

    Returns:
        Dict mapping defect_type → n_generated
    """
    from diffusers import (
        StableDiffusionInpaintPipeline,
        StableDiffusionControlNetInpaintPipeline,
        UniPCMultistepScheduler,
    )
    from peft import PeftModel

    n = n_per_defect or N_SYNTHETIC_PER_DEFECT
    cat_lora_dir = os.path.join(LORA_DIR, category)
    has_lora     = os.path.exists(os.path.join(cat_lora_dir, ".done"))

    # Verify LoRA was trained on the same base model
    meta_path = os.path.join(cat_lora_dir, "training_meta.json")
    if has_lora and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("base_model") != BASE_MODEL_ID:
            raise ValueError(
                f"LoRA mismatch for {category}: "
                f"trained on {meta.get('base_model')} but pipeline uses {BASE_MODEL_ID}"
            )

    cat_hypo_dir = os.path.join(HYPO_DIR, category)
    if not os.path.exists(cat_hypo_dir):
        print(f"  {category}: no hypotheses, skipping")
        return {}

    normal_paths = get_image_paths(category, "train", "good")
    if not normal_paths:
        return {}

    # ── Build pipeline ────────────────────────────────────────────────────────
    use_controlnet = controlnet_model is not None
    try:
        if use_controlnet:
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                BASE_MODEL_ID, controlnet=controlnet_model,
                torch_dtype=torch.float16, safety_checker=None,
            ).to(DEVICE)
        else:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                BASE_MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
            ).to(DEVICE)

        if has_lora:
            pipe.unet = PeftModel.from_pretrained(
                pipe.unet, cat_lora_dir, is_trainable=False,
            )
            pipe.unet.eval().to(device=DEVICE, dtype=torch.float16)

        if use_controlnet:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()

    except Exception as e:
        print(f"  {category}: ControlNet pipeline failed ({e}), falling back to inpainting-only")
        use_controlnet = False
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
        ).to(DEVICE)
        if has_lora:
            pipe.unet = PeftModel.from_pretrained(
                pipe.unet, cat_lora_dir, is_trainable=False,
            )
            pipe.unet.eval().to(device=DEVICE, dtype=torch.float16)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()

    cat_results = {}

    for hypo_file in sorted(Path(cat_hypo_dir).glob("*.json")):
        dt          = hypo_file.stem
        dt_syn_dir  = os.path.join(SYN_DIR, category, dt)
        os.makedirs(dt_syn_dir, exist_ok=True)

        existing = [
            p for p in sorted(Path(dt_syn_dir).glob("syn_*[0-9].png"))
            if "mask" not in p.name and "normal" not in p.name
        ]
        if len(existing) >= n:
            print(f"    {dt}: {len(existing)} images (cached)")
            cat_results[dt] = len(existing)
            continue

        with open(hypo_file) as f:
            hypo = json.load(f)

        bbox       = hypo.get("defect_bbox_normalized", [0.3, 0.3, 0.6, 0.6])
        gen_prompt = hypo.get("generation_prompt", f"a {dt} defect on a {category}")
        neg_prompt = hypo.get(
            "negative_prompt",
            "cartoon, illustration, painting, low quality, blurry, watermark, text",
        )

        full_prompt = (
            f"{gen_prompt}, industrial inspection photograph, macro, high resolution, photorealistic"
        )

        generated = 0

        for i in range(n):
            try:
                normal_path = normal_paths[i % len(normal_paths)]
                normal_img  = Image.open(normal_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

                # Jitter bbox for diversity
                jitter = 0.02
                jittered_bbox = [
                    max(0.0, min(1.0, bbox[j] + random.uniform(-jitter, jitter)))
                    for j in range(4)
                ]
                shape = ["ellipse", "irregular", "rectangle"][i % 3]
                defect_mask, mask_ratio = generate_defect_mask(
                    normal_img, jittered_bbox, mask_shape=shape,
                )

                canny = extract_canny(normal_img) if use_controlnet else None
                generator = torch.Generator(device=DEVICE).manual_seed(42 + i * 7)

                with torch.autocast("cuda"):
                    if use_controlnet:
                        result = pipe(
                            prompt=full_prompt, negative_prompt=neg_prompt,
                            image=normal_img, mask_image=defect_mask,
                            control_image=canny,
                            guidance_scale=7.5, controlnet_conditioning_scale=0.7,
                            num_inference_steps=25,
                            height=IMG_SIZE, width=IMG_SIZE,
                            generator=generator,
                        ).images[0]
                    else:
                        result = pipe(
                            prompt=full_prompt, negative_prompt=neg_prompt,
                            image=normal_img, mask_image=defect_mask,
                            guidance_scale=7.5, num_inference_steps=25,
                            height=IMG_SIZE, width=IMG_SIZE,
                            generator=generator,
                        ).images[0]

                result.save(os.path.join(dt_syn_dir, f"syn_{i:03d}.png"))
                defect_mask.save(os.path.join(dt_syn_dir, f"syn_{i:03d}_mask.png"))
                normal_img.save(os.path.join(dt_syn_dir, f"syn_{i:03d}_normal.png"))
                generated += 1

            except Exception as e:
                print(f"    {dt}/{i}: {e}")

        cat_results[dt] = generated
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        print(
            f"    {dt}: {generated}/{n} generated "
            f"(bbox_area={bbox_area:.2%}, lora={'Y' if has_lora else 'N'})"
        )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return cat_results
