"""
Stage 1 pipeline — Training Foundation.

Steps:
  1. Build training triplets from MVTec AD using DINOv2 nearest-normal matching
  2. Fine-tune SD Inpainting LoRA adapters per category
  3. Validate LoRA with ControlNet edge-conditioned test generation
  4. Extract and FAISS-index DINOv2 features for all train/test images
"""

import os
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from config.settings import (
    ALL_CATEGORIES, DEVICE, IMG_SIZE,
    TRIPLET_DIR, LORA_DIR, FEATURES_DIR, EVAL_S1_DIR, CKPT_DIR,
    LORA_CONFIG,
)
from data.paths import get_image_paths, get_defect_types
from features.extraction import (
    load_dinov2, extract_cls_feature, build_faiss_index, dinov2_transform,
)
from models.lora import train_lora_for_category
from models.controlnet import extract_canny_edges, load_controlnet


# ── Step 1: Build training triplets ──────────────────────────────────────────

def build_triplets(categories: list[str] = None) -> dict[str, int]:
    """
    Build (normal, mask, defect) triplets for all categories.
    Uses DINOv2 CLS features to find the nearest normal image for each defect.

    Returns dict: category → n_triplets
    """
    cats   = categories or ALL_CATEGORIES
    stats  = {}

    print("Loading DINOv2 for triplet building ...")
    dinov2 = load_dinov2()
    print("  DINOv2 loaded")

    for cat in cats:
        cat_triplet_dir = os.path.join(TRIPLET_DIR, cat)
        os.makedirs(cat_triplet_dir, exist_ok=True)
        manifest_path = os.path.join(cat_triplet_dir, "manifest.json")

        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                existing = json.load(f)
            n = len(existing["triplets"])
            print(f"  {cat}: {n} triplets (cached)")
            stats[cat] = n
            continue

        print(f"\n── {cat} ──")
        normal_paths = get_image_paths(cat, "train", "good")
        if not normal_paths:
            print(f"  No normal training images, skipping")
            continue

        # Load or extract normal CLS features
        cached_cls = os.path.join(FEATURES_DIR, cat, "cls_features.pt")
        if os.path.exists(cached_cls):
            normal_feats = torch.load(cached_cls, map_location="cpu")
            if normal_feats.shape[0] != len(normal_paths):
                normal_feats = torch.stack([
                    extract_cls_feature(p, dinov2)
                    for p in tqdm(normal_paths, desc="  Normal feats")
                ])
        else:
            normal_feats = torch.stack([
                extract_cls_feature(p, dinov2)
                for p in tqdm(normal_paths, desc="  Normal feats")
            ])

        defect_types = get_defect_types(cat)
        triplets     = []

        for dt in defect_types:
            defect_paths = get_image_paths(cat, "test", dt)
            mask_dir     = os.path.join(MVTEC_ROOT, cat, "ground_truth", dt)

            for dp in defect_paths:
                fname     = os.path.splitext(os.path.basename(dp))[0]
                mask_path = os.path.join(mask_dir, f"{fname}_mask.png")
                if not os.path.exists(mask_path):
                    continue

                mask = np.array(Image.open(mask_path).convert("L"))
                defect_ratio = (mask > 127).sum() / mask.size
                if defect_ratio < 0.001 or defect_ratio > 0.7:
                    continue

                defect_feat = extract_cls_feature(dp, dinov2)
                sims = torch.nn.functional.cosine_similarity(
                    defect_feat.unsqueeze(0), normal_feats, dim=1,
                )
                best_idx = sims.argmax().item()
                nearest_normal = normal_paths[best_idx]

                triplets.append({
                    "defect_image":  dp,
                    "mask_image":    mask_path,
                    "normal_image":  nearest_normal,
                    "category":      cat,
                    "defect_type":   dt,
                    "similarity":    round(sims[best_idx].item(), 4),
                    "defect_ratio":  round(defect_ratio, 4),
                })

        manifest = {
            "category":      cat,
            "n_triplets":    len(triplets),
            "n_defect_types": len(defect_types),
            "defect_types":  defect_types,
            "created":       datetime.now().isoformat(),
            "triplets":      triplets,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        stats[cat] = len(triplets)
        print(f"  {cat}: {len(triplets)} triplets across {len(defect_types)} defect types")

    del dinov2
    gc.collect()
    torch.cuda.empty_cache()
    return stats


# ── Step 2: Train LoRA adapters ───────────────────────────────────────────────

def train_all_lora(categories: list[str] = None, config: dict = None) -> None:
    """Train a LoRA adapter for each category (skips already-trained ones)."""
    cats = categories or ALL_CATEGORIES
    cfg  = config or LORA_CONFIG

    for cat in cats:
        print(f"\nTraining LoRA: {cat}")
        train_lora_for_category(cat, cfg)


# ── Step 3: ControlNet validation ─────────────────────────────────────────────

def validate_with_controlnet(
    test_categories: list[str] = None,
) -> None:
    """
    Run a quick ControlNet + LoRA generation test on 3 categories.
    Saves side-by-side grids: Normal | Canny | Real | Generated | Mask
    """
    from models.controlnet import generate_with_controlnet

    cats      = test_categories or ["bottle", "metal_nut", "transistor"]
    cnet_dir  = os.path.join(EVAL_S1_DIR, "controlnet_test")
    os.makedirs(cnet_dir, exist_ok=True)

    controlnet = load_controlnet()

    for cat in cats:
        manifest_path = os.path.join(TRIPLET_DIR, cat, "manifest.json")
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        if not manifest["triplets"]:
            continue

        t = manifest["triplets"][0]
        gen_img, canny_img = generate_with_controlnet(
            cat, t["normal_image"], t["mask_image"], t["defect_type"],
            controlnet_model=controlnet,
        )

        normal = Image.open(t["normal_image"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        real   = Image.open(t["defect_image"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        mask   = Image.open(t["mask_image"]).convert("L").resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

        grid = Image.new("RGB", (IMG_SIZE * 5, IMG_SIZE))
        for i, img in enumerate([normal, canny_img, real, gen_img, mask]):
            grid.paste(img, (IMG_SIZE * i, 0))
        grid.save(os.path.join(cnet_dir, f"{cat}_test.png"))
        print(f"  {cat}: saved ControlNet test grid")

    del controlnet
    gc.collect()
    torch.cuda.empty_cache()


# ── Step 4: Feature extraction ────────────────────────────────────────────────

def extract_all_features(categories: list[str] = None) -> None:
    """Extract DINOv2 features and build FAISS indices for all categories."""
    cats   = categories or ALL_CATEGORIES
    print("Loading DINOv2 for feature extraction ...")
    dinov2 = load_dinov2()

    for cat in cats:
        print(f"  {cat} ...", end=" ", flush=True)
        build_faiss_index(cat, dinov2)
        print("done")

    del dinov2
    gc.collect()
    torch.cuda.empty_cache()


# ── Main entry point ──────────────────────────────────────────────────────────

def run_stage1(
    categories: list[str] = None,
    train_lora: bool = True,
    validate_cnet: bool = True,
    extract_features: bool = True,
) -> None:
    """Run the complete Stage 1 pipeline."""
    from config.settings import MVTEC as _MVTEC
    global MVTEC_ROOT
    MVTEC_ROOT = _MVTEC

    print("=" * 70)
    print("STAGE 1: Training Foundation")
    print("=" * 70)

    # 1. Triplets
    print("\n[1] Building training triplets ...")
    triplet_stats = build_triplets(categories)
    total = sum(triplet_stats.values())
    print(f"  Total triplets: {total}")

    # 2. LoRA training
    if train_lora:
        print("\n[2] Training LoRA adapters ...")
        train_all_lora(categories)

    # 3. ControlNet validation
    if validate_cnet:
        print("\n[3] ControlNet validation ...")
        validate_with_controlnet()

    # 4. Feature extraction
    if extract_features:
        print("\n[4] Extracting DINOv2 features ...")
        extract_all_features(categories)

    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETE")
    print(f"  Triplets:    {TRIPLET_DIR}/")
    print(f"  LoRA models: {LORA_DIR}/")
    print(f"  Features:    {FEATURES_DIR}/")
    print("=" * 70)
