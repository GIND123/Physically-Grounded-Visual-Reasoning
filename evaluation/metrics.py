"""
Stage 3 evaluation — pipeline metrics, generation quality (LPIPS), paper figures.
"""

import os
import gc
import json
import random
import math
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm.auto import tqdm

from config.settings import (
    DEVICE, IMG_SIZE, MVTEC, S2_DIR, S3_DIR, FIGURES_DIR, TABLES_DIR,
    HYPO_DIR, SYN_DIR, VERIFIED_DIR, CF_DIR, ALL_CATEGORIES,
)
from data.paths import get_image_paths


# ── Pipeline metrics ──────────────────────────────────────────────────────────

def compute_pipeline_metrics(
    verification_results: dict,
    counterfactual_results: dict,
) -> dict:
    """
    Aggregate per-category pipeline metrics from verification and counterfactual results.

    Returns a dict keyed by category with fields:
        n_defect_types, n_generated, n_accepted, accept_rate,
        cf_total, cf_suppressed, cf_rate, mean_cf_delta, stage_pass_rates
    """
    pipeline_metrics = {}

    for cat in ALL_CATEGORIES:
        vr = verification_results.get(cat, {})
        cr = counterfactual_results.get(cat, {})

        n_total    = sum(d.get("n_total",    0) for d in vr.values())
        n_accepted = sum(d.get("n_accepted", 0) for d in vr.values())

        stage_passes = defaultdict(list)
        for dt_data in vr.values():
            for score in dt_data.get("scores", []):
                for sk, sv in score.get("details", {}).items():
                    if isinstance(sv, dict) and "passed" in sv:
                        stage_passes[sk].append(1 if sv["passed"] else 0)

        cf_total      = len(cr)
        cf_suppressed = sum(1 for r in cr.values() if r.get("defect_suppressed"))
        mean_delta    = (
            float(np.mean([r["suppression_delta"] for r in cr.values()])) if cr else 0.0
        )

        pipeline_metrics[cat] = {
            "n_defect_types": len(vr),
            "n_generated":    n_total,
            "n_accepted":     n_accepted,
            "accept_rate":    round(n_accepted / max(n_total, 1), 3),
            "cf_total":       cf_total,
            "cf_suppressed":  cf_suppressed,
            "cf_rate":        round(cf_suppressed / max(cf_total, 1), 3),
            "mean_cf_delta":  round(mean_delta, 4),
            "stage_pass_rates": {k: round(float(np.mean(v)), 3) for k, v in stage_passes.items()},
        }

    return pipeline_metrics


# ── Generation quality (LPIPS) ────────────────────────────────────────────────

def compute_generation_quality(n_syn_per_cat: int = 15, n_real_samples: int = 10) -> dict:
    """
    Compute LPIPS between verified synthetic images and real defect images per category.

    Lower LPIPS = more perceptually similar to real defects.
    """
    import lpips as lpips_lib

    lpips_fn  = lpips_lib.LPIPS(net="alex").to(DEVICE)
    to_tensor = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    gen_quality = {}

    for cat in ALL_CATEGORIES:
        cat_ver = Path(VERIFIED_DIR) / cat
        if not cat_ver.exists():
            continue

        syn_paths = [
            str(p)
            for dt_dir in sorted(cat_ver.iterdir()) if dt_dir.is_dir()
            for p in sorted(dt_dir.glob("syn_*[0-9].png"))
            if "mask" not in p.name and "normal" not in p.name
        ]
        if not syn_paths:
            continue

        test_dir  = os.path.join(MVTEC, cat, "test")
        real_paths = [
            str(p)
            for dd in sorted(os.listdir(test_dir))
            if dd != "good" and os.path.isdir(os.path.join(test_dir, dd))
            for p in map(str, Path(os.path.join(test_dir, dd)).glob("*"))
            if os.path.splitext(p)[1].lower() in {".png", ".jpg", ".jpeg"}
        ]
        if not real_paths:
            continue

        cat_lpips = []
        for sp in syn_paths[:n_syn_per_cat]:
            syn_t = (to_tensor(Image.open(sp).convert("RGB")) * 2 - 1).unsqueeze(0).to(DEVICE)
            best  = float("inf")
            for rp in random.sample(real_paths, min(n_real_samples, len(real_paths))):
                real_t = (to_tensor(Image.open(rp).convert("RGB")) * 2 - 1).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    best = min(best, lpips_fn(syn_t, real_t).item())
            cat_lpips.append(best)

        mean_lp = round(float(np.mean(cat_lpips)), 4)
        gen_quality[cat] = {
            "lpips":  mean_lp,
            "n_syn":  len(syn_paths),
            "n_real": len(real_paths),
        }
        print(f"  {cat}: LPIPS={mean_lp:.4f}  ({len(syn_paths)} syn, {len(real_paths)} real)")

    del lpips_fn; gc.collect(); torch.cuda.empty_cache()
    return gen_quality


# ── Paper figures ─────────────────────────────────────────────────────────────

def add_label(img: Image.Image, text: str) -> Image.Image:
    """Overlay a black banner with white text at the top of an image."""
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


def make_qualitative_grid(
    showcase: list[tuple[str, str]],
    thumb: int = 160,
    output_path: str = None,
) -> Image.Image:
    """
    Build a qualitative figure: Normal | Mask | Synthetic | Real | Counterfactual.

    Args:
        showcase:    List of (category, defect_type) pairs to visualise
        thumb:       Thumbnail size per cell in pixels
        output_path: Where to save the PNG (optional)

    Returns:
        PIL Image of the grid
    """
    rows = []

    for cat, dt in showcase:
        normal_paths = get_image_paths(cat, "train", "good")
        if not normal_paths:
            continue

        normal = Image.open(normal_paths[0]).convert("RGB").resize((thumb, thumb))
        normal = add_label(normal, "Normal")

        syn_dir    = Path(SYN_DIR) / cat / dt
        mask_files = sorted(syn_dir.glob("syn_*_mask.png")) if syn_dir.exists() else []
        mask_img   = (
            Image.open(mask_files[0]).convert("RGB").resize((thumb, thumb))
            if mask_files else Image.new("RGB", (thumb, thumb), (40, 40, 40))
        )
        mask_img = add_label(mask_img, "Mask")

        ver_dir    = Path(VERIFIED_DIR) / cat / dt
        gen_files  = (
            [g for g in sorted(ver_dir.glob("syn_*[0-9].png"))
             if "mask" not in g.name and "normal" not in g.name]
            if ver_dir.exists() else []
        )
        if not gen_files and syn_dir.exists():
            gen_files = [
                g for g in sorted(syn_dir.glob("syn_*[0-9].png"))
                if "mask" not in g.name and "normal" not in g.name
            ]
        gen_img = (
            Image.open(gen_files[0]).convert("RGB").resize((thumb, thumb))
            if gen_files else Image.new("RGB", (thumb, thumb), (40, 40, 40))
        )
        gen_img = add_label(gen_img, "Synthetic")

        real_paths = get_image_paths(cat, "test", dt)
        real_img   = (
            Image.open(real_paths[0]).convert("RGB").resize((thumb, thumb))
            if real_paths else Image.new("RGB", (thumb, thumb), (40, 40, 40))
        )
        real_img = add_label(real_img, "Real Defect")

        cf_path = Path(CF_DIR) / cat / dt / "counterfactual.png"
        cf_img  = (
            Image.open(cf_path).convert("RGB").resize((thumb, thumb))
            if cf_path.exists() else Image.new("RGB", (thumb, thumb), (40, 40, 40))
        )
        cf_img = add_label(cf_img, "Counterfactual")

        rows.append((normal, mask_img, gen_img, real_img, cf_img))

    if not rows:
        return None

    col_labels = ["Normal", "Hypothesis Mask", "Synth. Defect", "Real Defect", "Counterfactual"]
    n_cols     = 5
    n_rows     = len(rows)
    header_h   = 22
    gap        = 3
    grid_w     = thumb * n_cols + gap * (n_cols - 1)
    grid_h     = header_h + thumb * n_rows + gap * (n_rows - 1)

    grid   = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw_g = ImageDraw.Draw(grid)
    try:
        hfont = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
        )
    except Exception:
        hfont = ImageFont.load_default()

    for c, label in enumerate(col_labels):
        x = c * (thumb + gap) + thumb // 2
        draw_g.text((x - len(label) * 3, 4), label, fill="black", font=hfont)

    for r, row_imgs in enumerate(rows):
        y0 = header_h + r * (thumb + gap)
        for c, img in enumerate(row_imgs):
            x0 = c * (thumb + gap)
            grid.paste(img, (x0, y0))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid.save(output_path)

    return grid


def plot_stage_pass_rates(pipeline_metrics: dict, output_path: str = None) -> None:
    """Bar chart of per-stage pass rates averaged across all categories."""
    all_stages = defaultdict(list)
    for m in pipeline_metrics.values():
        for stage, rate in m.get("stage_pass_rates", {}).items():
            all_stages[stage].append(rate)

    stages = list(all_stages.keys())
    rates  = [float(np.mean(all_stages[s])) for s in stages]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(stages, rates)
    ax.set_ylabel("Pass Rate")
    ax.set_title("Verification Stage Pass Rates (mean across categories)")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_acceptance_rates(pipeline_metrics: dict, output_path: str = None) -> None:
    """Bar chart of acceptance rates per category."""
    cats  = list(pipeline_metrics.keys())
    rates = [pipeline_metrics[c]["accept_rate"] for c in cats]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(cats, rates)
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Verification Acceptance Rate per Category")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path}")
    plt.close()
