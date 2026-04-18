"""
5-Stage Verification Critic for generated defect images.

Each generated image must pass at least 4 of 5 independent checks:

  Stage 1 — CLIP text-image consistency
  Stage 2 — WinCLIP anomaly scoring (normal vs defect prompts)
  Stage 3 — SSIM structure preservation (outside the defect mask)
  Stage 4 — DINOv2 patch-level anomaly localization (inside vs outside mask)
  Stage 5 — Defect region pixel intensity difference

Verdict:
  ACCEPT      ≥ 4 stages passed
  SOFT_REJECT ≥ 3 stages passed
  HARD_REJECT < 3 stages passed
"""

import math
import gc
import shutil
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config.settings import DEVICE, IMG_SIZE, VERIFICATION_THRESHOLDS, SYN_DIR, VERIFIED_DIR, HYPO_DIR
from data.paths import get_image_paths
from features.extraction import dinov2_transform


# ── Module-level model handles (loaded once and reused) ───────────────────────
_clip_model       = None
_clip_preprocess  = None
_clip_tokenizer   = None
_dinov2           = None
_ssim_fn          = None


def load_verification_models():
    """Load CLIP, DINOv2, and SSIM — call once before running verification."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _dinov2, _ssim_fn

    import open_clip
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    print("Loading verification models ...")
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k",
    )
    _clip_model   = _clip_model.eval().to(DEVICE)
    _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    _dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    _dinov2 = _dinov2.eval().to(DEVICE)

    _ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    print("  Verification models loaded")


def unload_verification_models():
    """Release verification models from GPU memory."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _dinov2, _ssim_fn
    _clip_model = _clip_preprocess = _clip_tokenizer = _dinov2 = _ssim_fn = None
    gc.collect()
    torch.cuda.empty_cache()


class VerificationCritic:
    """5-stage verification for generated defect images."""

    THRESHOLDS = VERIFICATION_THRESHOLDS

    @staticmethod
    @torch.no_grad()
    def stage1_clip_consistency(gen_image: Image.Image, defect_prompt: str) -> dict:
        """Check if generated image matches the defect description."""
        img_t  = _clip_preprocess(gen_image).unsqueeze(0).to(DEVICE)
        text_t = _clip_tokenizer([defect_prompt]).to(DEVICE)

        img_feat = F.normalize(_clip_model.encode_image(img_t), dim=-1)
        txt_feat = F.normalize(_clip_model.encode_text(text_t), dim=-1)

        sim    = (img_feat @ txt_feat.T).item()
        passed = sim > VerificationCritic.THRESHOLDS["clip_consistency"]
        return {"score": round(sim, 4), "passed": passed, "stage": "clip_consistency"}

    @staticmethod
    @torch.no_grad()
    def stage2_winclip_anomaly(gen_image: Image.Image, category: str) -> dict:
        """WinCLIP-style: compare similarity to normal vs anomaly prompts."""
        cat_name = category.replace("_", " ")
        img_t    = _clip_preprocess(gen_image).unsqueeze(0).to(DEVICE)

        normal_prompts  = [
            f"a photo of a normal {cat_name}",
            f"a photo of a perfect {cat_name} with no defects",
        ]
        anomaly_prompts = [
            f"a photo of a damaged {cat_name}",
            f"a photo of a {cat_name} with defects",
        ]

        img_feat = F.normalize(_clip_model.encode_image(img_t), dim=-1)

        normal_tok  = _clip_tokenizer(normal_prompts).to(DEVICE)
        anomaly_tok = _clip_tokenizer(anomaly_prompts).to(DEVICE)

        normal_sim  = (img_feat @ F.normalize(_clip_model.encode_text(normal_tok),  dim=-1).T).mean().item()
        anomaly_sim = (img_feat @ F.normalize(_clip_model.encode_text(anomaly_tok), dim=-1).T).mean().item()

        separation = anomaly_sim - normal_sim
        passed     = separation > VerificationCritic.THRESHOLDS["winclip_separation"]
        return {
            "normal_sim":  round(normal_sim,  4),
            "anomaly_sim": round(anomaly_sim, 4),
            "separation":  round(separation,  4),
            "passed":      passed,
            "stage":       "winclip_anomaly",
        }

    @staticmethod
    @torch.no_grad()
    def stage3_ssim_preservation(
        gen_image: Image.Image,
        normal_image: Image.Image,
        mask: Image.Image,
    ) -> dict:
        """Check that non-defect regions are structurally preserved."""
        gen_t  = transforms.ToTensor()(gen_image).unsqueeze(0).to(DEVICE)
        norm_t = transforms.ToTensor()(normal_image).unsqueeze(0).to(DEVICE)
        mask_t = transforms.ToTensor()(mask).unsqueeze(0).to(DEVICE)

        inv_mask = (mask_t < 0.5).float()

        if inv_mask.sum() < 1000:
            return {"score": 1.0, "passed": True, "stage": "ssim_preservation",
                    "note": "mask covers most of image"}

        ssim_val = _ssim_fn(gen_t * inv_mask, norm_t * inv_mask).item()
        passed   = ssim_val > VerificationCritic.THRESHOLDS["ssim_preservation"]
        return {"score": round(ssim_val, 4), "passed": passed, "stage": "ssim_preservation"}

    @staticmethod
    @torch.no_grad()
    def stage4_dinov2_patch_anomaly(
        gen_image: Image.Image,
        normal_image: Image.Image,
        mask: Image.Image,
    ) -> dict:
        """Check that patches inside the mask are more anomalous than outside."""
        gen_t  = dinov2_transform(gen_image).unsqueeze(0).to(DEVICE)
        norm_t = dinov2_transform(normal_image).unsqueeze(0).to(DEVICE)

        gen_feats  = _dinov2.forward_features(gen_t)["x_norm_patchtokens"].squeeze(0)
        norm_feats = _dinov2.forward_features(norm_t)["x_norm_patchtokens"].squeeze(0)

        patch_dists = (1 - F.cosine_similarity(gen_feats, norm_feats, dim=-1)).cpu().numpy()

        n_patches = gen_feats.shape[0]
        grid_size = int(math.sqrt(n_patches))

        mask_np   = np.array(mask.resize((grid_size, grid_size)))
        mask_flat = (mask_np.flatten() > 127)[:n_patches]

        inside_mean  = patch_dists[mask_flat].mean()  if mask_flat.any()  else 0.0
        outside_mean = patch_dists[~mask_flat].mean() if (~mask_flat).any() else 0.0

        diff   = inside_mean - outside_mean
        passed = diff > VerificationCritic.THRESHOLDS["dinov2_inside_anomaly"]
        return {
            "inside_anomaly":  round(float(inside_mean),  4),
            "outside_anomaly": round(float(outside_mean), 4),
            "diff":            round(float(diff),          4),
            "passed":          passed,
            "stage":           "dinov2_patch_anomaly",
        }

    @staticmethod
    def stage5_region_difference(
        gen_image: Image.Image,
        normal_image: Image.Image,
        mask: Image.Image,
    ) -> dict:
        """Check that the masked region actually changed relative to the normal."""
        gen_np  = np.array(gen_image.convert("RGB")).astype(float)  / 255
        norm_np = np.array(normal_image.convert("RGB")).astype(float) / 255
        mask_np = np.array(mask.resize(gen_image.size)) > 127

        if not mask_np.any():
            return {"score": 0, "passed": False, "stage": "region_difference"}

        diff        = np.abs(gen_np - norm_np).mean(axis=2)
        inside_diff = diff[mask_np].mean()

        passed = inside_diff > VerificationCritic.THRESHOLDS["region_diff_threshold"]
        return {"score": round(float(inside_diff), 4), "passed": passed,
                "stage": "region_difference"}

    @classmethod
    def verify(
        cls,
        gen_image: Image.Image,
        normal_image: Image.Image,
        mask: Image.Image,
        category: str,
        defect_prompt: str,
    ) -> dict:
        """
        Run all 5 verification stages and return verdict + per-stage results.

        Verdict:
            "ACCEPT"      — ≥ 4 stages passed
            "SOFT_REJECT" — 3 stages passed
            "HARD_REJECT" — < 3 stages passed
        """
        results = {
            "s1": cls.stage1_clip_consistency(gen_image, defect_prompt),
            "s2": cls.stage2_winclip_anomaly(gen_image, category),
            "s3": cls.stage3_ssim_preservation(gen_image, normal_image, mask),
            "s4": cls.stage4_dinov2_patch_anomaly(gen_image, normal_image, mask),
            "s5": cls.stage5_region_difference(gen_image, normal_image, mask),
        }

        n_passed = sum(1 for r in results.values() if r["passed"])

        if n_passed >= 4:
            verdict = "ACCEPT"
        elif n_passed >= 3:
            verdict = "SOFT_REJECT"
        else:
            verdict = "HARD_REJECT"

        return {
            "verdict":       verdict,
            "stages_passed": n_passed,
            "stages_total":  len(results),
            "details":       results,
        }


# ── Counterfactual scoring helper ─────────────────────────────────────────────

@torch.no_grad()
def compute_anomaly_score_dinov2(image: Image.Image, reference: Image.Image) -> float:
    """DINOv2 feature distance — higher means more anomalous vs reference."""
    img_t = dinov2_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    ref_t = dinov2_transform(reference.convert("RGB")).unsqueeze(0).to(DEVICE)
    img_f = _dinov2.forward_features(img_t)["x_norm_patchtokens"].squeeze(0)
    ref_f = _dinov2.forward_features(ref_t)["x_norm_patchtokens"].squeeze(0)
    return (1 - F.cosine_similarity(img_f, ref_f, dim=-1)).mean().item()


# ── Batch verification runner ──────────────────────────────────────────────────

def run_verification(
    all_categories: list[str],
    s2_dir: str,
) -> tuple[dict, int, int, int]:
    """
    Run verification on all synthetic images across all categories.

    Returns:
        (verification_results dict, accept_count, soft_reject_count, hard_reject_count)
    """
    verification_results = {}
    accept_count = soft_reject_count = hard_reject_count = 0

    for cat in all_categories:
        cat_syn_dir  = Path(SYN_DIR) / cat
        cat_hypo_dir = Path(HYPO_DIR) / cat
        cat_ver_dir  = Path(VERIFIED_DIR) / cat

        if not cat_syn_dir.exists():
            continue

        normal_paths = get_image_paths(cat, "train", "good")
        if not normal_paths:
            continue

        ref_normal   = Image.open(normal_paths[0]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        cat_verify   = {}

        print(f"\n── {cat} ──")

        for dt_dir in sorted(cat_syn_dir.iterdir()):
            if not dt_dir.is_dir():
                continue
            dt = dt_dir.name
            (cat_ver_dir / dt).mkdir(parents=True, exist_ok=True)

            hypo_path = cat_hypo_dir / f"{dt}.json"
            if hypo_path.exists():
                with open(hypo_path) as f:
                    hypo = json.load(f)
                defect_prompt = hypo.get(
                    "generation_prompt",
                    f"a {dt.replace('_', ' ')} defect on a {cat.replace('_', ' ')}",
                )
            else:
                defect_prompt = f"a {dt.replace('_', ' ')} defect on a {cat.replace('_', ' ')}"

            dt_scores = []

            for syn_path in sorted(dt_dir.glob("syn_*[0-9].png")):
                if "mask" in syn_path.name or "normal" in syn_path.name:
                    continue

                mask_path   = syn_path.parent / f"{syn_path.stem}_mask.png"
                normal_path = syn_path.parent / f"{syn_path.stem}_normal.png"

                gen_img = Image.open(syn_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                norm_img = (
                    Image.open(normal_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    if normal_path.exists() else ref_normal
                )
                mask_img = (
                    Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
                    if mask_path.exists() else Image.new("L", (IMG_SIZE, IMG_SIZE), 128)
                )

                result        = VerificationCritic.verify(gen_img, norm_img, mask_img, cat, defect_prompt)
                result["file"] = syn_path.name
                dt_scores.append(result)

                if result["verdict"] == "ACCEPT":
                    shutil.copy2(syn_path,  cat_ver_dir / dt / syn_path.name)
                    if mask_path.exists():
                        shutil.copy2(mask_path, cat_ver_dir / dt / mask_path.name)
                    accept_count += 1
                elif result["verdict"] == "SOFT_REJECT":
                    soft_reject_count += 1
                else:
                    hard_reject_count += 1

            if dt_scores:
                n_acc = sum(1 for s in dt_scores if s["verdict"] == "ACCEPT")
                print(f"    {dt}: {n_acc}/{len(dt_scores)} accepted")

            cat_verify[dt] = {
                "scores":     dt_scores,
                "n_total":    len(dt_scores),
                "n_accepted": sum(1 for s in dt_scores if s["verdict"] == "ACCEPT"),
            }

        verification_results[cat] = cat_verify

    return verification_results, accept_count, soft_reject_count, hard_reject_count
