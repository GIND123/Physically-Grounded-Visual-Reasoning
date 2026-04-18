"""
Stage 2 pipeline — Agentic Reasoning Pipeline.

Steps:
  1. Build RAG knowledge base from SOPs/FMEAs
  2. Generate structured defect hypotheses via GPT-4o vision
  3. Synthesize defect images (LoRA + ControlNet + hypothesis prompt)
  4. 5-stage verification of generated images
  5. Counterfactual generation and scoring
  6. Operator report generation
"""

import os
import gc
import json
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
from PIL import Image

from config.settings import (
    ALL_CATEGORIES, DEVICE, IMG_SIZE, OPENAI_API_KEY,
    S2_DIR, HYPO_DIR, SYN_DIR, VERIFIED_DIR, CF_DIR, REPORT_DIR, RAG_DIR,
    LORA_DIR, MVTEC, BASE_MODEL_ID,
)
from data.paths import get_image_paths, get_defect_types
from synthesis.rag import build_knowledge_base, retrieve_evidence
from synthesis.llm import generate_hypothesis, generate_operator_report
from synthesis.generator import synthesize_defects_for_category
from verification.critic import (
    load_verification_models, unload_verification_models,
    VerificationCritic, run_verification, compute_anomaly_score_dinov2,
)
from synthesis.mask_gen import generate_defect_mask
from features.extraction import dinov2_transform


# ── Step 1: Build RAG KB ──────────────────────────────────────────────────────

def build_rag(force: bool = False):
    """Build ChromaDB knowledge base from SOPs/FMEAs."""
    print("Building RAG knowledge base ...")
    return build_knowledge_base(rag_dir=RAG_DIR, force=force)


# ── Step 2: Generate hypotheses ───────────────────────────────────────────────

def generate_all_hypotheses(
    collection,
    embedder,
    total_chunks: int,
    categories: list[str] = None,
    api_key: str = None,
) -> dict[str, int]:
    """
    Generate GPT-4o hypotheses for all (category, defect_type) pairs.
    Returns stats dict.
    """
    cats  = categories or ALL_CATEGORIES
    key   = api_key or OPENAI_API_KEY
    stats = defaultdict(int)

    for cat in cats:
        cat_hypo_dir = os.path.join(HYPO_DIR, cat)
        os.makedirs(cat_hypo_dir, exist_ok=True)

        normal_paths  = get_image_paths(cat, "train", "good")
        if not normal_paths:
            continue
        ref_normals   = normal_paths[:3]
        defect_types  = get_defect_types(cat)

        print(f"\n── {cat} ({len(defect_types)} defect types) ──")

        for dt in defect_types:
            hypo_path = os.path.join(cat_hypo_dir, f"{dt}.json")

            if os.path.exists(hypo_path):
                with open(hypo_path) as f:
                    existing = json.load(f)
                if existing.get("_meta", {}).get("generation_method") == "gpt4o_vision_rag":
                    print(f"    {dt}: cached")
                    stats["cached"] += 1
                    continue

            evidence  = retrieve_evidence(collection, embedder, cat, dt, total_chunks)
            ref_idx   = defect_types.index(dt) % len(ref_normals)
            ref_normal = ref_normals[ref_idx]

            print(f"    {dt}: generating ...", end=" ", flush=True)
            hypothesis = generate_hypothesis(cat, dt, ref_normal, evidence, api_key=key)

            if hypothesis:
                with open(hypo_path, "w") as f:
                    json.dump(hypothesis, f, indent=2)
                bbox      = hypothesis.get("defect_bbox_normalized", [])
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if len(bbox) == 4 else 0
                print(f"bbox_area={bbox_area:.2%} OK")
                stats["generated"] += 1
            else:
                stats["failed"] += 1

            time.sleep(0.5)  # rate limiting

    return stats


# ── Step 3: Synthesis ─────────────────────────────────────────────────────────

def synthesize_all(categories: list[str] = None) -> dict:
    """Run LoRA + ControlNet synthesis for all categories."""
    from models.controlnet import load_controlnet

    cats = categories or ALL_CATEGORIES
    print("Loading ControlNet ...")
    controlnet = load_controlnet()

    synthesis_report = {}
    for cat in cats:
        print(f"\n── {cat} ──")
        results = synthesize_defects_for_category(cat, controlnet_model=controlnet)
        synthesis_report[cat] = results

    del controlnet
    gc.collect()
    torch.cuda.empty_cache()

    with open(os.path.join(S2_DIR, "synthesis_report.json"), "w") as f:
        json.dump(synthesis_report, f, indent=2)

    total = sum(sum(v.values()) for v in synthesis_report.values() if isinstance(v, dict))
    print(f"\nSynthesis complete: {total} images generated")
    return synthesis_report


# ── Step 4: Verification ──────────────────────────────────────────────────────

def verify_all(categories: list[str] = None) -> dict:
    """Run 5-stage verification and save results."""
    cats = categories or ALL_CATEGORIES

    load_verification_models()
    verification_results, accept, soft, hard = run_verification(cats, S2_DIR)
    unload_verification_models()

    with open(os.path.join(S2_DIR, "verification_results.json"), "w") as f:
        json.dump(verification_results, f, indent=2, default=str)

    total = accept + soft + hard
    print(f"\nVerification: {accept} ACCEPT / {soft} SOFT / {hard} HARD (total={total})")
    return verification_results


# ── Step 5: Counterfactual ────────────────────────────────────────────────────

def generate_counterfactuals(
    categories: list[str] = None,
    verification_results: dict = None,
) -> dict:
    """
    Generate counterfactual (corrected) images and compute defect suppression scores.
    """
    from diffusers import StableDiffusionInpaintPipeline

    cats = categories or ALL_CATEGORIES

    print("Loading DINOv2 and SD pipeline for counterfactuals ...")
    import torch.hub
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    dinov2 = dinov2.eval().to(DEVICE)

    cf_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
    ).to(DEVICE)
    cf_pipe.set_progress_bar_config(disable=True)
    cf_pipe.enable_attention_slicing()

    counterfactual_results = {}

    for cat in cats:
        cat_hypo_dir = os.path.join(HYPO_DIR, cat)
        cat_cf_dir   = os.path.join(CF_DIR, cat)
        if not os.path.exists(cat_hypo_dir):
            continue

        normal_paths = get_image_paths(cat, "train", "good")
        if not normal_paths:
            continue
        ref_normal = Image.open(normal_paths[0]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

        cat_results = {}
        print(f"\n── {cat} ──")

        for hypo_file in sorted(Path(cat_hypo_dir).glob("*.json")):
            dt = hypo_file.stem
            os.makedirs(os.path.join(cat_cf_dir, dt), exist_ok=True)

            with open(hypo_file) as f:
                hypo = json.load(f)

            bbox       = hypo.get("defect_bbox_normalized", [0.3, 0.3, 0.6, 0.6])
            corrective = hypo.get("corrective_action", "fix the process")
            cf_pred    = hypo.get("counterfactual_prediction", "defect absent")
            gen_prompt = hypo.get("generation_prompt", f"a {dt} defect on {cat}")

            base_img    = Image.open(normal_paths[0]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            defect_mask, _ = generate_defect_mask(base_img, bbox, mask_shape="ellipse")

            factual_prompt = (
                f"{gen_prompt}, industrial inspection photograph, macro, photorealistic"
            )
            cf_prompt = (
                f"pristine {cat.replace('_', ' ')}, {cf_pred}, "
                f"after {corrective}, no defects, clean surface, "
                f"industrial inspection photograph, macro, photorealistic"
            )
            neg = "cartoon, illustration, blurry, watermark, text, unrealistic"

            gen_f = torch.Generator(device=DEVICE).manual_seed(42)
            gen_c = torch.Generator(device=DEVICE).manual_seed(42)

            with torch.autocast("cuda"):
                factual_img = cf_pipe(
                    prompt=factual_prompt, negative_prompt=neg,
                    image=base_img, mask_image=defect_mask,
                    guidance_scale=7.5, num_inference_steps=20,
                    height=IMG_SIZE, width=IMG_SIZE, generator=gen_f,
                ).images[0]

                cf_img = cf_pipe(
                    prompt=cf_prompt, negative_prompt=neg,
                    image=base_img, mask_image=defect_mask,
                    guidance_scale=7.5, num_inference_steps=20,
                    height=IMG_SIZE, width=IMG_SIZE, generator=gen_c,
                ).images[0]

            factual_score = compute_anomaly_score_dinov2(factual_img, ref_normal)
            cf_score      = compute_anomaly_score_dinov2(cf_img, ref_normal)
            suppression   = factual_score - cf_score

            cat_results[dt] = {
                "factual_score":          round(factual_score, 4),
                "counterfactual_score":   round(cf_score,      4),
                "suppression_delta":      round(suppression,   4),
                "suppression_ratio":      round(suppression / max(factual_score, 1e-6), 4),
                "defect_suppressed":      bool(cf_score < factual_score),
                "corrective_action":      corrective,
                "counterfactual_prediction": cf_pred,
            }

            icon = "✓" if cf_score < factual_score else "✗"
            print(f"    {icon} {dt}: factual={factual_score:.4f} → cf={cf_score:.4f}")

            factual_img.save(os.path.join(cat_cf_dir, dt, "factual.png"))
            cf_img.save(os.path.join(cat_cf_dir, dt, "counterfactual.png"))
            base_img.save(os.path.join(cat_cf_dir, dt, "normal_reference.png"))
            defect_mask.save(os.path.join(cat_cf_dir, dt, "mask.png"))

        counterfactual_results[cat] = cat_results

    del cf_pipe, dinov2
    gc.collect()
    torch.cuda.empty_cache()

    with open(os.path.join(S2_DIR, "counterfactual_results.json"), "w") as f:
        json.dump(counterfactual_results, f, indent=2)

    total_cf   = sum(len(v) for v in counterfactual_results.values())
    suppressed = sum(
        1 for cat_r in counterfactual_results.values()
        for r in cat_r.values() if r["defect_suppressed"]
    )
    print(f"\nCounterfactual: {suppressed}/{total_cf} suppressed")
    return counterfactual_results


# ── Step 6: Reports ───────────────────────────────────────────────────────────

def generate_reports(
    verification_results: dict,
    counterfactual_results: dict,
    report_categories: list[str] = None,
    api_key: str = None,
) -> None:
    """Generate operator reports for key categories."""
    cats = report_categories or ["bottle", "metal_nut", "transistor", "capsule", "leather"]
    key  = api_key or OPENAI_API_KEY

    os.makedirs(REPORT_DIR, exist_ok=True)

    for cat in cats:
        cat_hypo_dir = os.path.join(HYPO_DIR, cat)
        if not os.path.exists(cat_hypo_dir):
            continue

        print(f"\n── {cat} ──")
        for hypo_file in sorted(Path(cat_hypo_dir).glob("*.json"))[:2]:
            dt = hypo_file.stem
            with open(hypo_file) as f:
                hypo = json.load(f)

            ver_data = verification_results.get(cat, {}).get(dt, {})
            cf_data  = counterfactual_results.get(cat, {}).get(dt, {})

            report = generate_operator_report(
                cat, dt, hypo, ver_data, cf_data, api_key=key,
            )
            if report:
                path = os.path.join(REPORT_DIR, f"{cat}_{dt}_report.txt")
                with open(path, "w") as f:
                    f.write(report)
                print(f"    {dt}: report saved")
            time.sleep(0.3)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_stage2(
    categories: list[str] = None,
    api_key: str = None,
    skip_rag: bool = False,
    skip_hypotheses: bool = False,
    skip_synthesis: bool = False,
    skip_verification: bool = False,
    skip_counterfactual: bool = False,
    skip_reports: bool = False,
) -> None:
    """Run the complete Stage 2 pipeline."""
    for d in [S2_DIR, HYPO_DIR, SYN_DIR, VERIFIED_DIR, CF_DIR, REPORT_DIR, RAG_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print("STAGE 2: Agentic Reasoning Pipeline")
    print("=" * 70)

    # 1. RAG
    collection = embedder = None
    total_chunks = 0
    if not skip_rag:
        print("\n[1] Building RAG knowledge base ...")
        collection, embedder, total_chunks = build_rag()

    # 2. Hypotheses
    if not skip_hypotheses and collection is not None:
        print("\n[2] Generating hypotheses via GPT-4o ...")
        stats = generate_all_hypotheses(collection, embedder, total_chunks, categories, api_key)
        print(f"  Generated: {stats['generated']}, Cached: {stats['cached']}, Failed: {stats['failed']}")

    # 3. Synthesis
    synthesis_report = None
    if not skip_synthesis:
        print("\n[3] Synthesizing defect images ...")
        synthesis_report = synthesize_all(categories)

    # 4. Verification
    verification_results = {}
    if not skip_verification:
        print("\n[4] Running 5-stage verification ...")
        verification_results = verify_all(categories)

    # 5. Counterfactual
    counterfactual_results = {}
    if not skip_counterfactual:
        print("\n[5] Generating counterfactuals ...")
        counterfactual_results = generate_counterfactuals(categories, verification_results)

    # 6. Reports
    if not skip_reports and api_key:
        print("\n[6] Generating operator reports ...")
        generate_reports(verification_results, counterfactual_results, api_key=api_key)

    print("\n" + "=" * 70)
    print("STAGE 2 COMPLETE")
    print(f"  Hypotheses:    {HYPO_DIR}/")
    print(f"  Synthetic:     {SYN_DIR}/")
    print(f"  Verified:      {VERIFIED_DIR}/")
    print(f"  Counterfactual:{CF_DIR}/")
    print(f"  Reports:       {REPORT_DIR}/")
    print("=" * 70)
