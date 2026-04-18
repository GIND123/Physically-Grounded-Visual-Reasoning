"""
Central configuration — paths, hyperparameters, constants.
Override ROOT to match your environment.
"""

import os
import torch

# ── Root directory ────────────────────────────────────────────────────────────
ROOT = os.environ.get("PGVR_ROOT", "/root/project")

# ── Dataset ──────────────────────────────────────────────────────────────────
MVTEC        = f"{ROOT}/datasets/mvtec_ad"
CFG_DIR      = f"{ROOT}/configs"
KB_DIR       = f"{ROOT}/knowledge_base"
SOP_DIR      = f"{KB_DIR}/sops"
FMEA_DIR     = f"{KB_DIR}/fmeas"

# ── Checkpoints ──────────────────────────────────────────────────────────────
CKPT_DIR     = f"{ROOT}/checkpoints"
LORA_DIR     = f"{CKPT_DIR}/sd_lora"
FEATURES_DIR = f"{CKPT_DIR}/features"
SAM_CKPT     = f"{CKPT_DIR}/sam_vit_h_4b8939.pth"

# ── Stage 1 outputs ───────────────────────────────────────────────────────────
TRIPLET_DIR  = f"{ROOT}/training_triplets"
EVAL_S1_DIR  = f"{ROOT}/results/stage1_validation"

# ── Stage 2 outputs ───────────────────────────────────────────────────────────
S2_DIR       = f"{ROOT}/stage2_outputs"
HYPO_DIR     = f"{S2_DIR}/hypotheses"
SYN_DIR      = f"{S2_DIR}/synthetic"
VERIFIED_DIR = f"{S2_DIR}/verified"
CF_DIR       = f"{S2_DIR}/counterfactual"
REPORT_DIR   = f"{S2_DIR}/reports"
RAG_DIR      = f"{S2_DIR}/rag_db"

# ── Stage 3 outputs ───────────────────────────────────────────────────────────
S3_DIR       = f"{ROOT}/stage3_results"
FIGURES_DIR  = f"{S3_DIR}/figures"
TABLES_DIR   = f"{S3_DIR}/tables"

# ── API keys (set via environment variables, never hardcode) ──────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")

# ── Model IDs ─────────────────────────────────────────────────────────────────
BASE_MODEL_ID      = "sd2-community/stable-diffusion-2-inpainting"
CONTROLNET_MODEL   = "thibaud/controlnet-sd21-canny-diffusers"
HF_REPO_ID         = "GOVINDFROM/Industrial-Visual-Reasoning-GenAI"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Image size ────────────────────────────────────────────────────────────────
IMG_SIZE = 512

# ── MVTec categories ─────────────────────────────────────────────────────────
ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill",
    "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]

# ── LoRA training config ──────────────────────────────────────────────────────
LORA_CONFIG = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "max_steps_per_category": 1500,
    "min_steps_per_category": 400,
    "steps_per_sample": 15,
    "batch_size": 1,
    "grad_accum": 4,
    "warmup_ratio": 0.1,
    "save_every": 500,
    "resolution": 512,
    "mixed_precision": True,
}

# ── Synthesis config ──────────────────────────────────────────────────────────
N_SYNTHETIC_PER_DEFECT = 3

# ── Verification thresholds ───────────────────────────────────────────────────
VERIFICATION_THRESHOLDS = {
    "clip_consistency":    0.20,
    "winclip_separation":  0.005,
    "ssim_preservation":   0.50,
    "dinov2_inside_anomaly": 0.02,
    "region_diff_threshold": 0.03,
}
