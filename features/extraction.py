"""
DINOv2 feature extraction and FAISS indexing for normal/test images.

Extracted features are used by:
  - Stage 1: triplet building (find nearest normal for each defect)
  - Stage 2: VerificationCritic stage 4 (patch-level anomaly)
  - Stage 3: evaluation
"""

import os
import json
import torch
import numpy as np
import faiss
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms

from config.settings import DEVICE, FEATURES_DIR, MVTEC
from data.paths import get_image_paths, get_defect_types


# ── DINOv2 transform ──────────────────────────────────────────────────────────

dinov2_transform = transforms.Compose([
    transforms.Resize((518, 518)),   # required input size for dinov2_vitb14
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_dinov2():
    """Load DINOv2 ViT-B/14 from torch hub."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    return model.eval().to(DEVICE)


@torch.no_grad()
def extract_cls_feature(img_path: str, model) -> torch.Tensor:
    """Extract DINOv2 CLS token for a single image. Returns 1-D tensor."""
    img = Image.open(img_path).convert("RGB")
    x = dinov2_transform(img).unsqueeze(0).to(DEVICE)
    out = model.forward_features(x)
    return out["x_norm_clstoken"].squeeze(0).cpu()


@torch.no_grad()
def extract_features_batch(
    image_paths: list[str],
    model,
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract CLS and patch features for a list of images.

    Returns:
        cls_feats   : (N, D) tensor of CLS tokens
        patch_feats : (N, P, D) tensor of patch tokens (subsampled every 10th)
    """
    all_cls, all_patches = [], []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = [dinov2_transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(DEVICE)
        out = model.forward_features(batch)

        all_cls.append(out["x_norm_clstoken"].cpu())
        # Subsample patches to save memory
        patches = out["x_norm_patchtokens"].cpu()
        all_patches.append(patches[:, ::10, :])

    return torch.cat(all_cls, dim=0), torch.cat(all_patches, dim=0)


def build_faiss_index(
    category: str,
    model,
    force: bool = False,
) -> None:
    """
    Extract train + test features for a category and build a FAISS index.

    Saves to FEATURES_DIR/{category}/:
        cls_features.pt       — train CLS features
        patch_features.pt     — train patch features (subsampled)
        test_cls_features.pt  — test CLS features
        test_patch_features.pt
        test_metadata.json    — paths, labels, defect types
        faiss_cls.index       — flat inner-product index (cosine after L2-norm)
    """
    cat_feat_dir = os.path.join(FEATURES_DIR, category)
    os.makedirs(cat_feat_dir, exist_ok=True)

    train_cls_path = os.path.join(cat_feat_dir, "cls_features.pt")
    test_cls_path  = os.path.join(cat_feat_dir, "test_cls_features.pt")

    if not force and os.path.exists(test_cls_path):
        return  # already done

    # ── Train features ────────────────────────────────────────────────────────
    train_paths = get_image_paths(category, "train", "good")
    if not os.path.exists(train_cls_path):
        train_cls, train_patches = extract_features_batch(train_paths, model)
        torch.save(train_cls,    train_cls_path)
        torch.save(train_patches, os.path.join(cat_feat_dir, "patch_features.pt"))
    else:
        train_cls = torch.load(train_cls_path, map_location="cpu")

    # ── Test features ─────────────────────────────────────────────────────────
    test_paths, test_labels, test_defect_types = [], [], []
    for p in get_image_paths(category, "test", "good"):
        test_paths.append(p); test_labels.append(0); test_defect_types.append("good")
    for dt in get_defect_types(category):
        for p in get_image_paths(category, "test", dt):
            test_paths.append(p); test_labels.append(1); test_defect_types.append(dt)

    if test_paths:
        test_cls, test_patches = extract_features_batch(test_paths, model)
        torch.save(test_cls,    test_cls_path)
        torch.save(test_patches, os.path.join(cat_feat_dir, "test_patch_features.pt"))
        with open(os.path.join(cat_feat_dir, "test_metadata.json"), "w") as f:
            json.dump({"paths": test_paths, "labels": test_labels,
                       "defect_types": test_defect_types}, f, indent=2)

    # ── FAISS index ───────────────────────────────────────────────────────────
    train_np = train_cls.numpy().astype(np.float32)
    faiss.normalize_L2(train_np)                         # cosine = inner-product after norm
    index = faiss.IndexFlatIP(train_np.shape[1])
    index.add(train_np)
    faiss.write_index(index, os.path.join(cat_feat_dir, "faiss_cls.index"))
