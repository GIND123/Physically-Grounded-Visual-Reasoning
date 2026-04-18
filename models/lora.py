"""
LoRA fine-tuning for Stable Diffusion Inpainting on MVTec AD categories.

Each category gets its own LoRA adapter trained on (normal, mask, defect) triplets.
The adapter learns the visual vocabulary of that category's real defect textures.
"""

import os
import json
import math
import gc
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datetime import datetime

from config.settings import (
    DEVICE, LORA_DIR, TRIPLET_DIR, BASE_MODEL_ID, LORA_CONFIG,
)
from data.dataset import InpaintingTripletDataset


def train_lora_for_category(category: str, config: dict = None) -> bool:
    """
    Fine-tune a LoRA adapter for one MVTec category.

    Args:
        category: MVTec category name (e.g. "bottle")
        config:   Optional override dict for LORA_CONFIG keys

    Returns:
        True if training completed (or was already done), False on skip/failure.
    """
    from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model

    cfg = {**LORA_CONFIG, **(config or {})}

    cat_lora_dir = os.path.join(LORA_DIR, category)
    done_flag    = os.path.join(cat_lora_dir, ".done")

    if os.path.exists(done_flag):
        print(f"  {category}: LoRA already trained (skipping)")
        return True

    manifest_path = os.path.join(TRIPLET_DIR, category, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"  {category}: No triplets found (skipping)")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)
    n_triplets = manifest["n_triplets"]
    if n_triplets < 5:
        print(f"  {category}: Only {n_triplets} triplets — too few (skipping)")
        return False

    os.makedirs(cat_lora_dir, exist_ok=True)

    steps = min(
        cfg["max_steps_per_category"],
        max(cfg["min_steps_per_category"], n_triplets * cfg["steps_per_sample"]),
    )
    print(f"\n  {category}: {n_triplets} triplets → {steps} steps")

    # ── Load base model ───────────────────────────────────────────────────────
    tokenizer    = CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_ID, subfolder="text_encoder").to(DEVICE)
    vae          = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae").to(DEVICE)
    unet         = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet").to(DEVICE)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # ── Attach LoRA ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        init_lora_weights="gaussian",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = InpaintingTripletDataset(manifest_path, resolution=cfg["resolution"])
    dataloader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
    )
    warmup_steps = int(steps * cfg["warmup_ratio"])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / max(warmup_steps, 1))
        * (0.5 * (1 + math.cos(math.pi * step / steps))),
    )
    scaler = torch.amp.GradScaler("cuda") if cfg["mixed_precision"] else None

    # ── Training loop ─────────────────────────────────────────────────────────
    unet.train(); text_encoder.eval(); vae.eval()

    data_iter = iter(dataloader)
    loss_log  = []
    pbar      = tqdm(range(steps), desc=f"  {category}")

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with torch.amp.autocast("cuda", enabled=cfg["mixed_precision"]):
            target_img = batch["defect"].to(DEVICE)
            latents    = vae.encode(target_img).latent_dist.sample() * vae.config.scaling_factor

            normal_img     = batch["normal"].to(DEVICE)
            normal_latents = vae.encode(normal_img).latent_dist.sample() * vae.config.scaling_factor

            mask       = batch["mask"].to(DEVICE)
            mask_latent = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")

            noise     = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=DEVICE,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # SD 2 inpainting: 9-channel input
            masked_normal_latents = normal_latents * (1 - mask_latent)
            unet_input = torch.cat([noisy_latents, mask_latent, masked_normal_latents], dim=1)

            text_inputs = tokenizer(
                batch["prompt"], padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).to(DEVICE)
            encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]

            noise_pred = unet(unet_input, timesteps, encoder_hidden_states).sample

            # Only compute loss in the mask region
            loss_mask = F.interpolate(mask, size=noise_pred.shape[-2:], mode="nearest")
            loss = F.mse_loss(noise_pred * loss_mask, noise * loss_mask, reduction="sum")
            loss = loss / loss_mask.sum().clamp(min=1)

        if scaler:
            scaler.scale(loss).backward()
            if (step + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

        lr_scheduler.step()
        loss_log.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if (step + 1) % cfg["save_every"] == 0:
            ckpt_path = os.path.join(cat_lora_dir, f"checkpoint-{step + 1}")
            unet.save_pretrained(ckpt_path)

    # ── Save final adapter ────────────────────────────────────────────────────
    unet.save_pretrained(cat_lora_dir)

    meta = {
        "category":   category,
        "base_model": BASE_MODEL_ID,
        "total_steps":   steps,
        "n_triplets":    n_triplets,
        "final_loss":    round(sum(loss_log[-50:]) / max(len(loss_log[-50:]), 1), 6),
        "lora_rank":     cfg["rank"],
        "lora_alpha":    cfg["alpha"],
        "trained_at":    datetime.now().isoformat(),
    }
    with open(os.path.join(cat_lora_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    open(done_flag, "w").close()
    print(f"  ✓ {category}: saved to {cat_lora_dir} (loss={meta['final_loss']:.4f})")

    # Cleanup
    del unet, text_encoder, vae, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return True


def load_lora_pipeline(category: str):
    """
    Load a StableDiffusionInpaintPipeline with the category's LoRA adapter.
    Returns the pipeline on DEVICE (float16).
    """
    from diffusers import StableDiffusionInpaintPipeline
    from peft import PeftModel

    cat_lora_dir = os.path.join(LORA_DIR, category)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
    ).to(DEVICE)

    if os.path.exists(os.path.join(cat_lora_dir, ".done")):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, cat_lora_dir, is_trainable=False)
        pipe.unet.eval().to(device=DEVICE, dtype=torch.float16)

    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    return pipe
