"""
ClearSpark AI — SparkleNet v3.0 Ultra Dual-T4 Training Script (200,000 Samples)
Engineered for Kaggle Dual Tesla T4 GPUs (32 GB Total VRAM).

Features:
- 200,000 real photography training iterations per epoch across 12 epochs.
- 100% Zero-Transparency Opaque Occlusion + Translucent + Faint Watermarks.
- Anti-Darkening Loss (L_dark) & Luminance Consistency Loss (L_luma) to eliminate dark discoloration.
- Autocast FP16 AMP stability with BCEWithLogitsLoss.
"""

import os
import sys
import time
import math
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sparklenet_v3 import MultiScaleAttentionUNet_v3, UltraContextualInpainter_v3, export_web_weights_v3
from training.dataset_200k import RealPhotoWatermarkDataset200k


# ── Advanced Anti-Darkening & Luminance Loss Functions ────────────────────────

class SobelLoss(nn.Module):
    """Sobel edge loss for enforcing sharp, natural texture boundaries."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, pred, target):
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        p_dx = F.conv2d(pred_gray, self.kx, padding=1)
        p_dy = F.conv2d(pred_gray, self.ky, padding=1)
        p_mag = torch.sqrt(p_dx ** 2 + p_dy ** 2 + 1e-6)

        t_dx = F.conv2d(target_gray, self.kx, padding=1)
        t_dy = F.conv2d(target_gray, self.ky, padding=1)
        t_mag = torch.sqrt(t_dx ** 2 + t_dy ** 2 + 1e-6)

        return F.l1_loss(p_mag, t_mag)


class LuminanceAndAntiDarkLoss(nn.Module):
    """
    Specially engineered loss function to eliminate dark spot/halo discoloration:
    1. Anti-Darkening: Penalizes cases where reconstructed pixels are darker than ground truth.
    2. Luminance Consistency: Enforces mean brightness inside the reconstructed hole
       to match the ground truth and surrounding collar.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        # 1. Anti-Darkening: Heavily penalize if target is brighter than pred in mask region
        # diff > 0 means target was brighter (i.e. pred was too dark!)
        dark_diff = F.relu(target - pred) * mask
        loss_anti_dark = torch.mean(dark_diff ** 2) * 3.0

        # 2. Luminance Consistency between pred hole and target hole
        mask_area = torch.clamp(mask.sum(dim=(2, 3), keepdim=True), min=1.0)
        pred_luma = (pred * mask).sum(dim=(2, 3), keepdim=True) / mask_area
        target_luma = (target * mask).sum(dim=(2, 3), keepdim=True) / mask_area
        loss_luma = F.l1_loss(pred_luma, target_luma)

        return loss_anti_dark + loss_luma * 1.5


# ── Training Engine ──────────────────────────────────────────────────────────

def train_sparklenet_v3_200k(
    image_paths=None,
    epochs=12,
    batch_size=64,
    samples_per_epoch=200000,
    lr=1e-3,
    output_weights="sparklenet_web_weights.json"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()

    print(f"🔥 PyTorch Active: {torch.__version__} | Device: {device} | GPUs: {num_gpus}")
    if num_gpus > 1:
        for i in range(num_gpus):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB VRAM)")

    # 1. Models & Multi-GPU DataParallel
    wal_net = MultiScaleAttentionUNet_v3(in_channels=3, base_channels=16).to(device)
    cis_net = UltraContextualInpainter_v3(in_channels=4, base_channels=16).to(device)

    if num_gpus > 1:
        wal_net = nn.DataParallel(wal_net)
        cis_net = nn.DataParallel(cis_net)
        print("⚡ DataParallel Active across Dual T4 GPUs!")

    # 2. Dataset & DataLoader (200,000 samples)
    dataset = RealPhotoWatermarkDataset200k(
        image_paths=image_paths,
        total_samples=samples_per_epoch,
        patch_size=128
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if num_gpus > 0 else 0,
        pin_memory=(num_gpus > 0),
        drop_last=True
    )

    # 3. Loss Functions & Optimizer
    bce_logits_fn = nn.BCEWithLogitsLoss()
    smooth_l1_fn = nn.SmoothL1Loss()
    l1_fn = nn.L1Loss()
    sobel_fn = SobelLoss().to(device)
    luma_dark_fn = LuminanceAndAntiDarkLoss().to(device)

    optimizer = torch.optim.AdamW(
        list(wal_net.parameters()) + list(cis_net.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader), eta_min=1e-5)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_time = time.time()
    print(f"🚀 Commencing SparkleNet v3.0 Ultra Training: {samples_per_epoch:,} samples/epoch | Epochs: {epochs} | Batch: {batch_size}")

    for epoch in range(1, epochs + 1):
        wal_net.train()
        cis_net.train()
        ep_losses = []
        ep_wal = []
        ep_cis = []

        for step, (wm_img, clean_img, gt_mask, gt_box, gt_conf) in enumerate(dataloader):
            wm_img = wm_img.to(device, non_blocking=True)
            clean_img = clean_img.to(device, non_blocking=True)
            gt_mask = gt_mask.to(device, non_blocking=True)
            gt_box = gt_box.to(device, non_blocking=True)
            gt_conf = gt_conf.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast('cuda', enabled=(device.type == 'cuda')):
                # WAL-Net v3 Forward
                wal_out = wal_net(wm_img)

                # Loss 1: WAL Localization (BCEWithLogits safe)
                loss_mask = bce_logits_fn(wal_out['mask_logits'], gt_mask)
                loss_box = smooth_l1_fn(wal_out['bbox'], gt_box)
                loss_conf = bce_logits_fn(wal_out['conf_logits'], gt_conf)
                loss_wal = loss_mask * 2.0 + loss_box + loss_conf * 0.5

                # CIS-Net v3 Ultra Forward
                pred_mask = (wal_out['mask'].detach() > 0.35).float()
                use_pred = (torch.rand((wm_img.size(0), 1, 1, 1), device=device) > 0.5)
                train_mask = torch.where(use_pred, pred_mask, gt_mask)

                inpainted, synth_patch = cis_net(wm_img, train_mask)

                # Loss 2: CIS Contextual Inpainting & Anti-Darkening
                loss_l1 = l1_fn(inpainted, clean_img)
                loss_dark_luma = luma_dark_fn(inpainted, clean_img, train_mask)
                loss_sobel = sobel_fn(inpainted, clean_img) * 0.25

                loss_cis = loss_l1 + loss_dark_luma + loss_sobel
                total_loss = loss_wal + loss_cis

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(wal_net.parameters()) + list(cis_net.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_losses.append(total_loss.item())
            ep_wal.append(loss_wal.item())
            ep_cis.append(loss_cis.item())

            if (step + 1) % 100 == 0 or (step + 1) == len(dataloader):
                elapsed_min = (time.time() - start_time) / 60.0
                print(f"[Epoch {epoch:02d}/{epochs:02d} | Step {step+1:04d}/{len(dataloader):04d}] "
                      f"Total: {total_loss.item():.4f} (WAL: {loss_wal.item():.4f}, CIS: {loss_cis.item():.4f}) | "
                      f"Elapsed: {elapsed_min:.1f}m")

        print(f"📊 >>> Epoch {epoch} Complete. Average Loss: {np.mean(ep_losses):.4f}")

    print(f"\n🎉 SparkleNet v3.0 Ultra Training Complete in {(time.time() - start_time)/60.0:.2f} minutes!")

    # 4. Export Web Weights
    print("📦 Exporting SparkleNet v3.0 Ultra Weights Bundle...")
    export_web_weights_v3(wal_net, cis_net, out_path=output_weights)
    return output_weights


if __name__ == "__main__":
    train_sparklenet_v3_200k(epochs=12, samples_per_epoch=200000)
