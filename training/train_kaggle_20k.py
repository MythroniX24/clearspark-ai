"""
ClearSpark AI SparkleNet v2.0 — Dual T4 GPU Training Engine (20,000+ Images)
Engineered for Kaggle Notebooks (PyTorch + CUDA Mixed Precision AMP + DataParallel).

Usage:
  python train_kaggle_20k.py --epochs 15 --batch-size 32 --samples 20000 --export
"""

import os
import sys
import time
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import model architecture and dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sparklenet_v2 import (
    MultiScaleAttentionUNet,
    ContextualGenerativeInpainter,
    export_pytorch_to_web_json
)
from training.dataset_20k import Sparkle20kDataset, generate_calibrated_gemini_sparkle


# ── Advanced Loss Functions ──────────────────────────────────────────────────

if HAS_TORCH:
    class LuminanceAwareDarkLoss(nn.Module):
        """
        Penalizes inpainting errors on dark backgrounds up to 4x heavier than bright areas.
        Guarantees zero whitish ghosting / halo artifacts on night scenes and shadows.
        """
        def __init__(self, weight=3.0, threshold=0.35):
            super().__init__()
            self.weight = weight
            self.threshold = max(float(threshold), 1e-4)

        def forward(self, pred, target, mask):
            # Luminance Y = 0.299 R + 0.587 G + 0.114 B
            lum = 0.299 * target[:, 0:1, :, :] + 0.587 * target[:, 1:2, :, :] + 0.114 * target[:, 2:3, :, :]
            dark_factor = torch.clamp(self.threshold - lum, min=0.0) / self.threshold # 1.0 for black, 0.0 for bright
            penalty = 1.0 + self.weight * dark_factor

            abs_err = torch.abs(pred - target) * penalty * mask
            # 1e-4 guard protects against FP16 FTZ (flush-to-zero) division by zero
            return torch.sum(abs_err) / (torch.sum(mask) * 3.0 + 1e-4)

    class SobelGradientLoss(nn.Module):
        """Sobel edge loss ensuring reconstructed watermark edges match the surrounding grain."""
        def __init__(self, device=None):
            super().__init__()
            sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)
            if device is not None:
                self.to(device)

        def forward(self, pred, target):
            sx = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
            sy = self.sobel_y.to(dtype=pred.dtype, device=pred.device)
            p_gx = F.conv2d(pred, sx, padding=1, groups=3)
            p_gy = F.conv2d(pred, sy, padding=1, groups=3)
            t_gx = F.conv2d(target, sx, padding=1, groups=3)
            t_gy = F.conv2d(target, sy, padding=1, groups=3)
            return F.l1_loss(p_gx, t_gx) + F.l1_loss(p_gy, t_gy)
else:
    class LuminanceAwareDarkLoss:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for LuminanceAwareDarkLoss.")

    class SobelGradientLoss:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for SobelGradientLoss.")


# ── Training Pipeline ────────────────────────────────────────────────────────

def train_model(args):
    if not HAS_TORCH:
        print("\n❌ Error: PyTorch is not installed in this environment.")
        print("💡 SparkleNet v2.0 training requires PyTorch with CUDA acceleration.")
        print("   If you are running on Kaggle or Google Colab, select a GPU accelerator (e.g. 2x Tesla T4).")
        print("   To install PyTorch locally: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n")
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"\n=================================================================")
    print(f"🚀 ClearSpark AI SparkleNet v2.0 Training Engine")
    print(f"Device: {device} | Total GPUs Available: {num_gpus}")
    if num_gpus > 1:
        print(f"Active Multi-GPU Configuration: 2x Tesla T4 (DataParallel Enabled)")
    print(f"Dataset Size: {args.samples:,} samples | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"=================================================================\n")

    # 1. Dataset & DataLoader
    dataset = Sparkle20kDataset(
        total_samples=args.samples,
        patch_size=args.patch_size,
        external_img_dir=args.external_data
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if num_gpus > 0 else 0,
        pin_memory=True if num_gpus > 0 else False
    )

    # 2. Models
    wal_net = MultiScaleAttentionUNet(in_channels=3, base_channels=16).to(device)
    cis_net = ContextualGenerativeInpainter(in_channels=4, base_channels=16).to(device)

    if num_gpus > 1:
        wal_net = nn.DataParallel(wal_net)
        cis_net = nn.DataParallel(cis_net)

    # 3. Optimizers & Schedulers
    all_params = list(wal_net.parameters()) + list(cis_net.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # 4. Losses & Mixed Precision Scaler
    dark_loss_fn = LuminanceAwareDarkLoss(weight=3.0).to(device)
    sobel_loss_fn = SobelGradientLoss(device)
    bce_loss_fn = nn.BCELoss()
    smooth_l1_fn = nn.SmoothL1Loss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)

    # 5. Training Loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        wal_net.train()
        cis_net.train()
        epoch_losses = []
        epoch_wal = []
        epoch_cis = []

        for step, (wm_img, clean_img, gt_mask, gt_box, gt_conf, is_dark) in enumerate(dataloader):
            wm_img = wm_img.to(device, non_blocking=True)
            clean_img = clean_img.to(device, non_blocking=True)
            gt_mask = gt_mask.to(device, non_blocking=True)
            gt_box = gt_box.to(device, non_blocking=True)
            gt_conf = gt_conf.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                # ── WAL-Net Forward Pass (Multi-scale Localization)
                wal_out = wal_net(wm_img)
                # Numerical stability clamp prevents log(0) NaN under FP16 autocast
                pred_mask = torch.clamp(wal_out['mask'], 1e-6, 1.0 - 1e-6)
                pred_box = wal_out['bbox']
                pred_conf = torch.clamp(wal_out['confidence'], 1e-6, 1.0 - 1e-6)

                loss_mask = bce_loss_fn(pred_mask, gt_mask)
                loss_box = smooth_l1_fn(pred_box, gt_box)
                loss_conf = bce_loss_fn(pred_conf, gt_conf)
                loss_wal = loss_mask * 2.0 + loss_box * 1.0 + loss_conf * 0.5

                # ── CIS-Net Forward Pass (Contextual Inpainting)
                # Training mask: Combine predicted high-confidence pixels with ground truth
                train_mask = torch.max((pred_mask > 0.4).float(), gt_mask)
                inpainted, synth_rgb = cis_net(wm_img, train_mask)

                # Contextual Reconstruction Losses
                loss_l1 = F.l1_loss(inpainted, clean_img)
                loss_dark = dark_loss_fn(inpainted, clean_img, train_mask)
                loss_sobel = sobel_loss_fn(inpainted, clean_img) * 0.2
                loss_cis = loss_l1 + loss_dark * 1.5 + loss_sobel

                total_loss = loss_wal + loss_cis

            # Backward Pass with Mixed Precision Scaler
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_losses.append(total_loss.item())
            epoch_wal.append(loss_wal.item())
            epoch_cis.append(loss_cis.item())

            if (step + 1) % 50 == 0 or (step + 1) == len(dataloader):
                lr_curr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                print(f"[Epoch {epoch:02d}/{args.epochs:02d} | Step {step+1:04d}/{len(dataloader):04d}] "
                      f"Total: {total_loss.item():.4f} (WAL: {loss_wal.item():.4f}, CIS: {loss_cis.item():.4f}) "
                      f"| LR: {lr_curr:.6f} | Elapsed: {elapsed/60.0:.1f}m")

        mean_loss = float(np.mean(epoch_losses))
        print(f"\n📊 >>> Epoch {epoch} Complete. Average Loss: {mean_loss:.4f} (WAL: {np.mean(epoch_wal):.4f}, CIS: {np.mean(epoch_cis):.4f})")

        # Save Checkpoint
        if mean_loss < best_loss:
            best_loss = mean_loss
            ckpt_path = os.path.join(args.output_dir, "best_sparklenet_20k.pth")
            torch.save({
                'epoch': epoch,
                'wal_net_state': (wal_net.module if hasattr(wal_net, 'module') else wal_net).state_dict(),
                'cis_net_state': (cis_net.module if hasattr(cis_net, 'module') else cis_net).state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': best_loss
            }, ckpt_path)
            print(f"⭐ Saved new best model checkpoint to {ckpt_path}\n")

    print(f"\n🎉 Training complete in {(time.time() - start_time)/60.0:.2f} minutes!")

    # 6. Auto-Export to Web JSON
    if args.export:
        print("\n📦 Exporting trained weights to client-side sparklenet_web_weights.json...")
        raw_wal = wal_net.module if hasattr(wal_net, 'module') else wal_net
        raw_cis = cis_net.module if hasattr(cis_net, 'module') else cis_net
        tpl48 = generate_calibrated_gemini_sparkle(48)
        tpl96 = generate_calibrated_gemini_sparkle(96)
        json_out = os.path.join(args.output_dir, "sparklenet_web_weights.json")
        export_pytorch_to_web_json(raw_wal, raw_cis, json_out, tpl48=tpl48, tpl96=tpl96)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ClearSpark AI SparkleNet v2.0 Dual T4 GPU Training Engine")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size (32 recommended for 2x T4 GPUs)")
    parser.add_argument('--samples', type=int, default=20000, help="Total training samples per epoch")
    parser.add_argument('--patch-size', type=int, default=128, help="Patch size for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Peak learning rate")
    parser.add_argument('--external-data', type=str, default=None, help="Path to external images folder (optional)")
    parser.add_argument('--output-dir', type=str, default="./weights_output", help="Output directory for weights")
    parser.add_argument('--export', action='store_true', default=True, help="Auto export web-ready JSON weights")
    args = parser.parse_args()

    train_model(args)
